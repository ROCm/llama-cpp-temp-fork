#include "routed-transformer.h"
#include "schedule.h"

#include <algorithm>
#include <queue>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

class RoutedTransformerAnalysisImplementation {
public:

static OperationId producer(const Graph & graph, ValueId value) {
    return value < graph.values.size() ? graph.values[value].producer : kInvalidId;
}

static OperationId unique_consumer(const GraphIndex & index, ValueId value, enum ggml_op kind,
                                   std::vector<std::string> & errors, const std::string & role) {
    OperationId result = kInvalidId;
    for (OperationId candidate : index.consumers(value)) {
        if (index.graph().operations[candidate].op != kind) continue;
        if (result != kInvalidId) {
            errors.push_back("routed transformer role " + role + " has multiple consumers");
            return kInvalidId;
        }
        result = candidate;
    }
    if (result == kInvalidId) errors.push_back("routed transformer role " + role + " is absent");
    return result;
}

static OperationId trace_layout_consumer(const GraphIndex & index, ValueId value, enum ggml_op target,
                                         std::vector<std::string> & errors, const std::string & role) {
    ValueId current = value;
    std::set<ValueId> visited;
    while (visited.insert(current).second) {
        OperationId found = kInvalidId;
        OperationId layout = kInvalidId;
        for (OperationId candidate : index.consumers(current)) {
            const enum ggml_op op = index.graph().operations[candidate].op;
            if (op == target) {
                if (found != kInvalidId) {
                    errors.push_back("routed transformer role " + role + " is ambiguous");
                    return kInvalidId;
                }
                found = candidate;
            } else if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
                       op == GGML_OP_TRANSPOSE || op == GGML_OP_GET_ROWS) {
                if (layout == kInvalidId) layout = candidate;
            }
        }
        if (found != kInvalidId) return found;
        if (layout == kInvalidId) break;
        current = index.graph().operations[layout].output;
    }
    errors.push_back("routed transformer role " + role + " is absent");
    return kInvalidId;
}

static OperationId find_ancestor(const Graph & graph, ValueId value, enum ggml_op kind) {
    std::queue<ValueId> worklist;
    std::set<ValueId> visited;
    worklist.push(value);
    visited.insert(value);
    while (!worklist.empty()) {
        const ValueId current = worklist.front();
        worklist.pop();
        const OperationId operation = producer(graph, current);
        if (operation == kInvalidId) continue;
        if (graph.operations[operation].op == kind) return operation;
        for (ValueId input : graph.operations[operation].inputs) {
            if (visited.insert(input).second) worklist.push(input);
        }
    }
    return kInvalidId;
}

static OperationId find_primary_ancestor_before_matmul(const Graph & graph, ValueId value, enum ggml_op kind) {
    std::set<ValueId> visited;
    while (visited.insert(value).second) {
        const OperationId operation = producer(graph, value);
        if (operation == kInvalidId) return kInvalidId;
        if (graph.operations[operation].op == kind) return operation;
        if (graph.operations[operation].op == GGML_OP_MUL_MAT || graph.operations[operation].inputs.empty()) {
            return kInvalidId;
        }
        value = graph.operations[operation].inputs[0];
    }
    return kInvalidId;
}

static std::set<OperationId> ancestors_within(const GraphIndex & index,
                                              const std::vector<OperationId> & roots,
                                              const std::set<OperationId> & allowed) {
    std::set<OperationId> result;
    std::queue<OperationId> worklist;
    for (OperationId root : roots) {
        if (allowed.count(root) != 0 && result.insert(root).second) worklist.push(root);
    }
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (allowed.count(predecessor) != 0 && result.insert(predecessor).second) worklist.push(predecessor);
        }
    }
    return result;
}

static std::set<OperationId> block_closure(const GraphIndex & index, OperationId root, OperationId stop) {
    std::set<OperationId> result;
    std::queue<OperationId> worklist;
    result.insert(root);
    worklist.push(root);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (predecessor == stop) continue;
            if (result.insert(predecessor).second) worklist.push(predecessor);
        }
    }
    return result;
}

static bool has_ancestor_in(const GraphIndex & index, OperationId root,
                            const std::set<OperationId> & candidates) {
    std::set<OperationId> visited;
    std::queue<OperationId> worklist;
    visited.insert(root);
    worklist.push(root);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (candidates.count(predecessor) != 0) return true;
            if (visited.insert(predecessor).second) worklist.push(predecessor);
        }
    }
    return false;
}

static std::vector<OperationId> set_vector(const std::set<OperationId> & values) {
    return { values.begin(), values.end() };
}

static std::set<OperationId> difference(const std::set<OperationId> & lhs, const std::set<OperationId> & rhs) {
    std::set<OperationId> result;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(result, result.end()));
    return result;
}

static void add_component(RoutedTransformerBlock & block, RoutedTransformerComponentKind kind, OperationId hero,
                          const std::set<OperationId> & operations) {
    if (operations.empty()) return;
    RoutedTransformerComponent component;
    component.kind = kind;
    component.hero = hero;
    component.operations = set_vector(operations);
    block.components.push_back(std::move(component));
}

static bool observe(FactDatabase & facts, const std::string & key, int64_t value,
                    const std::string & source, uint32_t graph_id, Decision & failure) {
    failure = facts.observe(key, value, { source, graph_id });
    return failure.allowed;
}

};

struct RoutedCandidatePayload final : CandidatePayload {
    size_t block_ordinal = 0;
    RoutedTransformerComponentKind component_kind = RoutedTransformerComponentKind::Atom;
    bool decode = false;
    bool terminal = false;
    std::vector<OperationId> following_prepare_operations;
    std::vector<OperationId> endpoint_operations;
};

} // namespace

RoutedTransformerModel RoutedTransformerModel::analyze(const GraphIndex & index) {
    RoutedTransformerModel model;
    model.graph_fingerprint = index.graph().fingerprint;
    if (!index.valid()) {
        model.errors = index.errors();
        return model;
    }
    const Graph & graph = index.graph();
    std::vector<OperationId> flash_operations;
    for (const Operation & operation : graph.operations) {
        if (operation.op == GGML_OP_FLASH_ATTN_EXT) flash_operations.push_back(operation.id);
    }
    if (flash_operations.empty()) {
        model.errors.push_back("graph contains no flash-attention heroes");
        return model;
    }

    std::set<OperationId> all_block_operations;
    for (OperationId flash : flash_operations) {
        RoutedTransformerBlock block;
        block.ordinal = model.blocks.size();
        const Operation & flash_op = graph.operations[flash];
        if (flash_op.inputs.size() < 3) {
            model.errors.push_back("flash-attention hero has fewer than three tensor inputs");
            return model;
        }

        const OperationId flash_reshape = RoutedTransformerAnalysisImplementation::unique_consumer(index, flash_op.output, GGML_OP_RESHAPE,
                                                           model.errors, "attention result reshape");
        if (flash_reshape == kInvalidId) return model;
        const OperationId attention_projection = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[flash_reshape].output, GGML_OP_MUL_MAT,
            model.errors, "attention output projection");
        if (attention_projection == kInvalidId) return model;
        const OperationId attention_residual = RoutedTransformerAnalysisImplementation::trace_layout_consumer(
            index, graph.operations[attention_projection].output, GGML_OP_ADD,
            model.errors, "attention residual");
        if (attention_residual == kInvalidId) return model;
        const Operation & residual_op = graph.operations[attention_residual];
        if (residual_op.inputs.size() != 2) {
            model.errors.push_back("attention residual is not binary");
            return model;
        }
        const bool first_is_attention = RoutedTransformerAnalysisImplementation::find_ancestor(graph, residual_op.inputs[0], GGML_OP_MUL_MAT) == attention_projection;
        const ValueId attention_input = first_is_attention ? residual_op.inputs[0] : residual_op.inputs[1];
        ValueId hidden_input = first_is_attention ? residual_op.inputs[1] : residual_op.inputs[0];
        const OperationId attention_adapter = RoutedTransformerAnalysisImplementation::producer(graph, attention_input);
        const OperationId hidden_adapter = RoutedTransformerAnalysisImplementation::producer(graph, hidden_input);
        const bool selects_attention = attention_adapter != kInvalidId &&
            graph.operations[attention_adapter].op == GGML_OP_GET_ROWS &&
            !graph.operations[attention_adapter].inputs.empty() &&
            RoutedTransformerAnalysisImplementation::producer(
                graph, graph.operations[attention_adapter].inputs[0]) != kInvalidId;
        const bool selects_hidden = hidden_adapter != kInvalidId &&
            graph.operations[hidden_adapter].op == GGML_OP_GET_ROWS &&
            !graph.operations[hidden_adapter].inputs.empty() &&
            RoutedTransformerAnalysisImplementation::producer(
                graph, graph.operations[hidden_adapter].inputs[0]) != kInvalidId;
        if (selects_attention != selects_hidden) {
            model.errors.push_back("attention residual selects only one of its two inputs in block " +
                                   std::to_string(block.ordinal));
            return model;
        }
        if (selects_attention) {
            const Operation & attention_selection = graph.operations[attention_adapter];
            const Operation & hidden_selection = graph.operations[hidden_adapter];
            if (attention_selection.inputs.size() != 2 || hidden_selection.inputs.size() != 2 ||
                attention_selection.inputs[1] != hidden_selection.inputs[1]) {
                model.errors.push_back("attention residual selections do not share one output-ID tensor in block " +
                                       std::to_string(block.ordinal));
                return model;
            }
        }
        if (selects_hidden) {
            hidden_input = graph.operations[hidden_adapter].inputs[0];
        }
        const OperationId hidden_input_producer = RoutedTransformerAnalysisImplementation::producer(graph, hidden_input);

        const OperationId feed_forward_norm = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, residual_op.output, GGML_OP_RMS_NORM, model.errors, "feed-forward normalization");
        if (feed_forward_norm == kInvalidId) return model;
        const OperationId feed_forward_prepared = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[feed_forward_norm].output, GGML_OP_MUL,
            model.errors, "feed-forward prepared input");
        if (feed_forward_prepared == kInvalidId) return model;
        const OperationId router_projection = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[feed_forward_prepared].output, GGML_OP_MUL_MAT,
            model.errors, "router projection");
        if (router_projection == kInvalidId) return model;
        const OperationId router_softmax = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[router_projection].output, GGML_OP_SOFT_MAX,
            model.errors, "router softmax");
        if (router_softmax == kInvalidId) return model;
        const OperationId router_argsort = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[router_softmax].output, GGML_OP_ARGSORT,
            model.errors, "router argsort");
        if (router_argsort == kInvalidId) return model;
        const OperationId route_ids = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[router_argsort].output, GGML_OP_VIEW,
            model.errors, "route identifiers");
        if (route_ids == kInvalidId) return model;

        OperationId gate_up = kInvalidId;
        for (const Operation & operation : graph.operations) {
            if (operation.op != GGML_OP_GLU || operation.inputs.size() < 2) continue;
            const OperationId gate = RoutedTransformerAnalysisImplementation::producer(graph, operation.inputs[0]);
            const OperationId up = RoutedTransformerAnalysisImplementation::producer(graph, operation.inputs[1]);
            if (gate == kInvalidId || up == kInvalidId || graph.operations[gate].op != GGML_OP_MUL_MAT_ID ||
                graph.operations[up].op != GGML_OP_MUL_MAT_ID) continue;
            const auto consumes_routes = [&](OperationId projection) {
                const auto & inputs = graph.operations[projection].inputs;
                return std::find(inputs.begin(), inputs.end(), graph.operations[route_ids].output) != inputs.end();
            };
            if (consumes_routes(gate) && consumes_routes(up)) {
                if (gate_up != kInvalidId) {
                    model.errors.push_back("routed transformer block has ambiguous gate/up hero");
                    return model;
                }
                gate_up = operation.id;
            }
        }
        if (gate_up == kInvalidId) {
            model.errors.push_back("routed transformer block has no gate/up hero");
            return model;
        }
        const OperationId routed_down = RoutedTransformerAnalysisImplementation::unique_consumer(
            index, graph.operations[gate_up].output, GGML_OP_MUL_MAT_ID,
            model.errors, "routed down projection");
        if (routed_down == kInvalidId) return model;

        OperationId final_residual = kInvalidId;
        for (OperationId consumer : index.consumers(residual_op.output)) {
            if (graph.operations[consumer].op == GGML_OP_ADD) {
                if (final_residual != kInvalidId) {
                    model.errors.push_back("routed transformer block has ambiguous final residual");
                    return model;
                }
                final_residual = consumer;
            }
        }
        if (final_residual == kInvalidId) {
            model.errors.push_back("routed transformer block has no final residual");
            return model;
        }

        const std::set<OperationId> block_set = RoutedTransformerAnalysisImplementation::block_closure(index, final_residual, hidden_input_producer);
        if (block_set.count(flash) == 0 || block_set.count(gate_up) == 0 || block_set.count(routed_down) == 0) {
            model.errors.push_back("routed transformer block closure does not contain its heroes");
            return model;
        }
        for (OperationId operation : block_set) {
            if (!all_block_operations.insert(operation).second) {
                model.errors.push_back("routed transformer block closures overlap");
                return model;
            }
        }
        block.operations = RoutedTransformerAnalysisImplementation::set_vector(block_set);

        const OperationId query_projection = RoutedTransformerAnalysisImplementation::find_ancestor(graph, flash_op.inputs[0], GGML_OP_MUL_MAT);
        if (query_projection == kInvalidId || graph.operations[query_projection].inputs.size() < 2) {
            model.errors.push_back("cannot recover query projection from flash-attention hero");
            return model;
        }
        const ValueId attention_prepared_value = graph.operations[query_projection].inputs[1];
        const OperationId attention_prepared = RoutedTransformerAnalysisImplementation::producer(graph, attention_prepared_value);
        if (attention_prepared == kInvalidId || graph.operations[attention_prepared].op != GGML_OP_MUL) {
            model.errors.push_back("query projection input is not a prepared attention row");
            return model;
        }
        const OperationId attention_norm = RoutedTransformerAnalysisImplementation::producer(graph, graph.operations[attention_prepared].inputs[0]);
        if (attention_norm == kInvalidId || graph.operations[attention_norm].op != GGML_OP_RMS_NORM) {
            model.errors.push_back("prepared attention row has no RMSNorm producer");
            return model;
        }
        std::vector<OperationId> qkv_projections;
        for (OperationId consumer : index.consumers(attention_prepared_value)) {
            if (graph.operations[consumer].op == GGML_OP_MUL_MAT) qkv_projections.push_back(consumer);
        }
        if (qkv_projections.size() != 3) {
            model.errors.push_back("prepared attention row does not feed exactly three projections");
            return model;
        }

        OperationId query_rope = RoutedTransformerAnalysisImplementation::find_ancestor(graph, flash_op.inputs[0], GGML_OP_ROPE);
        if (query_rope == kInvalidId) {
            model.errors.push_back("cannot recover query RoPE role");
            return model;
        }
        std::vector<OperationId> cache_writers;
        for (OperationId operation : block_set) {
            if (graph.operations[operation].op == GGML_OP_SET_ROWS) cache_writers.push_back(operation);
        }
        if (cache_writers.size() != 2) {
            model.errors.push_back("routed transformer block does not contain two cache writers");
            return model;
        }

        OperationId key_projection = kInvalidId;
        OperationId value_projection = kInvalidId;
        OperationId key_writer = kInvalidId;
        OperationId value_writer = kInvalidId;
        for (OperationId writer : cache_writers) {
            const Operation & writer_op = graph.operations[writer];
            if (writer_op.inputs.empty()) continue;
            const OperationId projection = RoutedTransformerAnalysisImplementation::find_ancestor(graph, writer_op.inputs[0], GGML_OP_MUL_MAT);
            const OperationId rope = RoutedTransformerAnalysisImplementation::find_primary_ancestor_before_matmul(graph, writer_op.inputs[0], GGML_OP_ROPE);
            if (projection == kInvalidId) continue;
            if (rope != kInvalidId) {
                key_projection = projection;
                key_writer = writer;
            } else {
                value_projection = projection;
                value_writer = writer;
            }
        }
        if (key_projection == kInvalidId || value_projection == kInvalidId) {
            model.errors.push_back("cannot distinguish key and value projection paths in block " +
                                   std::to_string(block.ordinal) + " writers=" +
                                   std::to_string(cache_writers[0]) + ',' + std::to_string(cache_writers[1]) +
                                   " key=" + std::to_string(key_projection) +
                                   " value=" + std::to_string(value_projection));
            return model;
        }

        const OperationId gate_projection = RoutedTransformerAnalysisImplementation::producer(graph, graph.operations[gate_up].inputs[0]);
        const OperationId up_projection = RoutedTransformerAnalysisImplementation::producer(graph, graph.operations[gate_up].inputs[1]);
        if (gate_projection == kInvalidId || up_projection == kInvalidId) {
            model.errors.push_back("gate/up hero has no routed projection producers");
            return model;
        }

        OperationId route_weights = kInvalidId;
        for (OperationId operation : block_set) {
            if (graph.operations[operation].op != GGML_OP_RESHAPE) continue;
            const auto & inputs = graph.operations[operation].inputs;
            if (!inputs.empty()) {
                const OperationId input_producer = RoutedTransformerAnalysisImplementation::producer(graph, inputs[0]);
                if (input_producer != kInvalidId && graph.operations[input_producer].op == GGML_OP_DIV) route_weights = operation;
            }
        }
        if (route_weights == kInvalidId) {
            model.errors.push_back("routed transformer block has no normalized route weights");
            return model;
        }

        block.operations_by_role.attention_norm = attention_norm;
        block.operations_by_role.attention_prepared = attention_prepared;
        block.operations_by_role.attention_query_projection = query_projection;
        block.operations_by_role.attention_key_projection = key_projection;
        block.operations_by_role.attention_value_projection = value_projection;
        block.operations_by_role.attention_query_rope = query_rope;
        block.operations_by_role.attention_key_cache_writer = key_writer;
        block.operations_by_role.attention_value_cache_writer = value_writer;
        block.operations_by_role.attention_flash = flash;
        block.operations_by_role.attention_result_reshape = flash_reshape;
        block.operations_by_role.attention_output_projection = attention_projection;
        block.operations_by_role.attention_output_selection =
            selects_attention
                ? attention_adapter : kInvalidId;
        block.operations_by_role.hidden_state_selection =
            selects_hidden
                ? hidden_adapter : kInvalidId;
        block.operations_by_role.attention_residual = attention_residual;
        block.operations_by_role.feed_forward_prepared = feed_forward_prepared;
        block.operations_by_role.router_projection = router_projection;
        block.operations_by_role.router_route_ids = route_ids;
        block.operations_by_role.router_route_weights = route_weights;
        block.operations_by_role.experts_gate_projection = gate_projection;
        block.operations_by_role.experts_up_projection = up_projection;
        block.operations_by_role.experts_gate_up = gate_up;
        block.operations_by_role.experts_routed_down = routed_down;
        block.operations_by_role.hidden_output = final_residual;
        block.values_by_role.attention_prepared = attention_prepared_value;
        block.values_by_role.router_route_ids = graph.operations[route_ids].output;
        block.values_by_role.experts_activation = graph.operations[gate_up].output;

        const std::set<OperationId> prepare = RoutedTransformerAnalysisImplementation::ancestors_within(index, { attention_prepared }, block_set);
        std::vector<OperationId> publication_roots = cache_writers;
        publication_roots.push_back(query_rope);
        const std::set<OperationId> qkv_cumulative = RoutedTransformerAnalysisImplementation::ancestors_within(index, publication_roots, block_set);
        const std::set<OperationId> qkv = RoutedTransformerAnalysisImplementation::difference(qkv_cumulative, prepare);
        const std::set<OperationId> flash_cumulative = RoutedTransformerAnalysisImplementation::ancestors_within(index, { flash, flash_reshape }, block_set);
        std::set<OperationId> claimed = prepare;
        claimed.insert(qkv.begin(), qkv.end());
        const std::set<OperationId> attention = RoutedTransformerAnalysisImplementation::difference(flash_cumulative, claimed);
        const std::set<OperationId> output_cumulative = RoutedTransformerAnalysisImplementation::ancestors_within(index, { feed_forward_prepared }, block_set);
        claimed.insert(attention.begin(), attention.end());
        // Recompute after including the attention component.
        const std::set<OperationId> output_component = RoutedTransformerAnalysisImplementation::difference(output_cumulative, claimed);
        claimed.insert(output_component.begin(), output_component.end());
        const std::set<OperationId> router_cumulative = RoutedTransformerAnalysisImplementation::ancestors_within(index, { route_ids, route_weights }, block_set);
        const std::set<OperationId> router = RoutedTransformerAnalysisImplementation::difference(router_cumulative, claimed);
        claimed.insert(router.begin(), router.end());
        const std::set<OperationId> gate_cumulative = RoutedTransformerAnalysisImplementation::ancestors_within(index, { gate_up }, block_set);
        const std::set<OperationId> gate = RoutedTransformerAnalysisImplementation::difference(gate_cumulative, claimed);
        claimed.insert(gate.begin(), gate.end());
        const std::set<OperationId> down = RoutedTransformerAnalysisImplementation::difference(block_set, claimed);

        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::AttentionPrepare, attention_prepared, prepare);
        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::AttentionQkvPublication, query_projection, qkv);
        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::Attention, flash, attention);
        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::AttentionOutputPrepare, attention_projection, output_component);
        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::RouterSelection, router_projection, router);
        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::ExpertGateUp, gate_up, gate);
        RoutedTransformerAnalysisImplementation::add_component(block, RoutedTransformerComponentKind::ExpertDownPublication, routed_down, down);
        model.blocks.push_back(std::move(block));
    }

    std::set<OperationId> endpoint_set;
    std::queue<OperationId> endpoint_worklist;
    for (ValueId root : graph.roots) {
        const OperationId producer = graph.values[root].producer;
        // Debug and auxiliary consumers may materialize values at arbitrary
        // block boundaries. Only roots downstream of the routed block body
        // belong to the program endpoint; an upstream root such as the token
        // embedding is part of the preamble, and a block-owned root remains
        // owned by that block.
        if (producer != kInvalidId && all_block_operations.count(producer) == 0 &&
            RoutedTransformerAnalysisImplementation::has_ancestor_in(index, producer, all_block_operations) &&
            endpoint_set.insert(producer).second) {
            endpoint_worklist.push(producer);
        }
    }
    while (!endpoint_worklist.empty()) {
        const OperationId current = endpoint_worklist.front();
        endpoint_worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (all_block_operations.count(predecessor) == 0 && endpoint_set.insert(predecessor).second) {
                endpoint_worklist.push(predecessor);
            }
        }
    }
    model.endpoint_operations = RoutedTransformerAnalysisImplementation::set_vector(endpoint_set);

    // Preamble and endpoint are dataflow slices, not positional graph
    // leftovers. This distinction is what permits an unfamiliar side/tail op
    // to remain an explicit atom instead of being accidentally swallowed by
    // a broad model-sized region.
    std::set<OperationId> preamble_set;
    std::queue<OperationId> preamble_worklist;
    for (OperationId block_operation : all_block_operations) {
        for (OperationId predecessor : index.predecessors(block_operation)) {
            if (all_block_operations.count(predecessor) == 0 && endpoint_set.count(predecessor) == 0 &&
                preamble_set.insert(predecessor).second) {
                preamble_worklist.push(predecessor);
            }
        }
    }
    while (!preamble_worklist.empty()) {
        const OperationId current = preamble_worklist.front();
        preamble_worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (all_block_operations.count(predecessor) == 0 && endpoint_set.count(predecessor) == 0 &&
                preamble_set.insert(predecessor).second) {
                preamble_worklist.push(predecessor);
            }
        }
    }
    model.preamble_operations = RoutedTransformerAnalysisImplementation::set_vector(preamble_set);
    if (!model.preamble_operations.empty()) {
        const OperationId embedding = model.preamble_operations.front();
        model.operations_by_role.program_embedding = embedding;
        model.values_by_role.program_hidden_state = graph.operations[embedding].output;
    }
    for (OperationId operation : model.endpoint_operations) {
        switch (graph.operations[operation].op) {
            case GGML_OP_RMS_NORM:
                model.operations_by_role.endpoint_norm = operation;
                break;
            case GGML_OP_MUL:
                model.operations_by_role.endpoint_prepared = operation;
                break;
            case GGML_OP_MUL_MAT:
                model.operations_by_role.endpoint_projection = operation;
                break;
            default: break;
        }
    }

    const RoutedTransformerBlock & first = model.blocks.front();
    const Value & prepared = graph.values[first.values_by_role.attention_prepared];
    model.hidden_size = prepared.access.shape[0];
    model.query_token_count = prepared.access.shape[1];
    const Operation & first_flash = graph.operations[first.operations_by_role.attention_flash];
    model.key_value_token_count = graph.values[first_flash.inputs[1]].access.shape[1];
    const Operation & first_query = graph.operations[first.operations_by_role.attention_query_projection];
    model.query_size = graph.values[first_query.output].access.shape[0];
    model.key_value_size = 0;
    for (OperationId projection : index.consumers(first.values_by_role.attention_prepared)) {
        if (graph.operations[projection].op != GGML_OP_MUL_MAT || projection == first_query.id) continue;
        const int64_t width = graph.values[graph.operations[projection].output].access.shape[0];
        if (model.key_value_size == 0 || width < model.key_value_size) model.key_value_size = width;
    }
    model.expert_count = graph.values[graph.operations[first.operations_by_role.router_projection].output].access.shape[0];
    model.route_count = graph.values[first.values_by_role.router_route_ids].access.shape[0];
    model.output_token_count = 1;
    for (ValueId root : graph.roots) {
        if (root < graph.values.size()) model.output_token_count = std::max<int64_t>(model.output_token_count, graph.values[root].access.shape[1]);
    }
    LogicalComponentId next_component = 0;
    model.preamble.id = next_component++;
    model.preamble.kind = RoutedTransformerComponentKind::ProgramPreamble;
    model.preamble.hero = model.preamble_operations.empty() ? kInvalidId : model.preamble_operations.back();
    model.preamble.operations = model.preamble_operations;
    model.preamble.boundary = index.boundary(model.preamble.operations);
    for (RoutedTransformerBlock & block : model.blocks) {
        for (RoutedTransformerComponent & component : block.components) {
            component.id = next_component++;
            component.boundary = index.boundary(component.operations);
        }
    }
    model.endpoint.id = next_component++;
    model.endpoint.kind = RoutedTransformerComponentKind::ProgramEndpoint;
    model.endpoint.hero = model.endpoint_operations.empty() ? kInvalidId : model.endpoint_operations.front();
    model.endpoint.operations = model.endpoint_operations;
    model.endpoint.boundary = index.boundary(model.endpoint.operations);

    std::vector<uint8_t> raised(graph.operations.size(), 0);
    auto mark_raised = [&](const RoutedTransformerComponent & component) {
        for (OperationId operation : component.operations) {
            if (operation < raised.size()) ++raised[operation];
        }
    };
    mark_raised(model.preamble);
    for (const RoutedTransformerBlock & block : model.blocks) {
        for (const RoutedTransformerComponent & component : block.components) mark_raised(component);
    }
    mark_raised(model.endpoint);
    for (OperationId operation = 0; operation < raised.size(); ++operation) {
        if (raised[operation] == 0) {
            model.unraised_operations.push_back(operation);
            RoutedTransformerComponent fallback;
            fallback.id = next_component++;
            fallback.kind = RoutedTransformerComponentKind::Atom;
            fallback.hero = operation;
            fallback.operations = { operation };
            fallback.boundary = index.boundary(fallback.operations);
            model.fallback_components.push_back(std::move(fallback));
        }
        else if (raised[operation] != 1) model.errors.push_back(
            "logical operation " + std::to_string(operation) + " is owned by more than one component");
    }
    const VerificationResult verification = RoutedTransformerModel::verify(index, model);
    model.errors.insert(model.errors.end(), verification.errors.begin(), verification.errors.end());
    return model;
}

const char * RoutedTransformerModel::component_kind_name(RoutedTransformerComponentKind kind) {
    switch (kind) {
        case RoutedTransformerComponentKind::ProgramPreamble: return "program.preamble";
        case RoutedTransformerComponentKind::AttentionPrepare: return "attention.prepare";
        case RoutedTransformerComponentKind::AttentionQkvPublication: return "attention.qkv_publication";
        case RoutedTransformerComponentKind::Attention: return "attention.flash";
        case RoutedTransformerComponentKind::AttentionOutputPrepare: return "attention.output_prepare";
        case RoutedTransformerComponentKind::RouterSelection: return "router.selection";
        case RoutedTransformerComponentKind::ExpertGateUp: return "experts.gate_up";
        case RoutedTransformerComponentKind::ExpertDownPublication: return "experts.down_publication";
        case RoutedTransformerComponentKind::ProgramEndpoint: return "program.endpoint";
        case RoutedTransformerComponentKind::Atom: return "atom";
    }
    return "unknown";
}

std::string RoutedTransformerModel::format(const RoutedTransformerModel & model) {
    std::ostringstream out;
    out << "schema=ggml-hrx-logical-routed-transformer-v1\n"
        << "graph=" << model.graph_fingerprint << '\n'
        << "blocks=" << model.blocks.size() << '\n'
        << "query_tokens=" << model.query_token_count << '\n'
        << "output_tokens=" << model.output_token_count << '\n'
        << "key_value_tokens=" << model.key_value_token_count << '\n'
        << "hidden_size=" << model.hidden_size << '\n'
        << "query_size=" << model.query_size << '\n'
        << "key_value_size=" << model.key_value_size << '\n'
        << "experts=" << model.expert_count << '\n'
        << "routes=" << model.route_count << '\n';
    auto component = [&](const RoutedTransformerComponent & value, int64_t block) {
        out << "component " << value.id << " kind=" << component_kind_name(value.kind)
            << " block=" << block << " hero=" << value.hero << " ops=";
        for (OperationId operation : value.operations) out << operation << ',';
        out << " inputs=";
        for (ValueId input : value.boundary.inputs) out << input << ',';
        out << " outputs=";
        for (ValueId output : value.boundary.outputs) out << output << ',';
        out << '\n';
    };
    component(model.preamble, -1);
    for (const RoutedTransformerBlock & block : model.blocks) {
        for (const RoutedTransformerComponent & value : block.components) component(value, block.ordinal);
    }
    component(model.endpoint, -1);
    for (const RoutedTransformerComponent & fallback : model.fallback_components) component(fallback, -1);
    if (!model.unraised_operations.empty()) {
        out << "unraised=";
        for (OperationId operation : model.unraised_operations) out << operation << ',';
        out << '\n';
    }
    for (const std::string & error : model.errors) out << "error=" << error << '\n';
    return out.str();
}

std::string RoutedTransformerModel::serialize_json(const RoutedTransformerModel & model) {
    auto escape = [](const std::string & value) {
        std::string result;
        for (char character : value) {
            if (character == '"' || character == '\\') result.push_back('\\');
            result.push_back(character);
        }
        return result;
    };
    std::ostringstream out;
    out << "{\"version\":1,\"graph_fingerprint\":\"" << escape(model.graph_fingerprint)
        << "\",\"facts\":{\"block_count\":" << model.blocks.size()
        << ",\"query_token_count\":" << model.query_token_count
        << ",\"output_token_count\":" << model.output_token_count
        << ",\"key_value_token_count\":" << model.key_value_token_count
        << ",\"hidden_size\":" << model.hidden_size
        << ",\"query_size\":" << model.query_size
        << ",\"key_value_size\":" << model.key_value_size
        << ",\"expert_count\":" << model.expert_count
        << ",\"route_count\":" << model.route_count << "},\"components\":[";
    bool first = true;
    auto component = [&](const RoutedTransformerComponent & value, int64_t block) {
        if (!first) out << ',';
        first = false;
        out << "{\"id\":" << value.id << ",\"kind\":\"" << component_kind_name(value.kind)
            << "\",\"block\":" << block << ",\"hero\":" << value.hero << ",\"operations\":[";
        for (size_t i = 0; i < value.operations.size(); ++i) {
            if (i) out << ',';
            out << value.operations[i];
        }
        out << "],\"inputs\":[";
        for (size_t i = 0; i < value.boundary.inputs.size(); ++i) {
            if (i) out << ',';
            out << value.boundary.inputs[i];
        }
        out << "],\"outputs\":[";
        for (size_t i = 0; i < value.boundary.outputs.size(); ++i) {
            if (i) out << ',';
            out << value.boundary.outputs[i];
        }
        out << "]}";
    };
    component(model.preamble, -1);
    for (const RoutedTransformerBlock & block : model.blocks) {
        for (const RoutedTransformerComponent & value : block.components) component(value, block.ordinal);
    }
    component(model.endpoint, -1);
    for (const RoutedTransformerComponent & fallback : model.fallback_components) component(fallback, -1);
    out << "],\"unraised_operations\":[";
    for (size_t i = 0; i < model.unraised_operations.size(); ++i) {
        if (i) out << ',';
        out << model.unraised_operations[i];
    }
    out << "]}";
    return out.str();
}

std::string RoutedTransformerModel::dot(const RoutedTransformerModel & model) {
    std::ostringstream out;
    out << "digraph logical_program {\n  rankdir=LR;\n";
    std::vector<const RoutedTransformerComponent *> ordered { &model.preamble };
    for (const RoutedTransformerBlock & block : model.blocks) {
        out << "  subgraph cluster_block_" << block.ordinal << " { label=\"block " << block.ordinal << "\";\n";
        for (const RoutedTransformerComponent & component : block.components) {
            ordered.push_back(&component);
            out << "    c" << component.id << " [label=\"" << component_kind_name(component.kind)
                << "\\nops=" << component.operations.size() << "\"];\n";
        }
        out << "  }\n";
    }
    ordered.push_back(&model.endpoint);
    for (const RoutedTransformerComponent & fallback : model.fallback_components) ordered.push_back(&fallback);
    out << "  c" << model.preamble.id << " [label=\"" << component_kind_name(model.preamble.kind) << "\"];\n"
        << "  c" << model.endpoint.id << " [label=\"" << component_kind_name(model.endpoint.kind) << "\"];\n";
    for (size_t i = 1; i < ordered.size(); ++i) out << "  c" << ordered[i - 1]->id << " -> c" << ordered[i]->id << ";\n";
    out << "}\n";
    return out.str();
}

VerificationResult RoutedTransformerModel::verify(const GraphIndex & index, const RoutedTransformerModel & model) {
    VerificationResult result;
    const Graph & graph = index.graph();
    std::vector<uint8_t> owners(graph.operations.size(), 0);
    std::vector<uint8_t> fallback_owners(graph.operations.size(), 0);
    LogicalComponentId expected_id = 0;
    auto verify_component = [&](const RoutedTransformerComponent & component) {
        const std::string name = RoutedTransformerModel::component_kind_name(component.kind);
        if (component.id != expected_id++) {
            result.errors.push_back("logical component " + std::string(name) + " has a non-canonical id");
        }
        if (component.operations.empty()) {
            result.errors.push_back("logical component " + std::string(name) + " owns no operations");
            return;
        }
        for (OperationId operation : component.operations) {
            if (operation >= owners.size()) {
                result.errors.push_back("logical component " + std::string(name) + " owns an invalid operation");
            } else if (++owners[operation] != 1) {
                result.errors.push_back("logical operation " + std::to_string(operation) + " has duplicate ownership");
            }
        }
        const RegionBoundary expected = index.boundary(component.operations);
        if (component.boundary.inputs != expected.inputs || component.boundary.outputs != expected.outputs) {
            result.errors.push_back("logical component " + std::string(name) + " has a stale graph boundary");
        }
        const Decision legality = index.validate_region(component.operations, component.boundary.outputs, true);
        if (!legality.allowed) result.errors.push_back(
            "logical component " + std::string(name) + " is illegal: " + legality.detail);
    };
    verify_component(model.preamble);
    for (const RoutedTransformerBlock & block : model.blocks) {
        if (block.components.size() != 7) {
            result.errors.push_back("routed transformer block " + std::to_string(block.ordinal) +
                                    " does not contain seven logical components");
        }
        const std::string prefix = "routed transformer block " + std::to_string(block.ordinal) + ' ';
        const Value & prepared = graph.values[block.values_by_role.attention_prepared];
        if (prepared.access.shape[0] != model.hidden_size || prepared.access.shape[1] != model.query_token_count) {
            result.errors.push_back(prefix + "attention input disagrees with recovered model geometry");
        }
        const Operation & flash = graph.operations[block.operations_by_role.attention_flash];
        if (flash.inputs.size() != 4) {
            result.errors.push_back(prefix + "flash attention does not have query, key, value, and mask operands");
        } else {
            const Value & query = graph.values[flash.inputs[0]];
            const Value & key = graph.values[flash.inputs[1]];
            const Value & value = graph.values[flash.inputs[2]];
            const Value & mask = graph.values[flash.inputs[3]];
            if (query.access.shape[1] != model.query_token_count ||
                key.access.shape[1] != model.key_value_token_count ||
                value.access.shape[1] != model.key_value_token_count ||
                mask.access.shape[0] != model.key_value_token_count ||
                mask.access.shape[1] != model.query_token_count) {
                result.errors.push_back(prefix + "attention operands disagree on query/key-value token geometry");
            }
        }
        const Value & routes = graph.values[block.values_by_role.router_route_ids];
        const Value & router = graph.values[graph.operations[block.operations_by_role.router_projection].output];
        if (routes.access.shape[0] != model.route_count || router.access.shape[0] != model.expert_count) {
            result.errors.push_back(prefix + "router operands disagree with recovered model geometry");
        }
        for (const RoutedTransformerComponent & component : block.components) verify_component(component);
    }
    verify_component(model.endpoint);
    for (const RoutedTransformerComponent & fallback : model.fallback_components) {
        verify_component(fallback);
        for (OperationId operation : fallback.operations) {
            if (operation < fallback_owners.size()) ++fallback_owners[operation];
        }
    }
    for (OperationId operation = 0; operation < owners.size(); ++operation) {
        const bool declared_unraised = std::find(model.unraised_operations.begin(), model.unraised_operations.end(), operation) !=
            model.unraised_operations.end();
        if (owners[operation] == 0) {
            result.errors.push_back("logical operation " + std::to_string(operation) + " has no owner");
        }
        if (declared_unraised != (fallback_owners[operation] == 1)) {
            result.errors.push_back("logical operation " + std::to_string(operation) +
                                    " has inconsistent atom-fallback ownership");
        }
    }
    return result;
}

Decision RoutedTransformerProvider::discover(const GraphIndex & index, FactDatabase & facts) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) {
        return Decision::reject(DecisionReason::ProviderError,
                                "supplied routed-transformer analysis belongs to another graph");
    }
    RoutedTransformerModel recovered_model;
    const RoutedTransformerModel & model = supplied_model_ ? *supplied_model_
        : (recovered_model = RoutedTransformerModel::analyze(index));
    if (!model.valid()) {
        return Decision::reject(DecisionReason::ProviderError,
                                model.errors.empty() ? "routed transformer analysis failed" : model.errors.front());
    }
    Decision failure;
    const uint32_t hero = model.blocks.front().operations_by_role.attention_flash;
    if (!RoutedTransformerAnalysisImplementation::observe(facts, "llm.layer_count", model.blocks.size(), "routed blocks", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.query_token_count", model.query_token_count, "attention input", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.output_token_count", model.output_token_count, "graph roots", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.key_value_token_count", model.key_value_token_count, "flash key", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.hidden_size", model.hidden_size, "attention input", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.query_size", model.query_size, "query projection", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.key_value_size", model.key_value_size, "key/value projections", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.expert_count", model.expert_count, "router projection", hero, failure) ||
        !RoutedTransformerAnalysisImplementation::observe(facts, "llm.route_count", model.route_count, "router selection", hero, failure)) return failure;

    // Every repeated block independently witnesses the global geometry.
    const Graph & graph = index.graph();
    for (const RoutedTransformerBlock & block : model.blocks) {
        const OperationId block_hero = block.operations_by_role.attention_flash;
        const Value & block_prepared = graph.values[block.values_by_role.attention_prepared];
        if (!RoutedTransformerAnalysisImplementation::observe(facts, "llm.hidden_size", block_prepared.access.shape[0], "block attention input", block_hero, failure) ||
            !RoutedTransformerAnalysisImplementation::observe(facts, "llm.query_token_count", block_prepared.access.shape[1], "block attention input", block_hero, failure)) {
            return failure;
        }
    }
    return Decision::allow();
}

void RoutedTransformerProvider::seed(const GraphIndex & index, const FactDatabase &,
                                     std::vector<FusionCandidate> & candidates) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) return;
    RoutedTransformerModel recovered_model;
    const RoutedTransformerModel & model = supplied_model_ ? *supplied_model_
        : (recovered_model = RoutedTransformerModel::analyze(index));
    if (!model.valid()) return;
    const bool decode = model.query_token_count == 1;
    auto dispatches_for = [&](RoutedTransformerComponentKind kind) -> int {
        if (decode) {
            if (kind == RoutedTransformerComponentKind::AttentionPrepare) return 0;
            if (kind == RoutedTransformerComponentKind::AttentionQkvPublication) return 2;
            if (kind == RoutedTransformerComponentKind::Attention) return 1;
            if (kind == RoutedTransformerComponentKind::AttentionOutputPrepare) return 3;
            if (kind == RoutedTransformerComponentKind::RouterSelection) return 2;
            if (kind == RoutedTransformerComponentKind::ExpertGateUp) return 2;
            if (kind == RoutedTransformerComponentKind::ExpertDownPublication) return 2;
        } else {
            if (kind == RoutedTransformerComponentKind::AttentionPrepare) return 1;
            if (kind == RoutedTransformerComponentKind::AttentionQkvPublication) return 4;
            if (kind == RoutedTransformerComponentKind::Attention) return 1;
            if (kind == RoutedTransformerComponentKind::AttentionOutputPrepare) return 2;
            if (kind == RoutedTransformerComponentKind::RouterSelection) return 4;
            if (kind == RoutedTransformerComponentKind::ExpertGateUp) return 1;
            if (kind == RoutedTransformerComponentKind::ExpertDownPublication) return 2;
        }
        return 0;
    };
    auto append = [&](const std::string & family, const std::string & key, LogicalComponentId component_id, OperationId hero,
                      const std::vector<OperationId> & operations,
                      int planned_dispatches, bool correctness_baseline = false) {
        FusionCandidate candidate;
        candidate.provider = id();
        candidate.family = family;
        candidate.key = std::string(id()) + ':' + key;
        candidate.hero = hero;
        candidate.logical_components.push_back(component_id);
        candidate.operations = operations;
        candidate.materialized_outputs = index.boundary(operations).outputs;
        candidate.allow_disconnected = true;
        candidate.correctness_baseline = correctness_baseline;
        candidate.economics.reference_dispatches = planned_dispatches;
        candidate.economics.planned_dispatches = planned_dispatches;
        candidates.push_back(std::move(candidate));
    };
    if (!model.preamble_operations.empty()) {
        append("program.preamble", "preamble", model.preamble.id, model.preamble_operations.back(), model.preamble_operations,
               decode ? 3 : 2, true);
    }
    for (const RoutedTransformerBlock & block : model.blocks) {
        for (const RoutedTransformerComponent & component : block.components) {
            const std::string role = RoutedTransformerModel::component_kind_name(component.kind);
            append(role, "block." + std::to_string(block.ordinal) + '.' + role, component.id,
                   component.hero, component.operations, 1);
            auto payload = std::make_shared<RoutedCandidatePayload>();
            payload->block_ordinal = block.ordinal;
            payload->component_kind = component.kind;
            payload->decode = decode;
            payload->terminal = block.ordinal + 1 == model.blocks.size();
            if (!payload->terminal) {
                payload->following_prepare_operations = model.blocks[block.ordinal + 1].components.front().operations;
            } else {
                payload->endpoint_operations = model.endpoint_operations;
            }
            candidates.back().payload = std::move(payload);
            candidates.back().economics.reference_dispatches = dispatches_for(component.kind);
            candidates.back().economics.planned_dispatches = dispatches_for(component.kind);
            candidates.back().correctness_baseline = true;
        }
    }
    if (!model.endpoint_operations.empty()) {
        append("program.endpoint", "endpoint", model.endpoint.id, model.endpoint_operations.front(), model.endpoint_operations,
               decode ? 1 : 2, true);
    }
    for (const RoutedTransformerComponent & fallback : model.fallback_components) {
        const std::string operation = ggml_op_name(index.graph().operations[fallback.hero].op);
        append("atom." + operation, "atom." + std::to_string(fallback.hero), fallback.id, fallback.hero,
               fallback.operations, 1, true);
        candidates.back().allow_disconnected = false;
    }
}

void RoutedTransformerProvider::expand(const GraphIndex & index, const FactDatabase &,
                                       const FusionCandidate & candidate,
                                       std::vector<FusionCandidate> & expansions) const {
    const auto payload = std::dynamic_pointer_cast<const RoutedCandidatePayload>(candidate.payload);
    if (!payload) return;
    const size_t block_ordinal = payload->block_ordinal;

    auto alternative = [&](const char * recipe, int planned_dispatches,
                           std::vector<OperationId> operations = {}) {
        if (!catalog_.contains(recipe)) return;
        FusionCandidate result = candidate;
        result.family = recipe;
        result.key = std::string(id()) + ':' + recipe + ".block." + std::to_string(block_ordinal);
        if (!operations.empty()) {
            result.operations = std::move(operations);
            result.materialized_outputs = index.boundary(result.operations).outputs;
        }
        result.correctness_baseline = false;
        result.payload.reset();
        result.economics.reference_dispatches = candidate.economics.planned_dispatches;
        result.economics.planned_dispatches = planned_dispatches;
        expansions.push_back(std::move(result));
    };
    auto union_operations = [](const std::vector<OperationId> & lhs, const std::vector<OperationId> & rhs) {
        std::vector<OperationId> result = lhs;
        result.insert(result.end(), rhs.begin(), rhs.end());
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    };
    const RoutedTransformerComponentKind kind = payload->component_kind;
    if (payload->decode) {
        if (kind == RoutedTransformerComponentKind::AttentionQkvPublication) alternative(routed_transformer_recipes::kDecodeQkvPostprocess, 1);
        if (kind == RoutedTransformerComponentKind::AttentionOutputPrepare) alternative(routed_transformer_recipes::kDecodeOutputNextQ8, 2);
        if (kind == RoutedTransformerComponentKind::RouterSelection) alternative(routed_transformer_recipes::kDecodeRouterTopK, 1);
        if (kind == RoutedTransformerComponentKind::ExpertGateUp) alternative(routed_transformer_recipes::kDecodeGateUpNextQ8, 1);
        if (kind == RoutedTransformerComponentKind::ExpertDownPublication && catalog_.contains(routed_transformer_recipes::kDecodeDownNextQ8)) {
            std::vector<OperationId> grown = candidate.operations;
            if (!payload->terminal) {
                grown = union_operations(grown, payload->following_prepare_operations);
            } else {
                grown = union_operations(grown, payload->endpoint_operations);
            }
            alternative(routed_transformer_recipes::kDecodeDownNextQ8, 1, std::move(grown));
            expansions.back().logical_components.push_back(candidate.logical_components.front() + 1);
            // The grown candidate replaces the neighboring zero/one-dispatch
            // baseline as well. Include that cost in its comparison.
            FusionCandidate & result = expansions.back();
            if (payload->terminal) {
                result.economics.reference_dispatches += 1;
                result.economics.planned_dispatches += 1; // endpoint projection remains separate
            }
        }
    } else {
        if (kind == RoutedTransformerComponentKind::RouterSelection && !payload->terminal) {
            alternative(routed_transformer_recipes::kPrefillExpertPartition, 3);
        }
        if (kind == RoutedTransformerComponentKind::ExpertDownPublication && !payload->terminal &&
            catalog_.contains(routed_transformer_recipes::kPrefillDownNextNorm)) {
            std::vector<OperationId> grown = union_operations(
                candidate.operations, payload->following_prepare_operations);
            alternative(routed_transformer_recipes::kPrefillDownNextNorm, 2, std::move(grown));
            expansions.back().logical_components.push_back(candidate.logical_components.front() + 1);
            expansions.back().economics.reference_dispatches = 3;
        }
    }
}

PlannerConfiguration RoutedTransformerProvider::make_planner(
    RoutedTransformerRecipeCatalog catalog,
    std::shared_ptr<const RoutedTransformerModel> supplied_model) {
    PlannerConfiguration configuration;
    configuration.add_provider(std::make_shared<RoutedTransformerProvider>(
        std::move(catalog), std::move(supplied_model)));
    return configuration;
}

} // namespace ggml::hrx
