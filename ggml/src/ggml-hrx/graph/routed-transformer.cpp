#include "routed-transformer.h"

#include <algorithm>
#include <queue>
#include <set>

namespace ggml::hrx {
namespace {

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

static std::vector<OperationId> set_vector(const std::set<OperationId> & values) {
    return { values.begin(), values.end() };
}

static std::set<OperationId> difference(const std::set<OperationId> & lhs, const std::set<OperationId> & rhs) {
    std::set<OperationId> result;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(result, result.end()));
    return result;
}

static void add_component(RoutedTransformerBlock & block, std::string role, OperationId hero,
                          const std::set<OperationId> & operations) {
    if (operations.empty()) return;
    RoutedTransformerComponent component;
    component.role = std::move(role);
    component.hero = hero;
    component.operations = set_vector(operations);
    block.components.push_back(std::move(component));
}

static bool observe(FactDatabase & facts, const std::string & key, int64_t value,
                    const std::string & source, uint32_t graph_id, Decision & failure) {
    failure = facts.observe(key, value, { source, graph_id });
    return failure.allowed;
}

struct RoutedCandidatePayload final : CandidatePayload {
    size_t block_ordinal = 0;
    std::string component_role;
    bool decode = false;
    bool terminal = false;
    std::vector<OperationId> following_prepare_operations;
    std::vector<OperationId> endpoint_operations;
};

} // namespace

RoutedTransformerModel analyze_routed_transformer(const GraphIndex & index) {
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

        const OperationId flash_reshape = unique_consumer(index, flash_op.output, GGML_OP_RESHAPE,
                                                           model.errors, "attention result reshape");
        if (flash_reshape == kInvalidId) return model;
        const OperationId attention_projection = unique_consumer(
            index, graph.operations[flash_reshape].output, GGML_OP_MUL_MAT,
            model.errors, "attention output projection");
        if (attention_projection == kInvalidId) return model;
        const OperationId attention_residual = trace_layout_consumer(
            index, graph.operations[attention_projection].output, GGML_OP_ADD,
            model.errors, "attention residual");
        if (attention_residual == kInvalidId) return model;
        const Operation & residual_op = graph.operations[attention_residual];
        if (residual_op.inputs.size() != 2) {
            model.errors.push_back("attention residual is not binary");
            return model;
        }
        const bool first_is_attention = find_ancestor(graph, residual_op.inputs[0], GGML_OP_MUL_MAT) == attention_projection;
        ValueId hidden_input = first_is_attention ? residual_op.inputs[1] : residual_op.inputs[0];
        const OperationId hidden_adapter = producer(graph, hidden_input);
        if (hidden_adapter != kInvalidId && graph.operations[hidden_adapter].op == GGML_OP_GET_ROWS &&
            !graph.operations[hidden_adapter].inputs.empty() &&
            producer(graph, graph.operations[hidden_adapter].inputs[0]) != kInvalidId) {
            hidden_input = graph.operations[hidden_adapter].inputs[0];
        }
        const OperationId hidden_input_producer = producer(graph, hidden_input);

        const OperationId feed_forward_norm = unique_consumer(
            index, residual_op.output, GGML_OP_RMS_NORM, model.errors, "feed-forward normalization");
        if (feed_forward_norm == kInvalidId) return model;
        const OperationId feed_forward_prepared = unique_consumer(
            index, graph.operations[feed_forward_norm].output, GGML_OP_MUL,
            model.errors, "feed-forward prepared input");
        if (feed_forward_prepared == kInvalidId) return model;
        const OperationId router_projection = unique_consumer(
            index, graph.operations[feed_forward_prepared].output, GGML_OP_MUL_MAT,
            model.errors, "router projection");
        if (router_projection == kInvalidId) return model;
        const OperationId router_softmax = unique_consumer(
            index, graph.operations[router_projection].output, GGML_OP_SOFT_MAX,
            model.errors, "router softmax");
        if (router_softmax == kInvalidId) return model;
        const OperationId router_argsort = unique_consumer(
            index, graph.operations[router_softmax].output, GGML_OP_ARGSORT,
            model.errors, "router argsort");
        if (router_argsort == kInvalidId) return model;
        const OperationId route_ids = unique_consumer(
            index, graph.operations[router_argsort].output, GGML_OP_VIEW,
            model.errors, "route identifiers");
        if (route_ids == kInvalidId) return model;

        OperationId gate_up = kInvalidId;
        for (const Operation & operation : graph.operations) {
            if (operation.op != GGML_OP_GLU || operation.inputs.size() < 2) continue;
            const OperationId gate = producer(graph, operation.inputs[0]);
            const OperationId up = producer(graph, operation.inputs[1]);
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
        const OperationId routed_down = unique_consumer(
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

        const std::set<OperationId> block_set = block_closure(index, final_residual, hidden_input_producer);
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
        block.operations = set_vector(block_set);

        const OperationId query_projection = find_ancestor(graph, flash_op.inputs[0], GGML_OP_MUL_MAT);
        if (query_projection == kInvalidId || graph.operations[query_projection].inputs.size() < 2) {
            model.errors.push_back("cannot recover query projection from flash-attention hero");
            return model;
        }
        const ValueId attention_prepared_value = graph.operations[query_projection].inputs[1];
        const OperationId attention_prepared = producer(graph, attention_prepared_value);
        if (attention_prepared == kInvalidId || graph.operations[attention_prepared].op != GGML_OP_MUL) {
            model.errors.push_back("query projection input is not a prepared attention row");
            return model;
        }
        const OperationId attention_norm = producer(graph, graph.operations[attention_prepared].inputs[0]);
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

        OperationId query_rope = find_ancestor(graph, flash_op.inputs[0], GGML_OP_ROPE);
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
            const OperationId projection = find_ancestor(graph, writer_op.inputs[0], GGML_OP_MUL_MAT);
            const OperationId rope = find_primary_ancestor_before_matmul(graph, writer_op.inputs[0], GGML_OP_ROPE);
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

        const OperationId gate_projection = producer(graph, graph.operations[gate_up].inputs[0]);
        const OperationId up_projection = producer(graph, graph.operations[gate_up].inputs[1]);
        if (gate_projection == kInvalidId || up_projection == kInvalidId) {
            model.errors.push_back("gate/up hero has no routed projection producers");
            return model;
        }

        OperationId route_weights = kInvalidId;
        for (OperationId operation : block_set) {
            if (graph.operations[operation].op != GGML_OP_RESHAPE) continue;
            const auto & inputs = graph.operations[operation].inputs;
            if (!inputs.empty()) {
                const OperationId input_producer = producer(graph, inputs[0]);
                if (input_producer != kInvalidId && graph.operations[input_producer].op == GGML_OP_DIV) route_weights = operation;
            }
        }
        if (route_weights == kInvalidId) {
            model.errors.push_back("routed transformer block has no normalized route weights");
            return model;
        }

        block.bindings.operations = {
            { "attention.norm", attention_norm }, { "attention.prepared", attention_prepared },
            { "attention.query_projection", query_projection }, { "attention.flash", flash },
            { "attention.key_projection", key_projection }, { "attention.value_projection", value_projection },
            { "attention.query_rope", query_rope }, { "attention.key_cache_writer", key_writer },
            { "attention.value_cache_writer", value_writer }, { "attention.result_reshape", flash_reshape },
            { "attention.output_projection", attention_projection }, { "attention.residual", attention_residual },
            { "feed_forward.norm", feed_forward_norm }, { "feed_forward.prepared", feed_forward_prepared },
            { "router.projection", router_projection }, { "router.softmax", router_softmax },
            { "router.argsort", router_argsort }, { "router.route_ids", route_ids },
            { "router.route_weights", route_weights }, { "experts.gate_up", gate_up },
            { "experts.gate_projection", gate_projection }, { "experts.up_projection", up_projection },
            { "experts.routed_down", routed_down }, { "hidden.output", final_residual },
        };
        block.bindings.values = {
            { "hidden.input", hidden_input }, { "attention.prepared", attention_prepared_value },
            { "attention.result", graph.operations[flash_reshape].output },
            { "attention.residual", residual_op.output },
            { "feed_forward.prepared", graph.operations[feed_forward_prepared].output },
            { "router.logits", graph.operations[router_projection].output },
            { "router.route_ids", graph.operations[route_ids].output },
            { "router.route_weights", graph.operations[route_weights].output },
            { "experts.activation", graph.operations[gate_up].output },
            { "experts.down", graph.operations[routed_down].output },
            { "hidden.output", graph.operations[final_residual].output },
        };

        const std::set<OperationId> prepare = ancestors_within(index, { attention_prepared }, block_set);
        std::vector<OperationId> publication_roots = cache_writers;
        publication_roots.push_back(query_rope);
        const std::set<OperationId> qkv_cumulative = ancestors_within(index, publication_roots, block_set);
        const std::set<OperationId> qkv = difference(qkv_cumulative, prepare);
        const std::set<OperationId> flash_cumulative = ancestors_within(index, { flash, flash_reshape }, block_set);
        std::set<OperationId> claimed = prepare;
        claimed.insert(qkv.begin(), qkv.end());
        const std::set<OperationId> attention = difference(flash_cumulative, claimed);
        const std::set<OperationId> output_cumulative = ancestors_within(index, { feed_forward_prepared }, block_set);
        claimed.insert(attention.begin(), attention.end());
        // Recompute after including the attention component.
        const std::set<OperationId> output_component = difference(output_cumulative, claimed);
        claimed.insert(output_component.begin(), output_component.end());
        const std::set<OperationId> router_cumulative = ancestors_within(index, { route_ids, route_weights }, block_set);
        const std::set<OperationId> router = difference(router_cumulative, claimed);
        claimed.insert(router.begin(), router.end());
        const std::set<OperationId> gate_cumulative = ancestors_within(index, { gate_up }, block_set);
        const std::set<OperationId> gate = difference(gate_cumulative, claimed);
        claimed.insert(gate.begin(), gate.end());
        const std::set<OperationId> down = difference(block_set, claimed);

        add_component(block, "attention.prepare", attention_prepared, prepare);
        add_component(block, "attention.qkv_publication", query_projection, qkv);
        add_component(block, "attention.flash", flash, attention);
        add_component(block, "attention.output_prepare", attention_projection, output_component);
        add_component(block, "router.selection", router_projection, router);
        add_component(block, "experts.gate_up", gate_up, gate);
        add_component(block, "experts.down_publication", routed_down, down);
        model.blocks.push_back(std::move(block));
    }

    std::set<OperationId> endpoint_set;
    std::queue<OperationId> endpoint_worklist;
    const OperationId final_block_output = model.blocks.back().bindings.operations.at("hidden.output");
    for (OperationId successor : index.successors(final_block_output)) {
        if (all_block_operations.count(successor) == 0 && endpoint_set.insert(successor).second) endpoint_worklist.push(successor);
    }
    while (!endpoint_worklist.empty()) {
        const OperationId current = endpoint_worklist.front();
        endpoint_worklist.pop();
        for (OperationId successor : index.successors(current)) {
            if (all_block_operations.count(successor) == 0 && endpoint_set.insert(successor).second) endpoint_worklist.push(successor);
        }
    }
    model.endpoint_operations = set_vector(endpoint_set);
    for (const Operation & operation : graph.operations) {
        if (all_block_operations.count(operation.id) == 0 && endpoint_set.count(operation.id) == 0) {
            model.preamble_operations.push_back(operation.id);
        }
    }
    if (!model.preamble_operations.empty()) {
        const OperationId embedding = model.preamble_operations.front();
        model.bindings.operations["program.embedding"] = embedding;
        model.bindings.values["program.hidden_state"] = graph.operations[embedding].output;
    }
    for (OperationId operation : model.endpoint_operations) {
        switch (graph.operations[operation].op) {
            case GGML_OP_RMS_NORM: model.bindings.operations["endpoint.norm"] = operation; break;
            case GGML_OP_MUL: model.bindings.operations["endpoint.prepared"] = operation; break;
            case GGML_OP_MUL_MAT: model.bindings.operations["endpoint.projection"] = operation; break;
            default: break;
        }
    }

    const RoutedTransformerBlock & first = model.blocks.front();
    const Value & prepared = graph.values[first.bindings.values.at("attention.prepared")];
    model.hidden_size = prepared.access.shape[0];
    model.query_token_count = prepared.access.shape[1];
    const Operation & first_flash = graph.operations[first.bindings.operations.at("attention.flash")];
    model.key_value_token_count = graph.values[first_flash.inputs[1]].access.shape[1];
    const Operation & first_query = graph.operations[first.bindings.operations.at("attention.query_projection")];
    model.query_size = graph.values[first_query.output].access.shape[0];
    model.key_value_size = 0;
    for (OperationId projection : index.consumers(first.bindings.values.at("attention.prepared"))) {
        if (graph.operations[projection].op != GGML_OP_MUL_MAT || projection == first_query.id) continue;
        const int64_t width = graph.values[graph.operations[projection].output].access.shape[0];
        if (model.key_value_size == 0 || width < model.key_value_size) model.key_value_size = width;
    }
    model.expert_count = graph.values[graph.operations[first.bindings.operations.at("router.projection")].output].access.shape[0];
    model.route_count = graph.values[first.bindings.values.at("router.route_ids")].access.shape[0];
    model.output_token_count = 1;
    for (ValueId root : graph.roots) {
        if (root < graph.values.size()) model.output_token_count = std::max<int64_t>(model.output_token_count, graph.values[root].access.shape[1]);
    }
    return model;
}

Decision RoutedTransformerProvider::discover(const GraphIndex & index, FactDatabase & facts) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) {
        return Decision::reject(DecisionReason::ProviderError,
                                "supplied routed-transformer analysis belongs to another graph");
    }
    RoutedTransformerModel recovered_model;
    const RoutedTransformerModel & model = supplied_model_ ? *supplied_model_
        : (recovered_model = analyze_routed_transformer(index));
    if (!model.valid()) {
        return Decision::reject(DecisionReason::ProviderError,
                                model.errors.empty() ? "routed transformer analysis failed" : model.errors.front());
    }
    Decision failure;
    const uint32_t hero = model.blocks.front().bindings.operations.at("attention.flash");
    if (!observe(facts, "llm.layer_count", model.blocks.size(), "routed blocks", hero, failure) ||
        !observe(facts, "llm.query_token_count", model.query_token_count, "attention input", hero, failure) ||
        !observe(facts, "llm.output_token_count", model.output_token_count, "graph roots", hero, failure) ||
        !observe(facts, "llm.key_value_token_count", model.key_value_token_count, "flash key", hero, failure) ||
        !observe(facts, "llm.hidden_size", model.hidden_size, "attention input", hero, failure) ||
        !observe(facts, "llm.query_size", model.query_size, "query projection", hero, failure) ||
        !observe(facts, "llm.key_value_size", model.key_value_size, "key/value projections", hero, failure) ||
        !observe(facts, "llm.expert_count", model.expert_count, "router projection", hero, failure) ||
        !observe(facts, "llm.route_count", model.route_count, "router selection", hero, failure)) return failure;

    // Every repeated block independently witnesses the global geometry.
    const Graph & graph = index.graph();
    for (const RoutedTransformerBlock & block : model.blocks) {
        const OperationId block_hero = block.bindings.operations.at("attention.flash");
        const Value & block_prepared = graph.values[block.bindings.values.at("attention.prepared")];
        if (!observe(facts, "llm.hidden_size", block_prepared.access.shape[0], "block attention input", block_hero, failure) ||
            !observe(facts, "llm.query_token_count", block_prepared.access.shape[1], "block attention input", block_hero, failure)) {
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
        : (recovered_model = analyze_routed_transformer(index));
    if (!model.valid()) return;
    const bool decode = model.query_token_count == 1;
    auto dispatches_for = [&](const std::string & role) -> int {
        if (decode) {
            if (role == "attention.prepare") return 0;
            if (role == "attention.qkv_publication") return 2;
            if (role == "attention.flash") return 1;
            if (role == "attention.output_prepare") return 3;
            if (role == "router.selection") return 2;
            if (role == "experts.gate_up") return 2;
            if (role == "experts.down_publication") return 2;
        } else {
            if (role == "attention.prepare") return 1;
            if (role == "attention.qkv_publication") return 4;
            if (role == "attention.flash") return 1;
            if (role == "attention.output_prepare") return 2;
            if (role == "router.selection") return 4;
            if (role == "experts.gate_up") return 1;
            if (role == "experts.down_publication") return 2;
        }
        return 0;
    };
    auto append = [&](const std::string & family, const std::string & key, OperationId hero,
                      const std::vector<OperationId> & operations, const SemanticBindings & bindings,
                      int planned_dispatches, bool correctness_baseline = false) {
        FusionCandidate candidate;
        candidate.provider = id();
        candidate.family = family;
        candidate.key = std::string(id()) + ':' + key;
        candidate.hero = hero;
        candidate.operations = operations;
        candidate.materialized_outputs = index.boundary(operations).outputs;
        candidate.allow_disconnected = true;
        candidate.correctness_baseline = correctness_baseline;
        candidate.bindings = bindings;
        candidate.economics.reference_dispatches = planned_dispatches;
        candidate.economics.planned_dispatches = planned_dispatches;
        candidates.push_back(std::move(candidate));
    };
    if (!model.preamble_operations.empty()) {
        append("program.preamble", "preamble", model.preamble_operations.back(), model.preamble_operations, {},
               decode ? 3 : 2, true);
    }
    for (const RoutedTransformerBlock & block : model.blocks) {
        for (const RoutedTransformerComponent & component : block.components) {
            append(component.role, "block." + std::to_string(block.ordinal) + '.' + component.role,
                   component.hero, component.operations, block.bindings, 1);
            auto payload = std::make_shared<RoutedCandidatePayload>();
            payload->block_ordinal = block.ordinal;
            payload->component_role = component.role;
            payload->decode = decode;
            payload->terminal = block.ordinal + 1 == model.blocks.size();
            if (!payload->terminal) {
                payload->following_prepare_operations = model.blocks[block.ordinal + 1].components.front().operations;
            } else {
                payload->endpoint_operations = model.endpoint_operations;
            }
            candidates.back().payload = std::move(payload);
            candidates.back().economics.reference_dispatches = dispatches_for(component.role);
            candidates.back().economics.planned_dispatches = dispatches_for(component.role);
            candidates.back().correctness_baseline = true;
        }
    }
    if (!model.endpoint_operations.empty()) {
        append("program.endpoint", "endpoint", model.endpoint_operations.front(), model.endpoint_operations, {},
               decode ? 1 : 2, true);
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
    const std::string & role = payload->component_role;
    if (payload->decode) {
        if (role == "attention.qkv_publication") alternative(routed_transformer_recipes::kDecodeQkvPostprocess, 1);
        if (role == "attention.output_prepare") alternative(routed_transformer_recipes::kDecodeOutputNextQ8, 2);
        if (role == "router.selection") alternative(routed_transformer_recipes::kDecodeRouterTopK, 1);
        if (role == "experts.gate_up") alternative(routed_transformer_recipes::kDecodeGateUpNextQ8, 1);
        if (role == "experts.down_publication" && catalog_.contains(routed_transformer_recipes::kDecodeDownNextQ8)) {
            std::vector<OperationId> grown = candidate.operations;
            if (!payload->terminal) {
                grown = union_operations(grown, payload->following_prepare_operations);
            } else {
                grown = union_operations(grown, payload->endpoint_operations);
            }
            alternative(routed_transformer_recipes::kDecodeDownNextQ8, 1, std::move(grown));
            // The grown candidate replaces the neighboring zero/one-dispatch
            // baseline as well. Include that cost in its comparison.
            FusionCandidate & result = expansions.back();
            if (payload->terminal) {
                result.economics.reference_dispatches += 1;
                result.economics.planned_dispatches += 1; // endpoint projection remains separate
            }
        }
    } else {
        if (role == "router.selection" && !payload->terminal) {
            alternative(routed_transformer_recipes::kPrefillExpertPartition, 3);
        }
        if (role == "experts.down_publication" && !payload->terminal &&
            catalog_.contains(routed_transformer_recipes::kPrefillDownNextNorm)) {
            std::vector<OperationId> grown = union_operations(
                candidate.operations, payload->following_prepare_operations);
            alternative(routed_transformer_recipes::kPrefillDownNextNorm, 2, std::move(grown));
            expansions.back().economics.reference_dispatches = 3;
        }
    }
}

PlannerConfiguration make_structural_routed_transformer_planner(
    RoutedTransformerRecipeCatalog catalog,
    std::shared_ptr<const RoutedTransformerModel> supplied_model) {
    PlannerConfiguration configuration;
    configuration.add_provider(std::make_shared<RoutedTransformerProvider>(
        std::move(catalog), std::move(supplied_model)));
    return configuration;
}

} // namespace ggml::hrx
