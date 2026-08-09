#include "hybrid-carrier-transformer.h"

#include "hybrid-carrier-recipes.h"
#include "hybrid-carrier-transformer-bindings.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

struct HybridCandidatePayload final : CandidatePayload {
    int32_t block = -1;
    HybridCarrierComponentKind component = HybridCarrierComponentKind::Atom;
};

bool metadata_only(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
        op == GGML_OP_TRANSPOSE;
}

OperationId producer(const Graph & graph, ValueId value) {
    return value < graph.values.size() ? graph.values[value].producer : kInvalidId;
}

std::set<OperationId> block_closure(const GraphIndex & index, OperationId root, OperationId stop) {
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

std::set<OperationId> data_closure(const Graph & graph, OperationId root, OperationId stop) {
    std::set<OperationId> result;
    std::queue<OperationId> worklist;
    result.insert(root);
    worklist.push(root);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (ValueId input : graph.operations[current].inputs) {
            const OperationId predecessor = producer(graph, input);
            if (predecessor == kInvalidId || predecessor == stop) continue;
            if (result.insert(predecessor).second) worklist.push(predecessor);
        }
    }
    return result;
}

std::set<OperationId> data_frontier_ancestors(
        const Graph & graph, OperationId root, enum ggml_op kind) {
    std::set<OperationId> result;
    std::set<OperationId> visited;
    std::queue<OperationId> worklist;
    worklist.push(root);
    visited.insert(root);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (ValueId input : graph.operations[current].inputs) {
            const OperationId predecessor = producer(graph, input);
            if (predecessor == kInvalidId || !visited.insert(predecessor).second) continue;
            if (graph.operations[predecessor].op == kind) result.insert(predecessor);
            else worklist.push(predecessor);
        }
    }
    return result;
}

std::set<OperationId> ancestors_within(const GraphIndex & index,
                                       const std::vector<OperationId> & roots,
                                       const std::set<OperationId> & allowed) {
    std::set<OperationId> result;
    std::queue<OperationId> worklist;
    for (OperationId root : roots) {
        if (root != kInvalidId && allowed.count(root) != 0 && result.insert(root).second) worklist.push(root);
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

std::set<OperationId> difference(const std::set<OperationId> & lhs, const std::set<OperationId> & rhs) {
    std::set<OperationId> result;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(result, result.end()));
    return result;
}

std::vector<OperationId> to_vector(const std::set<OperationId> & operations) {
    return { operations.begin(), operations.end() };
}

OperationId nearest_ancestor(const Graph & graph, ValueId value, enum ggml_op kind) {
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

bool has_ancestor_in(const GraphIndex & index, OperationId root, const std::set<OperationId> & candidates) {
    std::queue<OperationId> worklist;
    std::set<OperationId> visited;
    worklist.push(root);
    visited.insert(root);
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

void add_component(HybridCarrierBlock & block, HybridCarrierComponentKind kind, OperationId hero,
                   const std::set<OperationId> & operations) {
    if (operations.empty()) return;
    HybridCarrierComponent component;
    component.kind = kind;
    component.hero = hero;
    component.operations = to_vector(operations);
    block.components.push_back(std::move(component));
}

bool observe(FactDatabase & facts, const std::string & key, int64_t value,
             const std::string & source, uint32_t graph_id, Decision & failure) {
    failure = facts.observe(key, value, { source, graph_id });
    return failure.allowed;
}

std::string escape_json(const std::string & value) {
    std::string result;
    for (char character : value) {
        if (character == '"' || character == '\\') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

} // namespace

HybridCarrierTransformerModel HybridCarrierTransformerModel::analyze(const GraphIndex & index) {
    HybridCarrierTransformerModel model;
    model.graph_fingerprint = index.graph().fingerprint;
    if (!index.valid()) {
        model.errors = index.errors();
        return model;
    }

    const Graph & graph = index.graph();
    std::vector<OperationId> post_operations;
    for (const Operation & operation : graph.operations) {
        if (operation.op == GGML_OP_DSV4_HC_POST) post_operations.push_back(operation.id);
    }
    if (post_operations.empty()) {
        model.errors.push_back("graph contains no hybrid-carrier publication heroes");
        return model;
    }

    std::set<OperationId> paired_posts;
    std::set<OperationId> all_block_operations;
    for (OperationId final_post : post_operations) {
        const Operation & final_post_op = graph.operations[final_post];
        if (final_post_op.inputs.size() < 2) continue;
        const OperationId attention_post = producer(graph, final_post_op.inputs[1]);
        if (attention_post == kInvalidId || graph.operations[attention_post].op != GGML_OP_DSV4_HC_POST) continue;
        const std::set<OperationId> feed_forward_tail = block_closure(index, final_post, attention_post);
        const size_t routed_projections = std::count_if(
            feed_forward_tail.begin(), feed_forward_tail.end(), [&](OperationId operation) {
                return graph.operations[operation].op == GGML_OP_MUL_MAT_ID;
            });
        if (routed_projections < 3) continue;
        const Operation & attention_post_op = graph.operations[attention_post];
        if (attention_post_op.inputs.size() < 2) {
            model.errors.push_back("hybrid-carrier attention publication has fewer than two inputs");
            return model;
        }

        HybridCarrierBlock block;
        block.ordinal = model.blocks.size();
        const ValueId hidden_input = attention_post_op.inputs[1];
        const OperationId hidden_producer = producer(graph, hidden_input);
        std::set<OperationId> block_set = block_closure(index, final_post, hidden_producer);
        if (block_set.count(attention_post) == 0) {
            model.errors.push_back("hybrid-carrier block closure does not contain attention publication");
            return model;
        }

        OperationId attention_pre = kInvalidId;
        OperationId feed_forward_pre = kInvalidId;
        for (OperationId operation : block_set) {
            const Operation & candidate = graph.operations[operation];
            if (candidate.op != GGML_OP_DSV4_HC_PRE || candidate.inputs.empty()) continue;
            if (candidate.inputs[0] == hidden_input) {
                if (attention_pre != kInvalidId) {
                    model.errors.push_back("hybrid-carrier attention preparation is ambiguous");
                    return model;
                }
                attention_pre = operation;
            }
            if (candidate.inputs[0] == attention_post_op.output) {
                if (feed_forward_pre != kInvalidId) {
                    model.errors.push_back("hybrid-carrier feed-forward preparation is ambiguous");
                    return model;
                }
                feed_forward_pre = operation;
            }
        }
        if (attention_pre == kInvalidId || feed_forward_pre == kInvalidId) {
            model.errors.push_back("hybrid-carrier block does not contain both preparation heroes");
            return model;
        }

        for (const Operation & operation : graph.operations) {
            if (operation.op != GGML_OP_SET_ROWS || block_set.count(operation.id) != 0) continue;
            const std::set<OperationId> preparations =
                data_frontier_ancestors(graph, operation.id, GGML_OP_DSV4_HC_PRE);
            if (preparations.size() == 1 && *preparations.begin() == attention_pre) {
                const std::set<OperationId> state_update = data_closure(graph, operation.id, attention_pre);
                block_set.insert(state_update.begin(), state_update.end());
            }
        }
        for (OperationId operation : block_set) {
            if (!all_block_operations.insert(operation).second) {
                model.errors.push_back("hybrid-carrier block closures overlap");
                return model;
            }
        }
        paired_posts.insert(attention_post);
        paired_posts.insert(final_post);
        block.operations = to_vector(block_set);

        OperationId gate_up = kInvalidId;
        OperationId gate_projection = kInvalidId;
        OperationId up_projection = kInvalidId;
        for (OperationId operation : block_set) {
            const Operation & candidate = graph.operations[operation];
            if (candidate.op != GGML_OP_GLU || candidate.inputs.size() < 2) continue;
            const OperationId candidate_gate = nearest_ancestor(graph, candidate.inputs[0], GGML_OP_MUL_MAT_ID);
            const OperationId candidate_up = nearest_ancestor(graph, candidate.inputs[1], GGML_OP_MUL_MAT_ID);
            if (candidate_gate == kInvalidId || candidate_up == kInvalidId || candidate_gate == candidate_up ||
                block_set.count(candidate_gate) == 0 || block_set.count(candidate_up) == 0) continue;
            const Operation & candidate_gate_op = graph.operations[candidate_gate];
            const Operation & candidate_up_op = graph.operations[candidate_up];
            if (candidate_gate_op.inputs.size() < 3 || candidate_up_op.inputs.size() < 3 ||
                candidate_gate_op.inputs[2] != candidate_up_op.inputs[2]) continue;
            if (gate_up != kInvalidId) {
                model.errors.push_back("hybrid-carrier expert gate/up activation is ambiguous");
                return model;
            }
            gate_up = operation;
            gate_projection = candidate_gate;
            up_projection = candidate_up;
        }
        if (gate_up == kInvalidId) {
            model.errors.push_back("hybrid-carrier expert gate/up activation is absent");
            return model;
        }
        const Operation & gate_up_op = graph.operations[gate_up];

        OperationId down_projection = kInvalidId;
        for (OperationId consumer : index.consumers(gate_up_op.output)) {
            if (graph.operations[consumer].op != GGML_OP_MUL_MAT_ID) continue;
            if (down_projection != kInvalidId) {
                model.errors.push_back("hybrid-carrier expert down projection is ambiguous");
                return model;
            }
            down_projection = consumer;
        }
        if (down_projection == kInvalidId || block_set.count(down_projection) == 0) {
            model.errors.push_back("hybrid-carrier expert down projection is absent");
            return model;
        }

        const Operation & gate_projection_op = graph.operations[gate_projection];
        const Operation & up_projection_op = graph.operations[up_projection];
        const Operation & down_projection_op = graph.operations[down_projection];
        if (gate_projection_op.inputs.size() < 3 || up_projection_op.inputs.size() < 3 ||
            down_projection_op.inputs.size() < 3 ||
            gate_projection_op.inputs[2] != up_projection_op.inputs[2] ||
            gate_projection_op.inputs[2] != down_projection_op.inputs[2]) {
            model.errors.push_back("hybrid-carrier expert projections do not share one route-ID value");
            return model;
        }
        const ValueId route_ids = gate_projection_op.inputs[2];

        OperationId weighted = kInvalidId;
        ValueId route_weights = kInvalidId;
        for (OperationId consumer : index.consumers(down_projection_op.output)) {
            const Operation & candidate = graph.operations[consumer];
            if (candidate.op != GGML_OP_MUL || candidate.inputs.size() != 2) continue;
            if (weighted != kInvalidId) {
                model.errors.push_back("hybrid-carrier weighted expert output is ambiguous");
                return model;
            }
            weighted = consumer;
            route_weights = candidate.inputs[0] == down_projection_op.output ? candidate.inputs[1] : candidate.inputs[0];
        }
        if (weighted == kInvalidId || route_weights == kInvalidId) {
            model.errors.push_back("hybrid-carrier weighted expert output is absent");
            return model;
        }
        const OperationId router_projection = nearest_ancestor(graph, route_weights, GGML_OP_MUL_MAT);
        if (router_projection == kInvalidId || block_set.count(router_projection) == 0) {
            model.errors.push_back("hybrid-carrier router projection is absent");
            return model;
        }

        block.operations_by_role.attention_pre = attention_pre;
        block.operations_by_role.attention_post = attention_post;
        block.operations_by_role.feed_forward_pre = feed_forward_pre;
        block.operations_by_role.feed_forward_post = final_post;
        block.operations_by_role.router_projection = router_projection;
        block.operations_by_role.experts_gate_projection = gate_projection;
        block.operations_by_role.experts_up_projection = up_projection;
        block.operations_by_role.experts_gate_up = gate_up;
        block.operations_by_role.experts_down_projection = down_projection;
        block.operations_by_role.experts_weighted = weighted;
        block.values_by_role.hidden_input = hidden_input;
        block.values_by_role.attention_output = attention_post_op.output;
        block.values_by_role.feed_forward_input = graph.operations[feed_forward_pre].output;
        block.values_by_role.route_ids = route_ids;
        block.values_by_role.route_weights = route_weights;
        block.values_by_role.expert_activation = gate_up_op.output;

        const std::set<OperationId> attention_control = ancestors_within(index, { attention_pre }, block_set);
        const std::set<OperationId> attention_all = ancestors_within(index, { attention_post }, block_set);
        const std::set<OperationId> attention_body = difference(attention_all, attention_control);
        std::set<OperationId> claimed = attention_control;
        claimed.insert(attention_body.begin(), attention_body.end());
        const std::set<OperationId> feed_forward_all = ancestors_within(index, { feed_forward_pre }, block_set);
        const std::set<OperationId> feed_forward_control = difference(feed_forward_all, claimed);
        claimed.insert(feed_forward_control.begin(), feed_forward_control.end());
        const OperationId route_id_producer = producer(graph, route_ids);
        const OperationId route_weight_producer = producer(graph, route_weights);
        const std::set<OperationId> router_all = ancestors_within(
            index, { route_id_producer, route_weight_producer }, block_set);
        const std::set<OperationId> router = difference(router_all, claimed);
        claimed.insert(router.begin(), router.end());
        const std::set<OperationId> gate_up_all = ancestors_within(index, { gate_up }, block_set);
        const std::set<OperationId> expert_gate_up = difference(gate_up_all, claimed);
        claimed.insert(expert_gate_up.begin(), expert_gate_up.end());
        const std::set<OperationId> expert_down = difference(block_set, claimed);

        add_component(block, HybridCarrierComponentKind::AttentionControl, attention_pre, attention_control);
        add_component(block, HybridCarrierComponentKind::AttentionBody, attention_post, attention_body);
        add_component(block, HybridCarrierComponentKind::FeedForwardControl, feed_forward_pre, feed_forward_control);
        add_component(block, HybridCarrierComponentKind::RouterSelection, router_projection, router);
        add_component(block, HybridCarrierComponentKind::ExpertGateUp, gate_up, expert_gate_up);
        add_component(block, HybridCarrierComponentKind::ExpertDownPublication, final_post, expert_down);
        model.blocks.push_back(std::move(block));
    }

    if (model.blocks.empty()) {
        model.errors.push_back("graph contains no paired hybrid-carrier blocks");
        return model;
    }
    if (paired_posts.size() != post_operations.size()) {
        model.errors.push_back("hybrid-carrier publication heroes do not form exact attention/feed-forward pairs");
        return model;
    }

    std::set<OperationId> endpoint_set;
    std::queue<OperationId> endpoint_worklist;
    for (ValueId root : graph.roots) {
        const OperationId root_producer = producer(graph, root);
        if (root_producer != kInvalidId && all_block_operations.count(root_producer) == 0 &&
            has_ancestor_in(index, root_producer, all_block_operations) && endpoint_set.insert(root_producer).second) {
            endpoint_worklist.push(root_producer);
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
    model.endpoint_operations = to_vector(endpoint_set);

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
    model.preamble_operations = to_vector(preamble_set);

    const HybridCarrierBlock & first = model.blocks.front();
    const Value & prepared = graph.values[graph.operations[first.operations_by_role.attention_pre].output];
    const Value & hidden = graph.values[first.values_by_role.hidden_input];
    const Value & routes = graph.values[first.values_by_role.route_ids];
    const Value & router = graph.values[graph.operations[first.operations_by_role.router_projection].output];
    model.hidden_size = prepared.access.shape[0];
    model.token_count = prepared.access.shape[1];
    model.carrier_count = hidden.access.shape[1];
    model.route_count = routes.access.shape[0];
    model.expert_count = router.access.shape[0];

    uint32_t next_component = 0;
    model.preamble.id = next_component++;
    model.preamble.kind = HybridCarrierComponentKind::ProgramPreamble;
    model.preamble.hero = model.preamble_operations.empty() ? kInvalidId : model.preamble_operations.back();
    model.preamble.operations = model.preamble_operations;
    model.preamble.boundary = index.boundary(model.preamble.operations);
    for (HybridCarrierBlock & block : model.blocks) {
        for (HybridCarrierComponent & component : block.components) {
            component.id = next_component++;
            component.boundary = index.boundary(component.operations);
        }
    }
    model.endpoint.id = next_component++;
    model.endpoint.kind = HybridCarrierComponentKind::ProgramEndpoint;
    model.endpoint.hero = model.endpoint_operations.empty() ? kInvalidId : model.endpoint_operations.front();
    model.endpoint.operations = model.endpoint_operations;
    model.endpoint.boundary = index.boundary(model.endpoint.operations);

    std::vector<uint8_t> owners(graph.operations.size(), 0);
    auto mark = [&](const HybridCarrierComponent & component) {
        for (OperationId operation : component.operations) {
            if (operation < owners.size()) ++owners[operation];
        }
    };
    mark(model.preamble);
    for (const HybridCarrierBlock & block : model.blocks) {
        for (const HybridCarrierComponent & component : block.components) mark(component);
    }
    mark(model.endpoint);
    for (OperationId operation = 0; operation < owners.size(); ++operation) {
        if (owners[operation] == 0) {
            model.unraised_operations.push_back(operation);
            HybridCarrierComponent fallback;
            fallback.id = next_component++;
            fallback.kind = HybridCarrierComponentKind::Atom;
            fallback.hero = operation;
            fallback.operations = { operation };
            fallback.boundary = index.boundary(fallback.operations);
            model.fallback_components.push_back(std::move(fallback));
        } else if (owners[operation] != 1) {
            model.errors.push_back("hybrid-carrier operation " + std::to_string(operation) +
                                   " is owned by more than one component");
        }
    }

    const VerificationResult verification = verify(index, model);
    model.errors.insert(model.errors.end(), verification.errors.begin(), verification.errors.end());
    return model;
}

VerificationResult HybridCarrierTransformerModel::verify(
        const GraphIndex & index, const HybridCarrierTransformerModel & model) {
    VerificationResult result;
    const Graph & graph = index.graph();
    std::vector<uint8_t> owners(graph.operations.size(), 0);
    uint32_t expected_id = 0;
    auto verify_component = [&](const HybridCarrierComponent & component) {
        if (component.id != expected_id++) result.errors.push_back("hybrid-carrier component has a non-canonical id");
        if (component.operations.empty()) {
            result.errors.push_back("hybrid-carrier component owns no operations");
            return;
        }
        for (OperationId operation : component.operations) {
            if (operation >= owners.size()) result.errors.push_back("hybrid-carrier component owns an invalid operation");
            else if (++owners[operation] != 1) result.errors.push_back("hybrid-carrier operation has duplicate ownership");
        }
        const RegionBoundary boundary = index.boundary(component.operations);
        if (boundary.inputs != component.boundary.inputs || boundary.outputs != component.boundary.outputs) {
            result.errors.push_back("hybrid-carrier component has a stale graph boundary");
        }
        const Decision legality = index.validate_region(component.operations, component.boundary.outputs, true);
        if (!legality.allowed) result.errors.push_back("hybrid-carrier component is illegal: " + legality.detail);
    };

    verify_component(model.preamble);
    for (const HybridCarrierBlock & block : model.blocks) {
        if (block.components.size() != 6) {
            result.errors.push_back("hybrid-carrier block does not contain six logical components");
        }
        const Operation & attention_pre = graph.operations[block.operations_by_role.attention_pre];
        const Value & prepared = graph.values[attention_pre.output];
        const Value & hidden = graph.values[block.values_by_role.hidden_input];
        const Value & routes = graph.values[block.values_by_role.route_ids];
        const Value & router = graph.values[graph.operations[block.operations_by_role.router_projection].output];
        if (prepared.access.shape[0] != model.hidden_size || prepared.access.shape[1] != model.token_count ||
            hidden.access.shape[1] != model.carrier_count || routes.access.shape[0] != model.route_count ||
            routes.access.shape[1] != model.token_count || router.access.shape[0] != model.expert_count) {
            result.errors.push_back("hybrid-carrier block geometry disagrees with recovered facts");
        }
        for (const HybridCarrierComponent & component : block.components) verify_component(component);
    }
    verify_component(model.endpoint);
    for (const HybridCarrierComponent & fallback : model.fallback_components) verify_component(fallback);
    for (OperationId operation = 0; operation < owners.size(); ++operation) {
        if (owners[operation] != 1) result.errors.push_back("hybrid-carrier operation has no unique owner");
    }
    return result;
}

const char * HybridCarrierTransformerModel::component_kind_name(HybridCarrierComponentKind kind) {
    switch (kind) {
        case HybridCarrierComponentKind::ProgramPreamble: return "program.preamble";
        case HybridCarrierComponentKind::AttentionControl: return "attention.control";
        case HybridCarrierComponentKind::AttentionBody: return "attention.body";
        case HybridCarrierComponentKind::FeedForwardControl: return "feed_forward.control";
        case HybridCarrierComponentKind::RouterSelection: return "router.selection";
        case HybridCarrierComponentKind::ExpertGateUp: return "experts.gate_up";
        case HybridCarrierComponentKind::ExpertDownPublication: return "experts.down_publication";
        case HybridCarrierComponentKind::ProgramEndpoint: return "program.endpoint";
        case HybridCarrierComponentKind::Atom: return "atom";
    }
    return "unknown";
}

std::string HybridCarrierTransformerModel::format(const HybridCarrierTransformerModel & model) {
    std::ostringstream out;
    out << "schema=ggml-hrx-logical-hybrid-carrier-transformer-v1\n"
        << "graph=" << model.graph_fingerprint << '\n'
        << "blocks=" << model.blocks.size() << '\n'
        << "tokens=" << model.token_count << '\n'
        << "hidden_size=" << model.hidden_size << '\n'
        << "carriers=" << model.carrier_count << '\n'
        << "experts=" << model.expert_count << '\n'
        << "routes=" << model.route_count << '\n';
    auto append = [&](const HybridCarrierComponent & component, int64_t block) {
        out << "component " << component.id << " kind=" << component_kind_name(component.kind)
            << " block=" << block << " hero=" << component.hero << " ops=";
        for (OperationId operation : component.operations) out << operation << ',';
        out << '\n';
    };
    append(model.preamble, -1);
    for (const HybridCarrierBlock & block : model.blocks) {
        for (const HybridCarrierComponent & component : block.components) append(component, block.ordinal);
    }
    append(model.endpoint, -1);
    for (const HybridCarrierComponent & fallback : model.fallback_components) append(fallback, -1);
    for (const std::string & error : model.errors) out << "error=" << error << '\n';
    return out.str();
}

std::string HybridCarrierTransformerModel::serialize_json(const HybridCarrierTransformerModel & model) {
    std::ostringstream out;
    out << "{\"version\":1,\"graph_fingerprint\":\"" << escape_json(model.graph_fingerprint)
        << "\",\"facts\":{\"block_count\":" << model.blocks.size()
        << ",\"token_count\":" << model.token_count
        << ",\"hidden_size\":" << model.hidden_size
        << ",\"carrier_count\":" << model.carrier_count
        << ",\"expert_count\":" << model.expert_count
        << ",\"route_count\":" << model.route_count << "},\"components\":[";
    bool first = true;
    auto append = [&](const HybridCarrierComponent & component, int64_t block) {
        if (!first) out << ',';
        first = false;
        out << "{\"id\":" << component.id << ",\"kind\":\"" << component_kind_name(component.kind)
            << "\",\"block\":" << block << ",\"hero\":" << component.hero << ",\"operations\":[";
        for (size_t i = 0; i < component.operations.size(); ++i) {
            if (i) out << ',';
            out << component.operations[i];
        }
        out << "]}";
    };
    append(model.preamble, -1);
    for (const HybridCarrierBlock & block : model.blocks) {
        for (const HybridCarrierComponent & component : block.components) append(component, block.ordinal);
    }
    append(model.endpoint, -1);
    for (const HybridCarrierComponent & fallback : model.fallback_components) append(fallback, -1);
    out << "]}";
    return out.str();
}

std::string HybridCarrierTransformerModel::dot(const HybridCarrierTransformerModel & model) {
    std::ostringstream out;
    out << "digraph hybrid_carrier_program {\n  rankdir=LR;\n";
    std::vector<const HybridCarrierComponent *> ordered { &model.preamble };
    for (const HybridCarrierBlock & block : model.blocks) {
        out << "  subgraph cluster_block_" << block.ordinal << " { label=\"block " << block.ordinal << "\";\n";
        for (const HybridCarrierComponent & component : block.components) {
            ordered.push_back(&component);
            out << "    c" << component.id << " [label=\"" << component_kind_name(component.kind)
                << "\\nops=" << component.operations.size() << "\"];\n";
        }
        out << "  }\n";
    }
    ordered.push_back(&model.endpoint);
    for (const HybridCarrierComponent & fallback : model.fallback_components) ordered.push_back(&fallback);
    for (const HybridCarrierComponent * component : ordered) {
        out << "  c" << component->id << " [label=\"" << component_kind_name(component->kind) << "\"];\n";
    }
    for (size_t i = 1; i < ordered.size(); ++i) out << "  c" << ordered[i - 1]->id << " -> c" << ordered[i]->id << ";\n";
    out << "}\n";
    return out.str();
}

Decision HybridCarrierTransformerProvider::discover(const GraphIndex & index, FactDatabase & facts) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) {
        return Decision::reject(DecisionReason::ProviderError,
                                "supplied hybrid-carrier analysis belongs to another graph");
    }
    HybridCarrierTransformerModel recovered;
    const HybridCarrierTransformerModel & model = supplied_model_ ? *supplied_model_ :
        (recovered = HybridCarrierTransformerModel::analyze(index));
    if (!model.valid()) {
        return Decision::reject(DecisionReason::ProviderError,
                                model.errors.empty() ? "hybrid-carrier analysis failed" : model.errors.front());
    }
    const uint32_t hero = model.blocks.front().operations_by_role.feed_forward_post;
    Decision failure;
    if (!observe(facts, "llm.layer_count", model.blocks.size(), "hybrid-carrier blocks", hero, failure) ||
        !observe(facts, "llm.query_token_count", model.token_count, "hybrid-carrier input", hero, failure) ||
        !observe(facts, "llm.hidden_size", model.hidden_size, "hybrid-carrier input", hero, failure) ||
        !observe(facts, "llm.carrier_count", model.carrier_count, "hybrid-carrier state", hero, failure) ||
        !observe(facts, "llm.expert_count", model.expert_count, "router projection", hero, failure) ||
        !observe(facts, "llm.route_count", model.route_count, "expert route IDs", hero, failure)) return failure;
    return Decision::allow();
}

void HybridCarrierTransformerProvider::seed(
        const GraphIndex & index, const FactDatabase &,
        std::vector<FusionCandidate> & candidates) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) return;
    HybridCarrierTransformerModel recovered;
    const HybridCarrierTransformerModel & model = supplied_model_ ? *supplied_model_ :
        (recovered = HybridCarrierTransformerModel::analyze(index));
    if (!model.valid()) return;

    std::vector<uint8_t> native_coverage(index.graph().operations.size(), 0);
    size_t physical_ordinal = 0;
    if (supplied_recipes_ && supplied_recipes_->valid()) {
        for (const auto & match : supplied_recipes_->matches) {
            FusionCandidate candidate;
            candidate.provider = id();
            candidate.family = "native." + match->recipe;
            candidate.key = std::string(id()) + ':' + match->recipe + ".layer." +
                std::to_string(match->layer) + ".occurrence." + std::to_string(physical_ordinal++);
            candidate.hero = match->operations_by_role.empty() ? match->operations.front() :
                match->operations_by_role.begin()->second;
            candidate.logical_components = match->logical_components;
            candidate.operations = match->operations;
            candidate.materialized_outputs = index.boundary(candidate.operations).outputs;
            candidate.allow_disconnected = match->allow_disconnected;
            candidate.correctness_baseline = true;
            candidate.economics.reference_dispatches = std::count_if(
                candidate.operations.begin(), candidate.operations.end(), [&](OperationId operation) {
                    return !metadata_only(index.graph().operations[operation].op);
                });
            candidate.economics.planned_dispatches = static_cast<int32_t>(match->dispatch_count);
            candidate.economics.eliminated_materialization_bytes = static_cast<int64_t>(
                std::min(match->eliminated_materialization_bytes,
                         static_cast<size_t>(std::numeric_limits<int64_t>::max())));
            candidate.payload = match;
            for (OperationId operation : candidate.operations) native_coverage[operation] = 1;
            candidates.push_back(std::move(candidate));
        }
    }

    auto append_component = [&](const HybridCarrierComponent & component, int32_t block) {
        for (OperationId operation : component.operations) {
            const enum ggml_op op = index.graph().operations[operation].op;
            if (!metadata_only(op) && native_coverage[operation] != 0) continue;
            FusionCandidate candidate;
            candidate.provider = id();
            candidate.family = std::string(metadata_only(op) ? "metadata." : "atom.") + ggml_op_name(op);
            candidate.key = std::string(id()) + ':' + std::to_string(operation);
            candidate.hero = operation;
            candidate.logical_components = { component.id };
            candidate.operations = { operation };
            candidate.materialized_outputs = index.boundary(candidate.operations).outputs;
            candidate.correctness_baseline = true;
            candidate.economics.reference_dispatches = metadata_only(op) ? 0 : 1;
            candidate.economics.planned_dispatches = candidate.economics.reference_dispatches;
            auto payload = std::make_shared<HybridCandidatePayload>();
            payload->block = block;
            payload->component = component.kind;
            candidate.payload = std::move(payload);
            candidates.push_back(std::move(candidate));
        }
    };
    append_component(model.preamble, -1);
    for (const HybridCarrierBlock & block : model.blocks) {
        for (const HybridCarrierComponent & component : block.components) {
            append_component(component, static_cast<int32_t>(block.ordinal));
        }
    }
    append_component(model.endpoint, -1);
    for (const HybridCarrierComponent & component : model.fallback_components) append_component(component, -1);
}

PlannerConfiguration HybridCarrierTransformerProvider::make_planner(
        std::shared_ptr<const HybridCarrierTransformerModel> supplied_model,
        std::shared_ptr<const HybridCarrierRecipeDiscovery> supplied_recipes) {
    PlannerConfiguration configuration;
    configuration.add_provider(std::make_shared<HybridCarrierTransformerProvider>(
        std::move(supplied_model), std::move(supplied_recipes)));
    return configuration;
}

HybridCarrierTransformerProgramProof HybridCarrierTransformerProgramProof::recover(
        Graph & graph, const std::string & target) {
    HybridCarrierTransformerProgramProof result;
    result.structurally_recognized = std::any_of(graph.operations.begin(), graph.operations.end(),
        [](const Operation & operation) { return operation.op == GGML_OP_DSV4_HC_POST; });
    if (!result.structurally_recognized) return result;

    const GraphIndex index(graph);
    auto model = std::make_shared<HybridCarrierTransformerModel>(HybridCarrierTransformerModel::analyze(index));
    result.logical_program = model;
    if (!model->valid()) {
        result.errors = model->errors;
        return result;
    }
    auto recipes = std::make_shared<HybridCarrierRecipeDiscovery>(
        HybridCarrierRecipeDiscovery::discover(index, *model, target));
    if (!recipes->valid()) {
        result.errors = recipes->errors;
        return result;
    }
    result.search = SearchResult::search(
        index, HybridCarrierTransformerProvider::make_planner(model, recipes), { true, true });
    if (!result.search.valid()) {
        result.errors = result.search.errors;
        return result;
    }

    Schedule & schedule = result.schedule;
    schedule.graph_fingerprint = graph.fingerprint;
    schedule.workload = std::string(model->token_count == 1 ? "decode" : "prefill") +
        "-hybrid-carrier-" + std::to_string(model->token_count);
    schedule.oracle_revision = "hybrid-carrier-recipes-v1";
    uint32_t dispatch_ordinal = 0;
    for (const FusionCandidate & candidate : result.search.selected) {
        Invocation invocation;
        invocation.recipe = candidate.family;
        invocation.covered_operations = candidate.operations;
        invocation.logical_components = candidate.logical_components;
        const auto payload = std::dynamic_pointer_cast<const HybridCandidatePayload>(candidate.payload);
        const auto native = std::dynamic_pointer_cast<const HybridCarrierRecipeMatch>(candidate.payload);
        invocation.layer = native ? native->layer : payload ? payload->block : -1;
        invocation.stage = native ? "native." + native->recipe :
            payload ? HybridCarrierTransformerModel::component_kind_name(payload->component) : "atom";
        const RegionBoundary boundary = index.boundary(candidate.operations);
        for (size_t i = 0; i < boundary.inputs.size(); ++i) {
            invocation.inputs.push_back({ "arg" + std::to_string(i), boundary.inputs[i], 0, 0, {} });
        }
        for (size_t i = 0; i < boundary.outputs.size(); ++i) {
            invocation.outputs.push_back({ "result" + std::to_string(i), boundary.outputs[i], 0, 0, {} });
        }

        if (native) {
            const VerificationResult bindings = materialize_hybrid_carrier_recipe(
                graph, invocation, *native, dispatch_ordinal);
            result.errors.insert(result.errors.end(), bindings.errors.begin(), bindings.errors.end());
            if (!bindings.valid()) return result;
        } else {
            const Operation & operation = graph.operations[candidate.hero];
            invocation.kernel.family = metadata_only(operation.op) ? "hrx_metadata" : "hrx_atom";
            invocation.kernel.variant = ggml_op_name(operation.op);
            if (!metadata_only(operation.op)) invocation.kernel.kernel_id = kernel_catalog_id(
                invocation.kernel.family.c_str(), invocation.kernel.variant.c_str());
            invocation.kernel.execution_kind = KernelSpecialization::ExecutionKind::NativeEager;
            if (!metadata_only(operation.op)) {
            Dispatch dispatch;
            dispatch.kernel = invocation.kernel;
            dispatch.bindings.insert(dispatch.bindings.end(), invocation.inputs.begin(), invocation.inputs.end());
            dispatch.bindings.insert(dispatch.bindings.end(), invocation.outputs.begin(), invocation.outputs.end());
            if (dispatch_ordinal != 0) dispatch.dependencies.push_back(dispatch_ordinal - 1);
            invocation.dispatches.push_back(std::move(dispatch));
            ++dispatch_ordinal;
            }
        }
        schedule.invocations.push_back(std::move(invocation));
    }
    for (ValueId root : graph.roots) schedule.roots.push_back({ root, RootDisposition::Materialized, "ggml_hybrid_carrier_result" });
    schedule.expected_dispatch_count = dispatch_ordinal;
    const VerificationResult verification = verify_schedule(graph, schedule);
    result.errors.insert(result.errors.end(), verification.errors.begin(), verification.errors.end());
    return result;
}

} // namespace ggml::hrx
