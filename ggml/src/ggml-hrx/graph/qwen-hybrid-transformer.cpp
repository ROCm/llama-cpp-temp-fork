#include "qwen-hybrid-transformer.h"

#include "qwen-hybrid-recipes.h"
#include "qwen-hybrid-transformer-bindings.h"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

struct QwenHybridCandidatePayload final : CandidatePayload {
    int32_t                 block     = -1;
    QwenHybridComponentKind component = QwenHybridComponentKind::Block;
};

OperationId producer(const Graph & graph, ValueId value) {
    return value < graph.values.size() ? graph.values[value].producer : kInvalidId;
}

std::vector<OperationId> consumers_of_kind(const GraphIndex & index, ValueId value, enum ggml_op kind) {
    std::vector<OperationId> result;
    for (OperationId operation : index.consumers(value)) {
        if (index.graph().operations[operation].op == kind) {
            result.push_back(operation);
        }
    }
    return result;
}

ValueId strip_hidden_adapters(const Graph & graph, ValueId value) {
    std::set<ValueId> visited;
    while (visited.insert(value).second) {
        const OperationId operation_id = producer(graph, value);
        if (operation_id == kInvalidId) {
            break;
        }
        const Operation & operation = graph.operations[operation_id];
        const bool        layout    = operation.op == GGML_OP_VIEW || operation.op == GGML_OP_RESHAPE ||
                            operation.op == GGML_OP_PERMUTE || operation.op == GGML_OP_TRANSPOSE;
        const bool selection = operation.op == GGML_OP_GET_ROWS && operation.inputs.size() == 2 &&
                               producer(graph, operation.inputs[0]) != kInvalidId;
        if ((!layout && !selection) || operation.inputs.empty()) {
            break;
        }
        value = operation.inputs[0];
    }
    return value;
}

std::set<OperationId> block_closure(const GraphIndex & index, OperationId root, OperationId stop,
                                    const std::vector<uint8_t> & prior_ownership) {
    std::set<OperationId>   result;
    std::queue<OperationId> worklist;
    result.insert(root);
    worklist.push(root);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (predecessor == stop || (predecessor < prior_ownership.size() && prior_ownership[predecessor] != 0)) {
                continue;
            }
            if (result.insert(predecessor).second) {
                worklist.push(predecessor);
            }
        }
    }
    return result;
}

bool precedes(const GraphIndex & index, OperationId before, OperationId after) {
    if (before == kInvalidId || after == kInvalidId || before == after) {
        return false;
    }
    std::set<OperationId>   visited{ before };
    std::queue<OperationId> worklist;
    worklist.push(before);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId successor : index.successors(current)) {
            if (successor == after) {
                return true;
            }
            if (visited.insert(successor).second) {
                worklist.push(successor);
            }
        }
    }
    return false;
}

bool has_attention_ancestor(const GraphIndex & index, ValueId value) {
    const OperationId root = producer(index.graph(), value);
    if (root == kInvalidId) {
        return false;
    }
    std::set<OperationId>   visited{ root };
    std::queue<OperationId> worklist;
    worklist.push(root);
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        const enum ggml_op op     = index.graph().operations[current].op;
        worklist.pop();
        if (op == GGML_OP_FLASH_ATTN_EXT || op == GGML_OP_GATED_DELTA_NET) {
            return true;
        }
        for (OperationId predecessor : index.predecessors(current)) {
            if (visited.insert(predecessor).second) {
                worklist.push(predecessor);
            }
        }
    }
    return false;
}

struct BlockHeroes {
    QwenHybridFeedForwardKind feed_forward_kind  = QwenHybridFeedForwardKind::RoutedExperts;
    OperationId               feed_forward_hero  = kInvalidId;
    ValueId                   feed_forward_input = kInvalidId;
    OperationId router             = kInvalidId;
    OperationId attention_residual = kInvalidId;
    OperationId final_residual     = kInvalidId;
    ValueId     route_ids          = kInvalidId;
};

bool recover_routed_block_heroes(const GraphIndex & index, OperationId router, BlockHeroes & result) {
    const Graph &     graph      = index.graph();
    const Operation & projection = graph.operations[router];
    if (projection.op != GGML_OP_MUL_MAT || projection.inputs.size() < 2) {
        return false;
    }
    const auto softmax = consumers_of_kind(index, projection.output, GGML_OP_SOFT_MAX);
    if (softmax.size() != 1) {
        return false;
    }
    const auto argsort = consumers_of_kind(index, graph.operations[softmax.front()].output, GGML_OP_ARGSORT);
    if (argsort.size() != 1) {
        return false;
    }
    const auto route_views = consumers_of_kind(index, graph.operations[argsort.front()].output, GGML_OP_VIEW);
    if (route_views.size() != 1) {
        return false;
    }

    const OperationId prepared = producer(graph, projection.inputs[1]);
    if (prepared == kInvalidId || graph.operations[prepared].op != GGML_OP_MUL ||
        graph.operations[prepared].inputs.empty()) {
        return false;
    }
    const OperationId normalization = producer(graph, graph.operations[prepared].inputs[0]);
    if (normalization == kInvalidId || graph.operations[normalization].op != GGML_OP_RMS_NORM ||
        graph.operations[normalization].inputs.empty()) {
        return false;
    }
    const OperationId attention_residual = producer(graph, graph.operations[normalization].inputs[0]);
    if (attention_residual == kInvalidId || graph.operations[attention_residual].op != GGML_OP_ADD) {
        return false;
    }
    const auto final = consumers_of_kind(index, graph.operations[attention_residual].output, GGML_OP_ADD);
    if (final.size() != 1 || !precedes(index, router, final.front())) {
        return false;
    }

    result.feed_forward_kind  = QwenHybridFeedForwardKind::RoutedExperts;
    result.feed_forward_hero  = router;
    result.feed_forward_input = projection.inputs[1];
    result.router             = router;
    result.attention_residual = attention_residual;
    result.final_residual     = final.front();
    result.route_ids          = graph.operations[route_views.front()].output;
    return true;
}

bool recover_dense_block_heroes(const GraphIndex & index, OperationId glu, BlockHeroes & result) {
    const Graph &     graph      = index.graph();
    const Operation & activation = graph.operations[glu];
    if (activation.op != GGML_OP_GLU || activation.inputs.size() != 2) {
        return false;
    }

    const OperationId gate = producer(graph, activation.inputs[0]);
    const OperationId up   = producer(graph, activation.inputs[1]);
    if (gate == kInvalidId || up == kInvalidId || gate == up || graph.operations[gate].op != GGML_OP_MUL_MAT ||
        graph.operations[up].op != GGML_OP_MUL_MAT || graph.operations[gate].inputs.size() < 2 ||
        graph.operations[up].inputs.size() < 2 || graph.operations[gate].inputs[1] != graph.operations[up].inputs[1]) {
        return false;
    }
    const ValueId prepared_value = graph.operations[gate].inputs[1];
    const OperationId prepared   = producer(graph, prepared_value);
    if (prepared == kInvalidId || graph.operations[prepared].op != GGML_OP_MUL ||
        graph.operations[prepared].inputs.empty()) {
        return false;
    }
    const OperationId normalization = producer(graph, graph.operations[prepared].inputs[0]);
    if (normalization == kInvalidId || graph.operations[normalization].op != GGML_OP_RMS_NORM ||
        graph.operations[normalization].inputs.empty()) {
        return false;
    }
    const OperationId attention_residual = producer(graph, graph.operations[normalization].inputs[0]);
    if (attention_residual == kInvalidId || graph.operations[attention_residual].op != GGML_OP_ADD) {
        return false;
    }

    const auto down = consumers_of_kind(index, activation.output, GGML_OP_MUL_MAT);
    if (down.size() != 1) {
        return false;
    }
    const auto final = consumers_of_kind(index, graph.operations[down.front()].output, GGML_OP_ADD);
    if (final.size() != 1 || !precedes(index, glu, final.front())) {
        return false;
    }
    const Operation & final_residual = graph.operations[final.front()];
    if (std::find(final_residual.inputs.begin(), final_residual.inputs.end(),
                  graph.operations[attention_residual].output) == final_residual.inputs.end()) {
        return false;
    }

    result.feed_forward_kind  = QwenHybridFeedForwardKind::DenseSwiGLU;
    result.feed_forward_hero  = glu;
    result.feed_forward_input = prepared_value;
    result.attention_residual = attention_residual;
    result.final_residual     = final.front();
    return true;
}

bool no_dispatch_operation(const Graph & graph, OperationId operation_id) {
    const Operation & operation = graph.operations[operation_id];
    if (operation.op == GGML_OP_VIEW || operation.op == GGML_OP_RESHAPE || operation.op == GGML_OP_PERMUTE ||
        operation.op == GGML_OP_TRANSPOSE) {
        return true;
    }
    const Value & output = graph.values[operation.output];
    return std::any_of(output.access.shape.begin(), output.access.shape.end(),
                       [](int64_t extent) { return extent == 0; });
}

bool observe(FactDatabase &      facts,
             const std::string & key,
             int64_t             value,
             const std::string & source,
             uint32_t            graph_id,
             Decision &          failure) {
    failure = facts.observe(key, value, { source, graph_id });
    return failure.allowed;
}

void assign_component(QwenHybridComponent &    component,
                      uint32_t                 id,
                      QwenHybridComponentKind  kind,
                      OperationId              hero,
                      std::vector<OperationId> operations,
                      const GraphIndex &       index) {
    component.id         = id;
    component.kind       = kind;
    component.hero       = hero;
    component.operations = std::move(operations);
    component.boundary   = index.boundary(component.operations);
}

}  // namespace

QwenHybridTransformerModel QwenHybridTransformerModel::analyze(const GraphIndex & index) {
    QwenHybridTransformerModel model;
    model.graph_fingerprint = index.graph().fingerprint;
    if (!index.valid()) {
        model.errors = index.errors();
        return model;
    }
    const Graph &            graph = index.graph();
    std::vector<BlockHeroes> heroes;
    std::set<OperationId>    final_residuals;
    for (const Operation & operation : graph.operations) {
        BlockHeroes candidate;
        if (!recover_routed_block_heroes(index, operation.id, candidate) &&
            !recover_dense_block_heroes(index, operation.id, candidate)) {
            continue;
        }
        if (!final_residuals.insert(candidate.final_residual).second) {
            model.errors.push_back("Qwen hybrid block has multiple feed-forward anchors");
            return model;
        }
        heroes.push_back(candidate);
    }
    if (heroes.empty()) {
        model.errors.push_back("graph contains no Qwen hybrid transformer blocks");
        return model;
    }

    std::map<ValueId, size_t> block_by_output;
    for (size_t block = 0; block < heroes.size(); ++block) {
        const ValueId output = graph.operations[heroes[block].final_residual].output;
        if (!block_by_output.emplace(output, block).second) {
            model.errors.push_back("Qwen hybrid blocks have duplicate hidden-state outputs");
            return model;
        }
    }
    std::vector<size_t> predecessor(heroes.size(), heroes.size());
    std::vector<size_t> successor(heroes.size(), heroes.size());
    for (size_t block = 0; block < heroes.size(); ++block) {
        std::set<size_t> incoming;
        for (ValueId input : graph.operations[heroes[block].attention_residual].inputs) {
            const auto prior = block_by_output.find(strip_hidden_adapters(graph, input));
            if (prior != block_by_output.end() && prior->second != block) {
                incoming.insert(prior->second);
            }
        }
        if (incoming.size() > 1) {
            model.errors.push_back("Qwen hybrid block has multiple hidden-state predecessors");
            return model;
        }
        if (!incoming.empty()) {
            predecessor[block] = *incoming.begin();
            if (successor[predecessor[block]] != heroes.size()) {
                model.errors.push_back("Qwen hybrid hidden state feeds multiple blocks");
                return model;
            }
            successor[predecessor[block]] = block;
        }
    }
    std::vector<size_t> roots;
    for (size_t block = 0; block < heroes.size(); ++block) {
        if (predecessor[block] == heroes.size()) {
            roots.push_back(block);
        }
    }
    if (roots.size() != 1) {
        model.errors.push_back("Qwen hybrid blocks do not form one hidden-state chain");
        return model;
    }
    std::vector<BlockHeroes> ordered;
    std::vector<uint8_t>     visited(heroes.size(), 0);
    for (size_t block = roots.front(); block != heroes.size(); block = successor[block]) {
        if (visited[block] != 0) {
            model.errors.push_back("Qwen hybrid hidden-state chain is cyclic");
            return model;
        }
        visited[block] = 1;
        ordered.push_back(heroes[block]);
    }
    if (ordered.size() != heroes.size()) {
        model.errors.push_back("Qwen hybrid hidden-state chain is disconnected");
        return model;
    }
    heroes = std::move(ordered);

    ValueId initial_hidden = kInvalidId;
    for (ValueId input : graph.operations[heroes.front().attention_residual].inputs) {
        const ValueId stripped = strip_hidden_adapters(graph, input);
        if (!has_attention_ancestor(index, stripped)) {
            if (initial_hidden != kInvalidId) {
                model.errors.push_back("Qwen hybrid initial hidden state is ambiguous");
                return model;
            }
            initial_hidden = stripped;
        }
    }
    if (initial_hidden == kInvalidId) {
        model.errors.push_back("Qwen hybrid initial hidden state is not structurally identifiable");
        return model;
    }

    std::vector<uint8_t> block_ownership(graph.operations.size(), 0);
    for (size_t ordinal = 0; ordinal < heroes.size(); ++ordinal) {
        const BlockHeroes & recovered          = heroes[ordinal];
        const Operation &   attention_residual = graph.operations[recovered.attention_residual];
        if (attention_residual.inputs.size() != 2) {
            model.errors.push_back("Qwen hybrid attention residual is not binary");
            return model;
        }
        const ValueId hidden =
            ordinal == 0 ? initial_hidden : graph.operations[heroes[ordinal - 1].final_residual].output;
        if (ordinal != 0) {
            bool found_hidden = false;
            for (ValueId input : attention_residual.inputs) {
                if (strip_hidden_adapters(graph, input) == hidden) {
                    if (found_hidden) {
                        model.errors.push_back("Qwen hybrid hidden-state chain is ambiguous");
                        return model;
                    }
                    found_hidden = true;
                }
            }
            if (!found_hidden) {
                model.errors.push_back("Qwen hybrid block does not consume the previous hidden state");
                return model;
            }
        }
        const OperationId           stop    = producer(graph, hidden);
        const std::set<OperationId> closure =
            block_closure(index, recovered.final_residual, stop, block_ownership);
        if (closure.count(recovered.feed_forward_hero) == 0 ||
            closure.count(recovered.attention_residual) == 0) {
            model.errors.push_back("Qwen hybrid block closure misses a structural hero");
            return model;
        }
        QwenHybridBlock block;
        block.ordinal            = ordinal;
        block.feed_forward_kind  = recovered.feed_forward_kind;
        block.feed_forward_hero  = recovered.feed_forward_hero;
        block.feed_forward_input = recovered.feed_forward_input;
        block.router_projection  = recovered.router;
        block.attention_residual = recovered.attention_residual;
        block.final_residual     = recovered.final_residual;
        block.route_ids          = recovered.route_ids;
        block.operations.assign(closure.begin(), closure.end());
        for (OperationId operation : block.operations) {
            if (++block_ownership[operation] != 1) {
                model.errors.push_back("Qwen hybrid block closures overlap");
                return model;
            }
        }
        QwenHybridComponent component;
        component.kind       = QwenHybridComponentKind::Block;
        component.hero       = recovered.feed_forward_hero;
        component.operations = block.operations;
        block.components.push_back(std::move(component));
        model.blocks.push_back(std::move(block));
    }

    std::vector<uint8_t> boundary_ownership(graph.operations.size(), 0);
    const OperationId    preamble_root = producer(graph, initial_hidden);
    if (preamble_root != kInvalidId) {
        std::queue<OperationId> worklist;
        boundary_ownership[preamble_root] = 1;
        worklist.push(preamble_root);
        while (!worklist.empty()) {
            const OperationId operation = worklist.front();
            worklist.pop();
            for (OperationId predecessor_id : index.predecessors(operation)) {
                if (block_ownership[predecessor_id] == 0 && boundary_ownership[predecessor_id] == 0) {
                    boundary_ownership[predecessor_id] = 1;
                    worklist.push(predecessor_id);
                }
            }
        }
    }

    std::queue<OperationId> endpoint_worklist;
    endpoint_worklist.push(heroes.back().final_residual);
    while (!endpoint_worklist.empty()) {
        const OperationId operation = endpoint_worklist.front();
        endpoint_worklist.pop();
        for (OperationId successor_id : index.successors(operation)) {
            if (block_ownership[successor_id] == 0 && boundary_ownership[successor_id] == 0) {
                boundary_ownership[successor_id] = 2;
                endpoint_worklist.push(successor_id);
            }
        }
    }

    const size_t no_owner = model.blocks.size();
    std::vector<size_t> nearest_owner(graph.operations.size(), no_owner);
    std::vector<size_t> nearest_distance(graph.operations.size(), std::numeric_limits<size_t>::max());
    for (size_t block = 0; block < model.blocks.size(); ++block) {
        std::vector<size_t> distance(graph.operations.size(), std::numeric_limits<size_t>::max());
        std::queue<OperationId> worklist;
        for (OperationId operation : model.blocks[block].operations) {
            distance[operation] = 0;
            worklist.push(operation);
        }
        while (!worklist.empty()) {
            const OperationId operation = worklist.front();
            worklist.pop();
            auto visit = [&](OperationId adjacent) {
                if (boundary_ownership[adjacent] != 0 || distance[adjacent] <= distance[operation] + 1) {
                    return;
                }
                distance[adjacent] = distance[operation] + 1;
                worklist.push(adjacent);
            };
            for (OperationId predecessor_id : index.predecessors(operation)) {
                visit(predecessor_id);
            }
            for (OperationId successor_id : index.successors(operation)) {
                visit(successor_id);
            }
        }
        for (OperationId operation = 0; operation < graph.operations.size(); ++operation) {
            if (distance[operation] < nearest_distance[operation]) {
                nearest_distance[operation] = distance[operation];
                nearest_owner[operation]    = block;
            }
        }
    }

    std::vector<OperationId> preamble;
    std::vector<OperationId> endpoint;
    for (OperationId operation = 0; operation < graph.operations.size(); ++operation) {
        if (block_ownership[operation] != 0) {
            continue;
        }
        if (boundary_ownership[operation] == 1) {
            preamble.push_back(operation);
        } else if (boundary_ownership[operation] == 2) {
            endpoint.push_back(operation);
        } else if (nearest_owner[operation] != no_owner) {
            model.blocks[nearest_owner[operation]].operations.push_back(operation);
        } else {
            model.errors.push_back("Qwen hybrid operation is disconnected from every logical block");
            return model;
        }
    }
    for (QwenHybridBlock & block : model.blocks) {
        std::sort(block.operations.begin(), block.operations.end());
        block.components.front().operations = block.operations;
    }
    if (preamble.empty() || endpoint.empty()) {
        model.errors.push_back("Qwen hybrid graph has an empty program boundary");
        return model;
    }

    uint32_t component_id = 0;
    assign_component(model.preamble, component_id++, QwenHybridComponentKind::ProgramPreamble, preamble.back(),
                     std::move(preamble), index);
    for (QwenHybridBlock & block : model.blocks) {
        assign_component(block.components.front(), component_id++, QwenHybridComponentKind::Block,
                         block.feed_forward_hero, block.operations, index);
    }
    assign_component(model.endpoint, component_id++, QwenHybridComponentKind::ProgramEndpoint, endpoint.front(),
                     std::move(endpoint), index);

    const QwenHybridBlock & first    = model.blocks.front();
    const Value &           prepared = graph.values[first.feed_forward_input];
    model.hidden_size                = prepared.access.shape[0];
    model.token_count                = prepared.access.shape[1];
    if (first.feed_forward_kind == QwenHybridFeedForwardKind::RoutedExperts) {
        const Operation & first_router  = graph.operations[first.router_projection];
        const Value &     router_output = graph.values[first_router.output];
        const Value &     route_ids     = graph.values[first.route_ids];
        model.expert_count              = router_output.access.shape[0];
        model.route_count               = route_ids.access.shape[0];
    }

    const VerificationResult verification = verify(index, model);
    model.errors.insert(model.errors.end(), verification.errors.begin(), verification.errors.end());
    return model;
}

VerificationResult QwenHybridTransformerModel::verify(const GraphIndex &                 index,
                                                      const QwenHybridTransformerModel & model) {
    VerificationResult   result;
    const Graph &        graph = index.graph();
    std::vector<uint8_t> owners(graph.operations.size(), 0);
    uint32_t             expected_id      = 0;
    auto                 verify_component = [&](const QwenHybridComponent & component) {
        if (component.id != expected_id++) {
            result.errors.push_back("Qwen hybrid logical component has a non-canonical id");
        }
        if (component.operations.empty()) {
            result.errors.push_back("Qwen hybrid logical component owns no operations");
            return;
        }
        for (OperationId operation : component.operations) {
            if (operation >= owners.size()) {
                result.errors.push_back("Qwen hybrid logical component owns an invalid operation");
            } else if (++owners[operation] != 1) {
                result.errors.push_back("Qwen hybrid logical operation has duplicate ownership");
            }
        }
        const RegionBoundary expected = index.boundary(component.operations);
        if (component.boundary.inputs != expected.inputs || component.boundary.outputs != expected.outputs) {
            result.errors.push_back("Qwen hybrid logical component has a stale graph boundary");
        }
        const Decision legality = index.validate_region(component.operations, component.boundary.outputs, true);
        if (!legality.allowed) {
            result.errors.push_back("Qwen hybrid logical component is illegal: " + legality.detail);
        }
    };

    verify_component(model.preamble);
    for (const QwenHybridBlock & block : model.blocks) {
        if (block.components.size() != 1 || block.operations != block.components.front().operations) {
            result.errors.push_back("Qwen hybrid block has invalid logical ownership");
            continue;
        }
        const Value & prepared = graph.values[block.feed_forward_input];
        bool          geometry_matches = prepared.access.shape[0] == model.hidden_size &&
                                prepared.access.shape[1] == model.token_count;
        if (block.feed_forward_kind == QwenHybridFeedForwardKind::RoutedExperts) {
            const Operation & router = graph.operations[block.router_projection];
            const Value &     logits = graph.values[router.output];
            const Value &     routes = graph.values[block.route_ids];
            geometry_matches = geometry_matches && logits.access.shape[0] == model.expert_count &&
                               routes.access.shape[0] == model.route_count;
        } else {
            geometry_matches = geometry_matches && block.router_projection == kInvalidId &&
                               block.route_ids == kInvalidId && model.expert_count == 0 && model.route_count == 0;
        }
        if (!geometry_matches) {
            result.errors.push_back("Qwen hybrid block disagrees with recovered model geometry");
        }
        verify_component(block.components.front());
    }
    verify_component(model.endpoint);
    for (OperationId operation = 0; operation < owners.size(); ++operation) {
        if (owners[operation] != 1) {
            result.errors.push_back("Qwen hybrid logical operation does not have exactly one owner");
        }
    }
    return result;
}

const char * QwenHybridTransformerModel::component_kind_name(QwenHybridComponentKind kind) {
    switch (kind) {
        case QwenHybridComponentKind::ProgramPreamble:
            return "program.preamble";
        case QwenHybridComponentKind::Block:
            return "transformer.block";
        case QwenHybridComponentKind::ProgramEndpoint:
            return "program.endpoint";
    }
    return "unknown";
}

std::string QwenHybridTransformerModel::format(const QwenHybridTransformerModel & model) {
    std::ostringstream out;
    out << "schema=ggml-hrx-logical-qwen-hybrid-transformer-v1\n"
        << "graph=" << model.graph_fingerprint << '\n'
        << "blocks=" << model.blocks.size() << '\n'
        << "tokens=" << model.token_count << '\n'
        << "hidden_size=" << model.hidden_size << '\n'
        << "experts=" << model.expert_count << '\n'
        << "routes=" << model.route_count << '\n';
    auto append = [&](const QwenHybridComponent & component, int64_t block) {
        out << "component " << component.id << " kind=" << component_kind_name(component.kind) << " block=" << block
            << " hero=" << component.hero << " ops=";
        for (OperationId operation : component.operations) {
            out << operation << ',';
        }
        out << '\n';
    };
    append(model.preamble, -1);
    for (const QwenHybridBlock & block : model.blocks) {
        append(block.components.front(), static_cast<int64_t>(block.ordinal));
    }
    append(model.endpoint, -1);
    for (const std::string & error : model.errors) {
        out << "error=" << error << '\n';
    }
    return out.str();
}

std::string QwenHybridTransformerModel::serialize_json(const QwenHybridTransformerModel & model) {
    std::ostringstream out;
    out << "{\"version\":1,\"graph_fingerprint\":\"" << model.graph_fingerprint
        << "\",\"facts\":{\"block_count\":" << model.blocks.size() << ",\"token_count\":" << model.token_count
        << ",\"hidden_size\":" << model.hidden_size << ",\"expert_count\":" << model.expert_count
        << ",\"route_count\":" << model.route_count << "},\"components\":[";
    bool first  = true;
    auto append = [&](const QwenHybridComponent & component, int64_t block) {
        if (!first) {
            out << ',';
        }
        first = false;
        out << "{\"id\":" << component.id << ",\"kind\":\"" << component_kind_name(component.kind)
            << "\",\"block\":" << block << ",\"hero\":" << component.hero << ",\"operations\":[";
        for (size_t i = 0; i < component.operations.size(); ++i) {
            if (i) {
                out << ',';
            }
            out << component.operations[i];
        }
        out << "]}";
    };
    append(model.preamble, -1);
    for (const QwenHybridBlock & block : model.blocks) {
        append(block.components.front(), static_cast<int64_t>(block.ordinal));
    }
    append(model.endpoint, -1);
    out << "]}";
    return out.str();
}

std::string QwenHybridTransformerModel::dot(const QwenHybridTransformerModel & model) {
    std::ostringstream out;
    out << "digraph qwen_hybrid_program {\n  rankdir=LR;\n";
    std::vector<const QwenHybridComponent *> ordered{ &model.preamble };
    for (const QwenHybridBlock & block : model.blocks) {
        ordered.push_back(&block.components.front());
    }
    ordered.push_back(&model.endpoint);
    for (const QwenHybridComponent * component : ordered) {
        out << "  c" << component->id << " [label=\"" << component_kind_name(component->kind)
            << "\nops=" << component->operations.size() << "\"];\n";
    }
    for (size_t i = 1; i < ordered.size(); ++i) {
        out << "  c" << ordered[i - 1]->id << " -> c" << ordered[i]->id << ";\n";
    }
    out << "}\n";
    return out.str();
}

Decision QwenHybridTransformerProvider::discover(const GraphIndex & index, FactDatabase & facts) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) {
        return Decision::reject(DecisionReason::ProviderError,
                                "supplied Qwen hybrid analysis belongs to another graph");
    }
    QwenHybridTransformerModel         recovered;
    const QwenHybridTransformerModel & model =
        supplied_model_ ? *supplied_model_ : (recovered = QwenHybridTransformerModel::analyze(index));
    if (!model.valid()) {
        return Decision::reject(DecisionReason::ProviderError,
                                model.errors.empty() ? "Qwen hybrid analysis failed" : model.errors.front());
    }
    const uint32_t hero = model.blocks.front().feed_forward_hero;
    Decision       failure;
    if (!observe(facts, "llm.layer_count", model.blocks.size(), "Qwen hybrid blocks", hero, failure) ||
        !observe(facts, "llm.query_token_count", model.token_count, "Qwen hybrid input", hero, failure) ||
        !observe(facts, "llm.hidden_size", model.hidden_size, "Qwen hybrid input", hero, failure) ||
        !observe(facts, "llm.expert_count", model.expert_count, "Qwen hybrid feed-forward", hero, failure) ||
        !observe(facts, "llm.route_count", model.route_count, "Qwen hybrid feed-forward", hero, failure)) {
        return failure;
    }
    return Decision::allow();
}

void QwenHybridTransformerProvider::seed(const GraphIndex & index,
                                         const FactDatabase &,
                                         std::vector<FusionCandidate> & candidates) const {
    if (supplied_model_ && supplied_model_->graph_fingerprint != index.graph().fingerprint) {
        return;
    }
    QwenHybridTransformerModel         recovered;
    const QwenHybridTransformerModel & model =
        supplied_model_ ? *supplied_model_ : (recovered = QwenHybridTransformerModel::analyze(index));
    if (!model.valid()) {
        return;
    }

    if (supplied_recipes_ && supplied_recipes_->valid()) {
        for (const auto & match : supplied_recipes_->matches) {
            FusionCandidate candidate;
            candidate.provider = id();
            candidate.family   = "native." + match->recipe;
            candidate.key      = std::string(id()) + ':' + match->recipe + ':';
            for (OperationId operation : match->operations) {
                candidate.key += std::to_string(operation) + ',';
            }
            candidate.hero                 = match->operations_by_role.empty() ? match->operations.front() :
                                                                                 match->operations_by_role.begin()->second;
            candidate.logical_components   = match->logical_components;
            candidate.operations           = match->operations;
            candidate.materialized_outputs = index.boundary(candidate.operations).outputs;
            candidate.allow_disconnected   = match->allow_disconnected;
            candidate.correctness_baseline = true;
            candidate.economics.reference_dispatches =
                std::count_if(candidate.operations.begin(), candidate.operations.end(),
                              [&](OperationId operation) { return !no_dispatch_operation(index.graph(), operation); });
            candidate.economics.planned_dispatches               = static_cast<int32_t>(match->dispatch_count);
            candidate.economics.eliminated_materialization_bytes = static_cast<int64_t>(std::min(
                match->eliminated_materialization_bytes, static_cast<size_t>(std::numeric_limits<int64_t>::max())));
            candidate.payload                                    = match;
            candidates.push_back(std::move(candidate));
        }
    }

    auto append_component = [&](const QwenHybridComponent & component, int32_t block) {
        for (OperationId operation : component.operations) {
            if (!no_dispatch_operation(index.graph(), operation)) {
                continue;
            }
            FusionCandidate candidate;
            candidate.provider = id();
            candidate.family   = "metadata." + std::string(ggml_op_name(index.graph().operations[operation].op));
            candidate.key      = std::string(id()) + ":metadata:" + std::to_string(operation);
            candidate.hero     = operation;
            candidate.logical_components   = { component.id };
            candidate.operations           = { operation };
            candidate.materialized_outputs = index.boundary(candidate.operations).outputs;
            candidate.correctness_baseline = true;
            auto payload                   = std::make_shared<QwenHybridCandidatePayload>();
            payload->block                 = block;
            payload->component             = component.kind;
            candidate.payload              = std::move(payload);
            candidates.push_back(std::move(candidate));
        }
    };
    append_component(model.preamble, -1);
    for (const QwenHybridBlock & block : model.blocks) {
        append_component(block.components.front(), static_cast<int32_t>(block.ordinal));
    }
    append_component(model.endpoint, -1);
}

PlannerConfiguration QwenHybridTransformerProvider::make_planner(
    std::shared_ptr<const QwenHybridTransformerModel> supplied_model,
    std::shared_ptr<const QwenHybridRecipeDiscovery>  supplied_recipes) {
    PlannerConfiguration configuration;
    configuration.add_provider(
        std::make_shared<QwenHybridTransformerProvider>(std::move(supplied_model), std::move(supplied_recipes)));
    return configuration;
}

QwenHybridTransformerProgramProof QwenHybridTransformerProgramProof::recover(Graph &             graph,
                                                                             const std::string & target) {
    QwenHybridTransformerProgramProof result;
    const bool                        has_recurrent_attention =
        std::any_of(graph.operations.begin(), graph.operations.end(),
                    [](const Operation & operation) { return operation.op == GGML_OP_GATED_DELTA_NET; });
    const bool has_full_attention =
        std::any_of(graph.operations.begin(), graph.operations.end(),
                    [](const Operation & operation) { return operation.op == GGML_OP_FLASH_ATTN_EXT; });
    const bool has_feed_forward = std::any_of(graph.operations.begin(), graph.operations.end(),
                                              [](const Operation & operation) {
                                                  return operation.op == GGML_OP_MUL_MAT_ID ||
                                                         operation.op == GGML_OP_GLU;
                                              });
    result.structurally_recognized = has_recurrent_attention && has_full_attention && has_feed_forward;
    if (!result.structurally_recognized) {
        return result;
    }

    const GraphIndex index(graph);
    auto             model = std::make_shared<QwenHybridTransformerModel>(QwenHybridTransformerModel::analyze(index));
    result.logical_program = model;
    if (!model->valid()) {
        result.errors = model->errors;
        return result;
    }
    auto recipes =
        std::make_shared<QwenHybridRecipeDiscovery>(QwenHybridRecipeDiscovery::discover(index, *model, target));
    if (!recipes->valid()) {
        result.errors = recipes->errors;
        return result;
    }
    result.search =
        SearchResult::search(index, QwenHybridTransformerProvider::make_planner(model, recipes), { true, true });
    if (!result.search.valid()) {
        result.errors = result.search.errors;
        return result;
    }

    Schedule & schedule        = result.schedule;
    schedule.graph_fingerprint = graph.fingerprint;
    schedule.workload          = std::string(model->token_count == 1 ? "decode" : "prefill") + "-Qwen-hybrid-" +
                        std::to_string(model->token_count);
    schedule.oracle_revision  = "Qwen-hybrid-native-recipes-v1";
    uint32_t dispatch_ordinal = 0;
    for (const FusionCandidate & candidate : result.search.selected) {
        Invocation invocation;
        invocation.recipe             = candidate.family;
        invocation.covered_operations = candidate.operations;
        invocation.logical_components = candidate.logical_components;
        const auto payload            = std::dynamic_pointer_cast<const QwenHybridCandidatePayload>(candidate.payload);
        const auto native             = std::dynamic_pointer_cast<const QwenHybridRecipeMatch>(candidate.payload);
        invocation.layer              = native ? native->layer : payload ? payload->block : -1;
        invocation.stage              = native  ? "native." + native->recipe :
                                        payload ? QwenHybridTransformerModel::component_kind_name(payload->component) :
                                                  "metadata";
        const RegionBoundary boundary = index.boundary(candidate.operations);
        for (size_t i = 0; i < boundary.inputs.size(); ++i) {
            invocation.inputs.push_back({ "arg" + std::to_string(i), boundary.inputs[i], 0, 0, {} });
        }
        for (size_t i = 0; i < boundary.outputs.size(); ++i) {
            invocation.outputs.push_back({ "result" + std::to_string(i), boundary.outputs[i], 0, 0, {} });
        }

        if (native) {
            const VerificationResult bindings =
                materialize_qwen_hybrid_recipe(graph, invocation, *native, dispatch_ordinal);
            result.errors.insert(result.errors.end(), bindings.errors.begin(), bindings.errors.end());
            if (!bindings.valid()) {
                return result;
            }
        } else {
            invocation.kernel.family         = "hrx_metadata";
            invocation.kernel.variant        = ggml_op_name(graph.operations[candidate.hero].op);
            invocation.kernel.execution_kind = KernelSpecialization::ExecutionKind::NativeGap;
        }
        schedule.invocations.push_back(std::move(invocation));
    }
    for (ValueId root : graph.roots) {
        schedule.roots.push_back({ root, RootDisposition::Materialized, "ggml_qwen_hybrid_result" });
    }
    schedule.expected_dispatch_count      = dispatch_ordinal;
    const VerificationResult verification = verify_schedule(graph, schedule);
    result.errors.insert(result.errors.end(), verification.errors.begin(), verification.errors.end());
    return result;
}

}  // namespace ggml::hrx
