#include "matcher.h"

#include <algorithm>

namespace ggml::hrx {
namespace {

class MatchImplementation {
public:

static int tensor_rank(const Value & value) {
    int rank = GGML_MAX_DIMS;
    while (rank > 1 && value.access.shape[rank - 1] == 1) {
        --rank;
    }
    return rank;
}

static bool satisfies(const Graph & graph, const Operation & operation, const OpConstraint & constraint) {
    const Value & value = graph.values[operation.output];
    if (!constraint.alternatives.empty() &&
        std::find(constraint.alternatives.begin(), constraint.alternatives.end(), value.op) == constraint.alternatives.end()) {
        return false;
    }
    if (constraint.output_type != GGML_TYPE_COUNT && value.type != constraint.output_type) {
        return false;
    }
    if (constraint.rank >= 0 && tensor_rank(value) != constraint.rank) {
        return false;
    }
    for (size_t i = 0; i < constraint.shape.size(); ++i) {
        if (i >= value.access.shape.size() || (constraint.shape[i] >= 0 && constraint.shape[i] != value.access.shape[i])) {
            return false;
        }
    }
    for (const InputTypeConstraint & input_constraint : constraint.input_types) {
        if (input_constraint.input_index >= operation.inputs.size()) {
            return false;
        }
        const enum ggml_type type = graph.values[operation.inputs[input_constraint.input_index]].type;
        if (!input_constraint.alternatives.empty() &&
            std::find(input_constraint.alternatives.begin(), input_constraint.alternatives.end(), type) == input_constraint.alternatives.end()) {
            return false;
        }
    }
    return true;
}

};

} // namespace

Match MatchAutomaton::match(const Graph & graph, const MatchAutomaton & automaton, OperationId root) {
    Match result;
    if (automaton.states.empty() || automaton.root_state >= automaton.states.size() || root >= graph.operations.size()) {
        return result;
    }

    std::vector<size_t> uses(graph.values.size(), 0);
    for (const Operation & operation : graph.operations) {
        for (ValueId input : operation.inputs) {
            if (input < uses.size()) {
                ++uses[input];
            }
        }
    }
    std::vector<OperationId> bindings(automaton.states.size(), kInvalidId);
    std::vector<uint8_t> active(automaton.states.size(), 0);
    auto storage_writer = [&](const Operation & operation, size_t input_index) -> OperationId {
        if (input_index >= operation.inputs.size()) {
            return kInvalidId;
        }
        size_t read_index = 0;
        const Effect * read = nullptr;
        for (const Effect & effect : operation.effects) {
            if (effect.kind != EffectKind::Read) {
                continue;
            }
            if (read_index++ == input_index) {
                read = &effect;
                break;
            }
        }
        if (read == nullptr || read->before_version == 0) {
            return kInvalidId;
        }
        for (OperationId candidate = operation.id; candidate-- > 0;) {
            for (const Effect & effect : graph.operations[candidate].effects) {
                if (effect.kind == EffectKind::Write && effect.storage == read->storage &&
                    effect.after_version == read->before_version) {
                    return candidate;
                }
            }
        }
        return kInvalidId;
    };
    auto visit = [&](auto && self, size_t state_id, OperationId operation_id) -> bool {
        if (state_id >= automaton.states.size() || operation_id >= graph.operations.size()) {
            return false;
        }
        if (bindings[state_id] != kInvalidId) {
            return bindings[state_id] == operation_id;
        }
        if (std::find(bindings.begin(), bindings.end(), operation_id) != bindings.end()) {
            return false;
        }
        if (active[state_id] != 0) {
            return false;
        }
        active[state_id] = 1;
        const MatchState & state = automaton.states[state_id];
        const Operation & operation = graph.operations[operation_id];
        if (operation.output >= graph.values.size() || !MatchImplementation::satisfies(graph, operation, state.constraint)) {
            active[state_id] = 0;
            return false;
        }
        bindings[state_id] = operation_id;
        for (const Transition & transition : state.transitions) {
            std::vector<OperationId> candidates;
            if (transition.kind == Transition::Kind::OutputConsumer) {
                for (const Operation & candidate : graph.operations) {
                    if (candidate.id > operation.id &&
                        std::find(candidate.inputs.begin(), candidate.inputs.end(), operation.output) != candidate.inputs.end()) {
                        candidates.push_back(candidate.id);
                    }
                }
            } else if (transition.input_index < operation.inputs.size()) {
                ValueId input = operation.inputs[transition.input_index];
                if (input < graph.values.size()) {
                    OperationId predecessor = transition.kind == Transition::Kind::InputProducer ?
                        graph.values[input].producer : storage_writer(operation, transition.input_index);
                    if (predecessor != kInvalidId &&
                        !(automaton.require_internal_single_use && transition.kind == Transition::Kind::InputProducer && uses[input] != 1)) {
                        candidates.push_back(predecessor);
                    }
                }
            }
            const std::vector<OperationId> saved_bindings = bindings;
            const std::vector<uint8_t> saved_active = active;
            bool matched = false;
            for (OperationId candidate : candidates) {
                bindings = saved_bindings;
                active = saved_active;
                if (self(self, transition.target_state, candidate)) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                bindings[state_id] = kInvalidId;
                active[state_id] = 0;
                return false;
            }
        }
        active[state_id] = 0;
        return true;
    };

    if (!visit(visit, automaton.root_state, root)) {
        return result;
    }
    result.automaton = automaton.name;
    result.state_operations = std::move(bindings);
    for (size_t state_id = 0; state_id < automaton.states.size(); ++state_id) {
        if (automaton.states[state_id].capture) result.covered_operations.push_back(result.state_operations[state_id]);
    }
    std::sort(result.covered_operations.begin(), result.covered_operations.end());
    result.covered_operations.erase(std::unique(result.covered_operations.begin(), result.covered_operations.end()),
                                    result.covered_operations.end());
    return result;
}

} // namespace ggml::hrx
