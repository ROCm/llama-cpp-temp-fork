#pragma once

#include "graph-ir.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

struct Match;

struct InputTypeConstraint {
    size_t input_index = 0;
    std::vector<enum ggml_type> alternatives;
};

struct OpConstraint {
    std::vector<enum ggml_op> alternatives;
    enum ggml_type output_type = GGML_TYPE_COUNT;
    int rank = -1;
    std::vector<int64_t> shape;
    std::vector<InputTypeConstraint> input_types;

    OpConstraint() = default;
    OpConstraint(std::vector<enum ggml_op> alternatives, enum ggml_type output_type = GGML_TYPE_COUNT, int rank = -1,
                 std::vector<int64_t> shape = {}, std::vector<InputTypeConstraint> input_types = {})
        : alternatives(std::move(alternatives)), output_type(output_type), rank(rank),
          shape(std::move(shape)), input_types(std::move(input_types)) {}
};

struct Transition {
    enum class Kind : uint8_t {
        InputProducer,
        StorageWriter,
        OutputConsumer,
    };

    Kind kind = Kind::InputProducer;
    size_t input_index = 0;
    size_t target_state = 0;

    Transition() = default;
    Transition(Kind kind, size_t input_index, size_t target_state)
        : kind(kind), input_index(input_index), target_state(target_state) {}
};

struct MatchState {
    std::string name;
    OpConstraint constraint;
    std::vector<Transition> transitions;
    bool capture = true;
};

struct MatchAutomaton {
    std::string name;
    size_t root_state = 0;
    std::vector<MatchState> states;
    bool require_internal_single_use = true;

    static Match match(const Graph & graph, const MatchAutomaton & automaton, OperationId root);
};

struct Match {
    std::string automaton;
    std::vector<OperationId> state_operations;
    std::vector<OperationId> covered_operations;

    bool found() const { return !state_operations.empty(); }
};

} // namespace ggml::hrx
