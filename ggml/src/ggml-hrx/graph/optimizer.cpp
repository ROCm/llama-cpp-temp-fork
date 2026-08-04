#include "optimizer.h"

#include <algorithm>
#include <set>

namespace ggml::hrx {
namespace {

struct Candidate {
    size_t rule = 0;
    OperationId root = kInvalidId;
    int priority = 0;
    std::vector<OperationId> operations;
};

class SelectionImplementation {
public:

static std::vector<OperationId> matched_operations(const Match & match) {
    return match.covered_operations;
}

};

} // namespace

Selection Selection::select(const Graph & graph, const std::vector<FusionRule> & rules) {
    std::vector<Candidate> candidates;
    for (size_t rule_id = 0; rule_id < rules.size(); ++rule_id) {
        for (OperationId root = 0; root < graph.operations.size(); ++root) {
            Match match = MatchAutomaton::match(graph, rules[rule_id].automaton, root);
            if (match.found()) {
                candidates.push_back({ rule_id, root, rules[rule_id].priority, SelectionImplementation::matched_operations(match) });
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate & lhs, const Candidate & rhs) {
        if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
        if (lhs.operations.size() != rhs.operations.size()) return lhs.operations.size() > rhs.operations.size();
        if (lhs.root != rhs.root) return lhs.root < rhs.root;
        return lhs.rule < rhs.rule;
    });

    Selection selection;
    std::vector<uint8_t> claimed(graph.operations.size(), 0);
    for (const Candidate & candidate : candidates) {
        const bool overlaps = std::any_of(candidate.operations.begin(), candidate.operations.end(), [&](OperationId operation) {
            return operation >= claimed.size() || claimed[operation] != 0;
        });
        if (overlaps) continue;
        for (OperationId operation : candidate.operations) claimed[operation] = 1;
        selection.regions.push_back({ candidate.rule, candidate.root, candidate.operations });
    }
    std::sort(selection.regions.begin(), selection.regions.end(), [](const SelectedRegion & lhs, const SelectedRegion & rhs) {
        return lhs.operations.front() < rhs.operations.front();
    });
    for (OperationId operation = 0; operation < claimed.size(); ++operation) {
        if (claimed[operation] == 0) selection.uncovered_operations.push_back(operation);
    }
    return selection;
}

Schedule Schedule::materialize(const Graph & graph, const std::vector<FusionRule> & rules, const Selection & selection) {
    Schedule schedule;
    schedule.graph_fingerprint = graph.fingerprint;
    std::vector<size_t> region_for_operation(graph.operations.size(), SIZE_MAX);
    for (size_t region_id = 0; region_id < selection.regions.size(); ++region_id) {
        for (OperationId operation : selection.regions[region_id].operations) {
            if (operation < region_for_operation.size()) region_for_operation[operation] = region_id;
        }
    }
    std::vector<std::vector<OperationId>> consumers(graph.values.size());
    for (const Operation & operation : graph.operations) {
        for (ValueId input : operation.inputs) {
            if (input < consumers.size()) consumers[input].push_back(operation.id);
        }
    }
    std::set<ValueId> roots(graph.roots.begin(), graph.roots.end());
    for (size_t region_id = 0; region_id < selection.regions.size(); ++region_id) {
        const SelectedRegion & region = selection.regions[region_id];
        if (region.rule >= rules.size()) continue;
        Invocation invocation;
        invocation.kernel = rules[region.rule].kernel;
        invocation.covered_operations = region.operations;
        std::set<ValueId> inputs;
        std::set<ValueId> outputs;
        for (OperationId operation_id : region.operations) {
            const Operation & operation = graph.operations[operation_id];
            for (ValueId input : operation.inputs) {
                OperationId producer = graph.values[input].producer;
                if (producer == kInvalidId || region_for_operation[producer] != region_id) inputs.insert(input);
            }
            bool escapes = roots.count(operation.output) != 0;
            for (OperationId consumer : consumers[operation.output]) escapes |= region_for_operation[consumer] != region_id;
            for (const Effect & effect : operation.effects) escapes |= effect.kind == EffectKind::Write;
            if (escapes) outputs.insert(operation.output);
        }
        size_t ordinal = 0;
        for (ValueId input : inputs) invocation.inputs.push_back({ "arg" + std::to_string(ordinal++), input });
        ordinal = 0;
        for (ValueId output : outputs) invocation.outputs.push_back({ "result" + std::to_string(ordinal++), output });
        Dispatch dispatch;
        dispatch.kernel = invocation.kernel;
        dispatch.bindings.insert(dispatch.bindings.end(), invocation.inputs.begin(), invocation.inputs.end());
        dispatch.bindings.insert(dispatch.bindings.end(), invocation.outputs.begin(), invocation.outputs.end());
        invocation.dispatches.push_back(std::move(dispatch));
        schedule.invocations.push_back(std::move(invocation));
    }
    return schedule;
}

Schedule Schedule::materialize_with_cpu_fallback(const Graph & graph, const std::vector<FusionRule> & rules, const Selection & selection) {
    std::vector<FusionRule> augmented_rules = rules;
    Selection augmented_selection = selection;
    for (OperationId operation : selection.uncovered_operations) {
        KernelSpecialization fallback;
        fallback.family = "cpu";
        fallback.variant = "reference_subgraph";
        fallback.integer_parameters["ggml_op"] = graph.operations[operation].op;
        fallback.execution_kind = KernelSpecialization::ExecutionKind::CpuFallback;
        const size_t rule = augmented_rules.size();
        augmented_rules.push_back({ {}, std::move(fallback), 0 });
        augmented_selection.regions.push_back({ rule, operation, { operation } });
    }
    augmented_selection.uncovered_operations.clear();
    std::sort(augmented_selection.regions.begin(), augmented_selection.regions.end(), [](const SelectedRegion & lhs, const SelectedRegion & rhs) {
        return lhs.operations.front() < rhs.operations.front();
    });
    return Schedule::materialize(graph, augmented_rules, augmented_selection);
}

} // namespace ggml::hrx
