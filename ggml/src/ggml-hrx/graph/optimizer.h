#pragma once

#include "matcher.h"
#include "schedule.h"

#include <vector>

namespace ggml::hrx {

struct FusionRule {
    MatchAutomaton automaton;
    KernelSpecialization kernel;
    int priority = 0;

    static std::vector<FusionRule> canonical_qwen3_moe_rules();
};

struct SelectedRegion {
    size_t rule = 0;
    OperationId root = kInvalidId;
    std::vector<OperationId> operations;
};

struct Selection {
    std::vector<SelectedRegion> regions;
    std::vector<OperationId> uncovered_operations;

    static Selection select(const Graph & graph, const std::vector<FusionRule> & rules);
};

} // namespace ggml::hrx
