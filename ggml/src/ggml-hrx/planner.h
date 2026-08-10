#pragma once

#include "graph-plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct recipe_candidate {
    uint32_t               id         = ID_INVALID;
    schedule_capability    capability = schedule_capability::SCHEDULE_CAPABILITY_PRIMITIVE;
    std::vector<region_id> regions;
    int64_t                benefit           = 0;
    uint32_t               provider_priority = 0;
};

class planner {
  public:
    static bool select_recipes(graph_plan & plan, const std::string & target);

  private:
    static bool                          recover_facts(graph_plan & plan);
    static std::vector<recipe_candidate> enumerate_candidates(const graph_plan & plan);
    static bool                          commit_candidates(graph_plan & plan, std::vector<recipe_candidate> candidates);
    static bool candidate_legal(const graph_plan & plan, const recipe_candidate & candidate, std::string & reason);
};

}  // namespace ggml::hrx
