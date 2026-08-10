#include "planner.h"

#include "domains/llm-recipes.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <set>

namespace ggml::hrx {

const char * schedule_capability_name(schedule_capability capability) {
    switch (capability) {
        case schedule_capability::SCHEDULE_CAPABILITY_PRIMITIVE:
            return "primitive";
        case schedule_capability::SCHEDULE_CAPABILITY_EMBEDDING:
            return "embedding";
        case schedule_capability::SCHEDULE_CAPABILITY_ATTENTION_PREPARE:
            return "attention_prepare";
        case schedule_capability::SCHEDULE_CAPABILITY_ATTENTION_QKV:
            return "attention_qkv";
        case schedule_capability::SCHEDULE_CAPABILITY_ATTENTION:
            return "attention";
        case schedule_capability::SCHEDULE_CAPABILITY_ATTENTION_OUTPUT:
            return "attention_output";
        case schedule_capability::SCHEDULE_CAPABILITY_ROUTER_SELECTION:
            return "router_selection";
        case schedule_capability::SCHEDULE_CAPABILITY_EXPERT_GATE_UP:
            return "expert_gate_up";
        case schedule_capability::SCHEDULE_CAPABILITY_EXPERT_DOWN:
            return "expert_down";
        case schedule_capability::SCHEDULE_CAPABILITY_EXPERT_DOWN_NEXT_PREPARE:
            return "expert_down_next_prepare";
        case schedule_capability::SCHEDULE_CAPABILITY_ENDPOINT:
            return "endpoint";
    }
    return "unknown";
}

bool planner::recover_facts(graph_plan & plan) {
    const attention_prepare_region * first_prepare        = nullptr;
    const attention_region *         first_attention      = nullptr;
    const attention_qkv_region *     first_qkv            = nullptr;
    const router_selection_region *  first_router         = nullptr;
    const embedding_region *         first_embedding      = nullptr;
    const expert_gate_up_region *    first_gate           = nullptr;
    size_t                           layer_count          = 0;
    bool                             has_llm_block_region = false;
    for (region_id id : plan.active_regions_) {
        const graph_region & region = plan.regions_[id];
        if (const auto * payload = std::get_if<embedding_region>(&region.payload)) {
            if (first_embedding == nullptr) {
                first_embedding = payload;
            }
        } else if (const auto * payload = std::get_if<attention_prepare_region>(&region.payload)) {
            has_llm_block_region = true;
            if (first_prepare == nullptr) {
                first_prepare = payload;
            }
        } else if (const auto * payload = std::get_if<attention_region>(&region.payload)) {
            has_llm_block_region = true;
            if (first_attention == nullptr) {
                first_attention = payload;
            }
            ++layer_count;
        } else if (const auto * payload = std::get_if<attention_qkv_region>(&region.payload)) {
            has_llm_block_region = true;
            if (first_qkv == nullptr) {
                first_qkv = payload;
            }
        } else if (const auto * payload = std::get_if<router_selection_region>(&region.payload)) {
            has_llm_block_region = true;
            if (first_router == nullptr) {
                first_router = payload;
            }
        } else if (const auto * payload = std::get_if<expert_gate_up_region>(&region.payload)) {
            if (first_gate == nullptr) {
                first_gate = payload;
            }
        }
    }
    if (!has_llm_block_region) {
        return true;
    }
    if (plan.groups_.size() != layer_count ||
        std::any_of(plan.groups_.begin(), plan.groups_.end(),
                    [](const semantic_group & group) { return group.members.size() != 7; })) {
        plan.add_error("LLM semantic regions do not form complete seven-component block groups");
        return false;
    }
    if (first_embedding == nullptr || first_prepare == nullptr || first_attention == nullptr || first_qkv == nullptr ||
        first_router == nullptr || first_gate == nullptr) {
        plan.add_error("LLM recipe facts require attention, QKV, router, and preparation regions");
        return false;
    }
    const graph_value & prepared         = plan.values_[first_prepare->prepared];
    const graph_op &    flash            = plan.operations_[first_attention->flash];
    const graph_op &    query_projection = plan.operations_[first_qkv->query_projection];
    const graph_op &    key_projection   = plan.operations_[first_qkv->key_projection];
    if (flash.inputs.size() < 3) {
        plan.add_error("attention fact recovery found an invalid flash operation");
        return false;
    }
    op_id               router_projection = ID_INVALID;
    const graph_value & route_ids         = plan.values_[first_router->route_ids];
    const graph_value & query             = plan.values_[first_qkv->query];
    const graph_value & key               = plan.values_[flash.inputs[1]];
    const size_t        route_type_size   = ggml_type_size(route_ids.type);
    float               rms_epsilon       = 0.0f;
    for (region_id id : plan.active_regions_) {
        const graph_region & region = plan.regions_[id];
        if (!std::holds_alternative<attention_prepare_region>(region.payload)) {
            continue;
        }
        for (op_id operation : region.source_ops) {
            if (plan.operations_[operation].op == GGML_OP_RMS_NORM) {
                std::memcpy(&rms_epsilon, plan.operations_[operation].raw_params.data(), sizeof(rms_epsilon));
                break;
            }
        }
        break;
    }
    uint32_t rms_epsilon_bits = 0;
    std::memcpy(&rms_epsilon_bits, &rms_epsilon, sizeof(rms_epsilon_bits));
    for (region_id id : plan.active_regions_) {
        const graph_region & region = plan.regions_[id];
        if (region.kind != region_kind::REGION_KIND_ROUTER_SELECTION) {
            continue;
        }
        const auto & payload = std::get<router_selection_region>(region.payload);
        if (payload.route_ids != first_router->route_ids) {
            continue;
        }
        for (op_id operation : region.source_ops) {
            if (plan.operations_[operation].op == GGML_OP_MUL_MAT) {
                router_projection = operation;
                break;
            }
        }
    }
    if (router_projection == ID_INVALID) {
        plan.add_error("router fact recovery found no projection");
        return false;
    }
    int64_t output_tokens = 1;
    for (value_id root : plan.roots_) {
        output_tokens = std::max(output_tokens, plan.values_[root].access.shape[1]);
    }
    plan.facts_ = {
        { "layer_count",              static_cast<int64_t>(layer_count)                                        },
        { "query_token_count",        prepared.access.shape[1]                                                 },
        { "output_token_count",       output_tokens                                                            },
        { "key_value_token_count",    plan.values_[flash.inputs[1]].access.shape[1]                            },
        { "hidden_size",              prepared.access.shape[0]                                                 },
        { "query_size",               plan.values_[query_projection.output].access.shape[0]                    },
        { "key_value_size",           plan.values_[key_projection.output].access.shape[0]                      },
        { "expert_count",             plan.values_[plan.operations_[router_projection].output].access.shape[0] },
        { "route_count",              route_ids.access.shape[0]                                                },
        { "route_stride",
         route_type_size == 0 ? 0 : static_cast<int64_t>(route_ids.access.strides[1] / route_type_size)        },
        { "head_size",                query.access.shape[0]                                                    },
        { "query_head_count",         query.access.shape[2]                                                    },
        { "key_value_head_count",     key.access.shape[2]                                                      },
        { "expert_intermediate_size", plan.values_[first_gate->activation].access.shape[0]                     },
        { "vocabulary_count",         plan.values_[first_embedding->weight].access.shape[1]                    },
        { "rms_epsilon_bits",         rms_epsilon_bits                                                         },
    };
    for (semantic_group & group : plan.groups_) {
        group.facts                  = plan.facts_;
        group.facts["block_ordinal"] = group.id;
    }

    for (region_id id : plan.active_regions_) {
        const graph_region & region = plan.regions_[id];
        if (const auto * payload = std::get_if<attention_prepare_region>(&region.payload)) {
            const graph_value & value = plan.values_[payload->prepared];
            if (value.access.shape[0] != plan.facts_["hidden_size"] ||
                value.access.shape[1] != plan.facts_["query_token_count"]) {
                plan.add_error("attention preparation facts disagree across regions");
            }
        } else if (const auto * payload = std::get_if<attention_region>(&region.payload)) {
            const graph_op & operation = plan.operations_[payload->flash];
            if (operation.inputs.size() < 3 ||
                plan.values_[operation.inputs[1]].access.shape[1] != plan.facts_["key_value_token_count"]) {
                plan.add_error("attention KV extent disagrees across regions");
            }
        } else if (const auto * payload = std::get_if<router_selection_region>(&region.payload)) {
            if (plan.values_[payload->route_ids].access.shape[0] != plan.facts_["route_count"]) {
                plan.add_error("router route count disagrees across regions");
            }
        }
    }
    return plan.valid();
}

std::vector<recipe_candidate> planner::enumerate_candidates(const graph_plan & plan) {
    std::vector<recipe_candidate> result;
    std::map<value_id, region_id> prepare_by_hidden;
    for (region_id id : plan.active_regions_) {
        const graph_region & region = plan.regions_[id];
        if (const auto * payload = std::get_if<attention_prepare_region>(&region.payload)) {
            prepare_by_hidden.emplace(payload->hidden, id);
        }
    }
    for (region_id id : plan.active_regions_) {
        const graph_region & region = plan.regions_[id];
        recipe_candidate     candidate;
        candidate.id         = static_cast<uint32_t>(result.size());
        candidate.capability = llm_recipes::capability(region.kind);
        candidate.regions    = { id };
        candidate.benefit    = region.kind == region_kind::REGION_KIND_ATOM ? 0 : 10;
        result.push_back(candidate);
        if (const auto * down = std::get_if<expert_down_region>(&region.payload)) {
            const auto next = prepare_by_hidden.find(down->hidden_output);
            if (next != prepare_by_hidden.end()) {
                recipe_candidate fused;
                fused.id                = static_cast<uint32_t>(result.size());
                fused.capability        = schedule_capability::SCHEDULE_CAPABILITY_EXPERT_DOWN_NEXT_PREPARE;
                fused.regions           = { id, next->second };
                fused.benefit           = 100;
                fused.provider_priority = 1;
                result.push_back(std::move(fused));
            }
        }
    }
    return result;
}

bool planner::commit_candidates(graph_plan & plan, std::vector<recipe_candidate> candidates) {
    auto better = [&](uint32_t lhs, uint32_t rhs) {
        const recipe_candidate & a = candidates[lhs];
        const recipe_candidate & b = candidates[rhs];
        if (a.benefit != b.benefit) {
            return a.benefit < b.benefit;
        }
        if (a.provider_priority != b.provider_priority) {
            return a.provider_priority < b.provider_priority;
        }
        const op_id a_first = plan.regions_[a.regions.front()].source_ops.front();
        const op_id b_first = plan.regions_[b.regions.front()].source_ops.front();
        if (a_first != b_first) {
            return a_first > b_first;
        }
        return a.id > b.id;
    };
    std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(better)> queue(better);
    std::vector<std::vector<uint32_t>>                                     by_region(plan.regions_.size());
    for (const recipe_candidate & candidate : candidates) {
        queue.push(candidate.id);
        for (region_id region : candidate.regions) {
            by_region[region].push_back(candidate.id);
        }
    }
    std::vector<bool> stale(candidates.size(), false);
    while (!queue.empty()) {
        const uint32_t candidate_id = queue.top();
        queue.pop();
        if (stale[candidate_id]) {
            continue;
        }
        recipe_candidate & candidate = candidates[candidate_id];
        if (!std::all_of(candidate.regions.begin(), candidate.regions.end(), [&](region_id region) {
                return region < plan.regions_.size() && plan.regions_[region].parent == ID_INVALID &&
                       plan.regions_[region].selected_recipe == ID_INVALID;
            })) {
            continue;
        }
        std::string rejection;
        if (!candidate_legal(plan, candidate, rejection)) {
            plan.diagnostics_.warnings.push_back("rejected recipe candidate " + std::to_string(candidate.id) + ": " +
                                                 rejection);
            continue;
        }
        selected_recipe recipe;
        recipe.id         = static_cast<recipe_id>(plan.selected_recipes_.size());
        recipe.capability = candidate.capability;
        recipe.regions    = candidate.regions;
        recipe.facts      = plan.facts_;
        for (const auto & fact : plan.facts_) {
            recipe_parameter parameter;
            parameter.name  = fact.first;
            parameter.value = fact.second;
            if (fact.first == "query_token_count" || fact.first == "key_value_token_count" ||
                fact.first == "output_token_count") {
                parameter.kind = recipe_parameter_kind::RECIPE_PARAMETER_KIND_COMMAND_RECORDING;
            }
            recipe.parameters.push_back(std::move(parameter));
        }
        plan.selected_recipes_.push_back(recipe);

        region_id selected_region = candidate.regions.front();
        if (candidate.regions.size() > 1) {
            graph_region parent;
            parent.id       = static_cast<region_id>(plan.regions_.size());
            parent.kind     = region_kind::REGION_KIND_FUSION;
            parent.payload  = fusion_region{ static_cast<uint32_t>(candidate.capability) };
            parent.children = candidate.regions;
            std::set<op_id>                                                                         operations;
            std::set<std::tuple<effect_kind, storage_id, uint32_t, uint32_t, size_t, size_t, bool>> effects;
            for (region_id child : candidate.regions) {
                plan.regions_[child].parent = parent.id;
                operations.insert(plan.regions_[child].source_ops.begin(), plan.regions_[child].source_ops.end());
                for (const graph_effect & effect : plan.regions_[child].effects) {
                    effects.emplace(effect.kind, effect.storage, effect.before_version, effect.after_version,
                                    effect.offset, effect.size, effect.exact);
                }
            }
            parent.source_ops.assign(operations.begin(), operations.end());
            for (const auto & effect : effects) {
                parent.effects.push_back({ std::get<0>(effect), std::get<1>(effect), std::get<2>(effect),
                                           std::get<3>(effect), std::get<4>(effect), std::get<5>(effect),
                                           std::get<6>(effect) });
            }
            parent.boundary        = plan.calculate_boundary(parent.source_ops);
            parent.selected_recipe = recipe.id;
            selected_region        = parent.id;
            plan.regions_.push_back(std::move(parent));
            by_region.emplace_back();
        } else {
            plan.regions_[selected_region].selected_recipe = recipe.id;
        }
        for (region_id covered : candidate.regions) {
            for (uint32_t overlap : by_region[covered]) {
                stale[overlap] = true;
            }
        }
    }
    plan.active_regions_.clear();
    for (const graph_region & region : plan.regions_) {
        if (region.parent == ID_INVALID) {
            plan.active_regions_.push_back(region.id);
        }
    }
    std::sort(plan.active_regions_.begin(), plan.active_regions_.end(), [&](region_id lhs, region_id rhs) {
        return plan.regions_[lhs].source_ops.front() < plan.regions_[rhs].source_ops.front();
    });
    return true;
}

bool planner::candidate_legal(const graph_plan & plan, const recipe_candidate & candidate, std::string & reason) {
    if (candidate.regions.empty()) {
        reason = "candidate has no regions";
        return false;
    }
    std::set<op_id> operations;
    for (region_id region : candidate.regions) {
        if (region >= plan.regions_.size()) {
            reason = "candidate references an invalid region";
            return false;
        }
        operations.insert(plan.regions_[region].source_ops.begin(), plan.regions_[region].source_ops.end());
    }
    for (op_id operation : operations) {
        if (plan.operations_[operation].storage_semantics ==
            operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_UNKNOWN) {
            reason = "candidate contains an operation with unknown storage semantics";
            return false;
        }
    }
    if (candidate.regions.size() == 1) {
        return true;
    }

    std::set<op_id>   visited;
    std::queue<op_id> worklist;
    worklist.push(*operations.begin());
    visited.insert(*operations.begin());
    while (!worklist.empty()) {
        const op_id current = worklist.front();
        worklist.pop();
        auto visit = [&](op_id adjacent) {
            if (operations.count(adjacent) != 0 && visited.insert(adjacent).second) {
                worklist.push(adjacent);
            }
        };
        for (op_id predecessor : plan.predecessors_[current]) {
            visit(predecessor);
        }
        for (op_id successor : plan.successors_[current]) {
            visit(successor);
        }
    }
    if (visited.size() != operations.size()) {
        reason = "candidate source operations are disconnected";
        return false;
    }

    // A contraction creates a cycle exactly when a path leaves the candidate
    // and later re-enters it. Operation IDs are topological, so only the
    // candidate's local ordinal span can participate. This convexity check is
    // linear in that span instead of walking the whole unrolled graph once per
    // layer candidate.
    const op_id       first = *operations.begin();
    const op_id       last  = *operations.rbegin();
    std::vector<bool> reachable_from_candidate(last - first + 1, false);
    for (op_id operation = first; operation <= last; ++operation) {
        if (operations.count(operation) != 0) {
            continue;
        }
        bool reachable = false;
        for (op_id predecessor : plan.predecessors_[operation]) {
            if (operations.count(predecessor) != 0 ||
                (predecessor >= first && reachable_from_candidate[predecessor - first])) {
                reachable = true;
                break;
            }
        }
        reachable_from_candidate[operation - first] = reachable;
        if (reachable && std::any_of(plan.successors_[operation].begin(), plan.successors_[operation].end(),
                                     [&](op_id successor) { return operations.count(successor) != 0; })) {
            reason = "candidate source operations are not convex";
            return false;
        }
    }
    return true;
}

bool planner::select_recipes(graph_plan & plan, const std::string & target) {
    if (plan.phase_ != plan_phase::PLAN_PHASE_RECOGNIZED || target.empty()) {
        plan.add_error("planner requires a recognized graph plan and target");
        return false;
    }
    if (!recover_facts(plan) || !commit_candidates(plan, enumerate_candidates(plan))) {
        return false;
    }
    plan.phase_ = plan_phase::PLAN_PHASE_PLANNED;
    return plan.verify(true);
}

}  // namespace ggml::hrx
