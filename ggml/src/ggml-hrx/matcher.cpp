#include "matcher.h"

#include <algorithm>
#include <set>
#include <sstream>

namespace ggml::hrx {

match_result match_result::no_match() {
    return {};
}

match_result match_result::matched(region_match match) {
    match_result result;
    result.status = match_status::MATCH_STATUS_MATCHED;
    result.match  = std::move(match);
    return result;
}

match_result match_result::malformed(std::string diagnostic, op_id anchor) {
    match_result result;
    result.status       = match_status::MATCH_STATUS_MALFORMED;
    result.match.anchor = anchor;
    result.diagnostic   = std::move(diagnostic);
    return result;
}

match_cursor::match_cursor(const graph_plan & plan, size_t budget) : plan_(plan), budget_(budget) {}

bool match_cursor::probe() {
    if (probes_ == budget_) {
        exhausted_ = true;
        return false;
    }
    ++probes_;
    return true;
}

const graph_op * match_cursor::op(op_id id) {
    if (!probe() || id >= plan_.operations().size()) {
        return nullptr;
    }
    return &plan_.operations()[id];
}

const graph_value * match_cursor::value(value_id id) {
    if (!probe() || id >= plan_.values().size()) {
        return nullptr;
    }
    return &plan_.values()[id];
}

op_id match_cursor::producer(value_id value_id, enum ggml_op kind) {
    const graph_value * graph_value = value(value_id);
    if (graph_value == nullptr || graph_value->producer == ID_INVALID) {
        return ID_INVALID;
    }
    const graph_op * producer_op = op(graph_value->producer);
    if (producer_op == nullptr || (kind != GGML_OP_COUNT && producer_op->op != kind)) {
        return ID_INVALID;
    }
    return producer_op->id;
}

std::vector<op_id> match_cursor::consumers(value_id value_id, enum ggml_op kind) {
    std::vector<op_id> result;
    if (!probe() || value_id >= plan_.values().size()) {
        return result;
    }
    for (op_id consumer : plan_.consumers(value_id)) {
        if (kind == GGML_OP_COUNT) {
            result.push_back(consumer);
            continue;
        }
        const graph_op * consumer_op = op(consumer);
        if (consumer_op != nullptr && consumer_op->op == kind) {
            result.push_back(consumer);
        }
    }
    return result;
}

op_id match_cursor::unique_consumer(value_id value_id, enum ggml_op kind) {
    const std::vector<op_id> users = consumers(value_id, kind);
    return users.size() == 1 ? users.front() : ID_INVALID;
}

value_id match_cursor::output(op_id operation) {
    const graph_op * operation_value = op(operation);
    return operation_value == nullptr ? ID_INVALID : operation_value->output;
}

bool match_cursor::is_root(value_id value_id) {
    if (!probe()) {
        return false;
    }
    return std::find(plan_.roots().begin(), plan_.roots().end(), value_id) != plan_.roots().end();
}

std::vector<op_id> match_cursor::chain(value_id value_id, const std::vector<enum ggml_op> & kinds) {
    std::vector<op_id> result;
    for (enum ggml_op kind : kinds) {
        const op_id operation = unique_consumer(value_id, kind);
        if (operation == ID_INVALID) {
            return {};
        }
        result.push_back(operation);
        value_id = output(operation);
    }
    return result;
}

void pattern_registry::add(pattern_rule rule) {
    rules_.push_back(rule);
}

void pattern_registry::add(semantic_group_callback callback) {
    group_callbacks_.push_back(callback);
}

bool matcher::recognize(graph_plan & plan, const pattern_registry & registry) {
    if (plan.phase_ != plan_phase::PLAN_PHASE_IMPORTED || !plan.valid()) {
        plan.add_error("matcher requires a valid imported graph plan");
        return false;
    }

    std::vector<region_match> matches;
    for (const pattern_rule & rule : registry.rules()) {
        if (rule.callback == nullptr || rule.anchor >= GGML_OP_COUNT) {
            plan.add_error("pattern registry contains an invalid rule");
            continue;
        }
        for (op_id anchor : plan.operations_by_kind(rule.anchor)) {
            match_cursor cursor(plan, rule.probe_budget);
            match_result result = rule.callback(cursor, anchor);
            plan.diagnostics_.matcher_probes += cursor.probes();
            if (cursor.exhausted()) {
                plan.add_error(std::string("pattern '") + rule.name + "' exceeded its probe budget at op " +
                               std::to_string(anchor));
                continue;
            }
            if (result.status == match_status::MATCH_STATUS_MALFORMED) {
                plan.add_error(std::string("pattern '") + rule.name + "' is malformed at op " + std::to_string(anchor) +
                               ": " + result.diagnostic);
            } else if (result.status == match_status::MATCH_STATUS_MATCHED) {
                matches.push_back(std::move(result.match));
            }
        }
    }
    if (!plan.valid() || !commit(plan, std::move(matches)) || !commit_groups(plan, registry)) {
        return false;
    }
    plan.phase_ = plan_phase::PLAN_PHASE_RECOGNIZED;
    return plan.verify(true);
}

bool matcher::commit_groups(graph_plan & plan, const pattern_registry & registry) {
    std::vector<uint32_t> membership(plan.regions_.size(), 0);
    for (semantic_group_callback callback : registry.group_callbacks()) {
        if (callback == nullptr) {
            plan.add_error("pattern registry contains a null semantic-group callback");
            continue;
        }
        for (semantic_group_match match : callback(plan)) {
            if (match.members.empty()) {
                plan.add_error("semantic group contains no regions");
                continue;
            }
            semantic_group group;
            group.id      = static_cast<group_id>(plan.groups_.size());
            group.kind    = match.kind;
            group.members = std::move(match.members);
            group.facts   = std::move(match.facts);
            for (region_id member : group.members) {
                if (member >= plan.regions_.size() || plan.regions_[member].kind == region_kind::REGION_KIND_ATOM) {
                    plan.add_error("semantic group references an invalid or atom region");
                    continue;
                }
                if (++membership[member] != 1) {
                    plan.add_error("semantic region belongs to multiple semantic groups");
                }
            }
            plan.groups_.push_back(std::move(group));
        }
    }
    std::sort(plan.groups_.begin(), plan.groups_.end(), [&](const semantic_group & lhs, const semantic_group & rhs) {
        return plan.regions_[lhs.members.front()].source_ops.front() <
               plan.regions_[rhs.members.front()].source_ops.front();
    });
    for (size_t i = 0; i < plan.groups_.size(); ++i) {
        plan.groups_[i].id       = static_cast<group_id>(i);
        plan.groups_[i].previous = i == 0 ? ID_INVALID : static_cast<group_id>(i - 1);
        plan.groups_[i].next     = i + 1 == plan.groups_.size() ? ID_INVALID : static_cast<group_id>(i + 1);
    }
    return plan.valid();
}

bool matcher::commit(graph_plan & plan, std::vector<region_match> matches) {
    for (region_match & match : matches) {
        std::sort(match.operations.begin(), match.operations.end());
        match.operations.erase(std::unique(match.operations.begin(), match.operations.end()), match.operations.end());
    }
    std::sort(matches.begin(), matches.end(), [](const region_match & lhs, const region_match & rhs) {
        const op_id lhs_first = lhs.operations.empty() ? ID_INVALID : lhs.operations.front();
        const op_id rhs_first = rhs.operations.empty() ? ID_INVALID : rhs.operations.front();
        if (lhs_first != rhs_first) {
            return lhs_first < rhs_first;
        }
        if (lhs.kind != rhs.kind) {
            return lhs.kind < rhs.kind;
        }
        return lhs.anchor < rhs.anchor;
    });

    std::vector<region_match> unique_matches;
    for (region_match & match : matches) {
        if (!unique_matches.empty() && unique_matches.back().kind == match.kind &&
            unique_matches.back().operations == match.operations) {
            continue;
        }
        unique_matches.push_back(std::move(match));
    }

    std::vector<region_id> semantic_owner(plan.operations_.size(), ID_INVALID);
    for (const region_match & match : unique_matches) {
        if (match.operations.empty()) {
            plan.add_error("semantic match covers no operations");
            continue;
        }
        for (op_id operation : match.operations) {
            if (operation >= semantic_owner.size()) {
                plan.add_error("semantic match references an invalid operation");
                continue;
            }
            if (semantic_owner[operation] != ID_INVALID) {
                std::ostringstream error;
                error << "semantic matches overlap at op " << operation;
                plan.add_error(error.str());
            }
            semantic_owner[operation] = static_cast<region_id>(&match - unique_matches.data());
        }
    }
    if (!plan.valid()) {
        return false;
    }

    for (region_match & match : unique_matches) {
        graph_region region;
        region.id         = static_cast<region_id>(plan.regions_.size());
        region.kind       = match.kind;
        region.payload    = std::move(match.payload);
        region.source_ops = std::move(match.operations);
        region.boundary   = plan.calculate_boundary(region.source_ops);
        std::set<std::tuple<effect_kind, storage_id, uint32_t, uint32_t, size_t, size_t, bool>> effects;
        for (op_id operation : region.source_ops) {
            const region_id child = plan.source_op_owner_[operation];
            if (child >= plan.regions_.size() || plan.regions_[child].parent != ID_INVALID) {
                plan.add_error("semantic match does not own active atom children");
                continue;
            }
            region.children.push_back(child);
            plan.regions_[child].parent = region.id;
            for (const graph_effect & effect : plan.operations_[operation].effects) {
                effects.emplace(effect.kind, effect.storage, effect.before_version, effect.after_version, effect.offset,
                                effect.size, effect.exact);
            }
            plan.source_op_owner_[operation] = region.id;
        }
        for (const auto & item : effects) {
            region.effects.push_back({ std::get<0>(item), std::get<1>(item), std::get<2>(item), std::get<3>(item),
                                       std::get<4>(item), std::get<5>(item), std::get<6>(item) });
        }
        plan.regions_.push_back(std::move(region));
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
    return plan.valid();
}

}  // namespace ggml::hrx
