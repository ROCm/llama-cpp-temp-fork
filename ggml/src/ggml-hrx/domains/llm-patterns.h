#pragma once

#include "matcher.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace ggml::hrx {
namespace llm_patterns {

inline bool contains_value(const graph_op & operation, value_id value) {
    return std::find(operation.inputs.begin(), operation.inputs.end(), value) != operation.inputs.end();
}

inline match_result embedding(match_cursor & cursor, op_id anchor) {
    const graph_op * operation = cursor.op(anchor);
    if (operation == nullptr || operation->inputs.size() != 2) {
        return match_result::no_match();
    }
    const graph_value * weight = cursor.value(operation->inputs[0]);
    const graph_value * ids    = cursor.value(operation->inputs[1]);
    if (weight == nullptr || ids == nullptr || weight->boundary != boundary_kind::BOUNDARY_KIND_WEIGHT ||
        ids->boundary != boundary_kind::BOUNDARY_KIND_INPUT) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_EMBEDDING;
    match.anchor     = anchor;
    match.operations = { anchor };
    match.payload    = embedding_region{ operation->inputs[1], operation->inputs[0], operation->output };
    return match_result::matched(std::move(match));
}

inline match_result attention_prepare(match_cursor & cursor, op_id anchor) {
    const graph_op * operation = cursor.op(anchor);
    if (operation == nullptr || operation->inputs.size() != 2) {
        return match_result::no_match();
    }
    op_id    norm  = cursor.producer(operation->inputs[0], GGML_OP_RMS_NORM);
    value_id scale = operation->inputs[1];
    if (norm == ID_INVALID) {
        norm  = cursor.producer(operation->inputs[1], GGML_OP_RMS_NORM);
        scale = operation->inputs[0];
    }
    if (norm == ID_INVALID || cursor.consumers(operation->output, GGML_OP_MUL_MAT).size() != 3) {
        return match_result::no_match();
    }
    const graph_op * norm_op = cursor.op(norm);
    if (norm_op == nullptr || norm_op->inputs.empty()) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_ATTENTION_PREPARE;
    match.anchor     = anchor;
    match.operations = { norm, anchor };
    match.payload    = attention_prepare_region{ norm_op->inputs[0], scale, operation->output };
    return match_result::matched(std::move(match));
}

inline match_result attention_qkv(match_cursor & cursor, op_id anchor) {
    const value_id anchor_output = cursor.output(anchor);
    if (anchor_output == ID_INVALID) {
        return match_result::no_match();
    }
    bool feeds_attention = false;
    for (op_id view : cursor.consumers(anchor_output, GGML_OP_VIEW)) {
        const op_id permute = cursor.unique_consumer(cursor.output(view), GGML_OP_PERMUTE);
        if (permute != ID_INVALID &&
            cursor.unique_consumer(cursor.output(permute), GGML_OP_FLASH_ATTN_EXT) != ID_INVALID) {
            feeds_attention = true;
            break;
        }
    }
    if (!feeds_attention) {
        return match_result::no_match();
    }

    const graph_op * rope_op = cursor.op(anchor);
    if (rope_op == nullptr || rope_op->inputs.empty()) {
        return match_result::no_match();
    }
    const op_id      scale    = cursor.producer(rope_op->inputs[0], GGML_OP_MUL);
    const graph_op * scale_op = cursor.op(scale);
    if (scale_op == nullptr) {
        return match_result::no_match();
    }
    op_id norm = ID_INVALID;
    for (value_id input : scale_op->inputs) {
        norm = cursor.producer(input, GGML_OP_RMS_NORM);
        if (norm != ID_INVALID) {
            break;
        }
    }
    const graph_op * norm_op = cursor.op(norm);
    if (norm_op == nullptr || norm_op->inputs.empty()) {
        return match_result::no_match();
    }
    const op_id      query_reshape = cursor.producer(norm_op->inputs[0], GGML_OP_RESHAPE);
    const graph_op * reshape_op    = cursor.op(query_reshape);
    if (reshape_op == nullptr || reshape_op->inputs.empty()) {
        return match_result::no_match();
    }
    const op_id      query_projection    = cursor.producer(reshape_op->inputs[0], GGML_OP_MUL_MAT);
    const graph_op * query_projection_op = cursor.op(query_projection);
    if (query_projection_op == nullptr) {
        return match_result::no_match();
    }
    value_id prepared = ID_INVALID;
    for (value_id input : query_projection_op->inputs) {
        const graph_value * input_value = cursor.value(input);
        if (input_value != nullptr && input_value->boundary != boundary_kind::BOUNDARY_KIND_WEIGHT) {
            if (prepared != ID_INVALID) {
                return match_result::no_match();
            }
            prepared = input;
        }
    }
    const std::vector<op_id> projections = cursor.consumers(prepared, GGML_OP_MUL_MAT);
    if (prepared == ID_INVALID || projections.size() != 3) {
        return match_result::no_match();
    }

    std::set<op_id> owned;
    value_id        query            = ID_INVALID;
    op_id           key_projection   = ID_INVALID;
    op_id           value_projection = ID_INVALID;
    storage_id      key_cache        = ID_INVALID;
    storage_id      value_cache      = ID_INVALID;
    for (op_id projection : projections) {
        const op_id reshape = cursor.unique_consumer(cursor.output(projection), GGML_OP_RESHAPE);
        if (reshape == ID_INVALID) {
            return match_result::no_match();
        }
        const value_id reshape_output = cursor.output(reshape);
        const op_id    branch_norm    = cursor.unique_consumer(reshape_output, GGML_OP_RMS_NORM);
        if (branch_norm != ID_INVALID) {
            const std::vector<op_id> tail = cursor.chain(cursor.output(branch_norm), { GGML_OP_MUL, GGML_OP_ROPE });
            if (tail.size() != 2) {
                return match_result::no_match();
            }
            std::vector<std::pair<op_id, op_id>> writers;
            for (op_id view : cursor.consumers(cursor.output(tail.back()), GGML_OP_VIEW)) {
                const op_id writer = cursor.unique_consumer(cursor.output(view), GGML_OP_SET_ROWS);
                if (writer != ID_INVALID) {
                    writers.emplace_back(view, writer);
                }
            }
            owned.insert({ projection, reshape, branch_norm, tail[0], tail[1] });
            if (writers.empty() && query == ID_INVALID) {
                query = cursor.output(tail.back());
            } else if (writers.size() == 1) {
                owned.insert(writers.front().first);
                owned.insert(writers.front().second);
                key_projection                   = projection;
                const graph_value * writer_value = cursor.value(cursor.output(writers.front().second));
                if (writer_value == nullptr) {
                    return match_result::no_match();
                }
                key_cache = writer_value->access.storage;
            } else {
                return match_result::no_match();
            }
        } else {
            const std::vector<op_id> tail = cursor.chain(reshape_output, { GGML_OP_VIEW, GGML_OP_SET_ROWS });
            if (tail.size() != 2) {
                return match_result::no_match();
            }
            owned.insert({ projection, reshape, tail[0], tail[1] });
            value_projection                 = projection;
            const graph_value * writer_value = cursor.value(cursor.output(tail.back()));
            if (writer_value == nullptr) {
                return match_result::no_match();
            }
            value_cache = writer_value->access.storage;
        }
    }
    if (owned.size() != 16 || query == ID_INVALID || key_projection == ID_INVALID || value_projection == ID_INVALID ||
        key_cache == value_cache) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_ATTENTION_QKV;
    match.anchor     = anchor;
    match.operations = { owned.begin(), owned.end() };
    match.payload    = attention_qkv_region{ prepared,         query,          key_cache,       value_cache,
                                          query_projection, key_projection, value_projection };
    return match_result::matched(std::move(match));
}

inline match_result attention(match_cursor & cursor, op_id anchor) {
    const graph_op * operation = cursor.op(anchor);
    if (operation == nullptr || operation->inputs.size() != 4) {
        return match_result::no_match();
    }
    std::set<op_id>       owned{ anchor };
    std::vector<value_id> sources;
    for (size_t i = 0; i < 3; ++i) {
        const op_id      permute    = cursor.producer(operation->inputs[i], GGML_OP_PERMUTE);
        const graph_op * permute_op = cursor.op(permute);
        if (permute_op == nullptr || permute_op->inputs.empty()) {
            return match_result::no_match();
        }
        const op_id      view    = cursor.producer(permute_op->inputs[0], GGML_OP_VIEW);
        const graph_op * view_op = cursor.op(view);
        if (view_op == nullptr || view_op->inputs.empty()) {
            return match_result::no_match();
        }
        owned.insert(view);
        owned.insert(permute);
        sources.push_back(view_op->inputs[0]);
    }
    const op_id reshape = cursor.unique_consumer(operation->output, GGML_OP_RESHAPE);
    if (reshape == ID_INVALID) {
        return match_result::no_match();
    }
    owned.insert(reshape);
    const graph_value * key   = cursor.value(sources[1]);
    const graph_value * value = cursor.value(sources[2]);
    if (owned.size() != 8 || key == nullptr || value == nullptr) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_ATTENTION;
    match.anchor     = anchor;
    match.operations = { owned.begin(), owned.end() };
    match.payload    = attention_region{
        anchor, sources[0], key->access.storage, value->access.storage, operation->inputs[3], cursor.output(reshape)
    };
    return match_result::matched(std::move(match));
}

inline match_result attention_output(match_cursor & cursor, op_id anchor) {
    const graph_op * operation = cursor.op(anchor);
    if (operation == nullptr) {
        return match_result::no_match();
    }
    op_id flash = ID_INVALID;
    for (value_id input : operation->inputs) {
        const op_id      reshape    = cursor.producer(input, GGML_OP_RESHAPE);
        const graph_op * reshape_op = cursor.op(reshape);
        if (reshape_op != nullptr && !reshape_op->inputs.empty()) {
            flash = cursor.producer(reshape_op->inputs[0], GGML_OP_FLASH_ATTN_EXT);
            if (flash != ID_INVALID) {
                break;
            }
        }
    }
    if (flash == ID_INVALID) {
        return match_result::no_match();
    }
    std::set<op_id>          owned{ anchor };
    value_id                 selected        = operation->output;
    value_id                 residual_hidden = ID_INVALID;
    value_id                 selection_ids   = ID_INVALID;
    const std::vector<op_id> selections      = cursor.consumers(operation->output, GGML_OP_GET_ROWS);
    op_id                    residual        = ID_INVALID;
    if (selections.size() == 1) {
        const op_id attention_selection = selections.front();
        selected                        = cursor.output(attention_selection);
        residual                        = cursor.unique_consumer(selected, GGML_OP_ADD);
        const graph_op * residual_op    = cursor.op(residual);
        const graph_op * selection_op   = cursor.op(attention_selection);
        if (residual_op == nullptr || selection_op == nullptr || selection_op->inputs.size() < 2) {
            return match_result::no_match();
        }
        const value_id   other = residual_op->inputs[0] == selected ? residual_op->inputs[1] : residual_op->inputs[0];
        const op_id      hidden_selection    = cursor.producer(other, GGML_OP_GET_ROWS);
        const graph_op * hidden_selection_op = cursor.op(hidden_selection);
        if (hidden_selection_op == nullptr || hidden_selection_op->inputs.size() < 2 ||
            hidden_selection_op->inputs[1] != selection_op->inputs[1]) {
            return match_result::no_match();
        }
        residual_hidden = hidden_selection_op->inputs[0];
        selection_ids   = selection_op->inputs[1];
        owned.insert(attention_selection);
        owned.insert(hidden_selection);
    } else if (selections.empty()) {
        residual                     = cursor.unique_consumer(operation->output, GGML_OP_ADD);
        const graph_op * residual_op = cursor.op(residual);
        if (residual_op == nullptr) {
            return match_result::no_match();
        }
        residual_hidden = residual_op->inputs[0] == operation->output ? residual_op->inputs[1] : residual_op->inputs[0];
    } else {
        return match_result::no_match();
    }
    const std::vector<op_id> tail = cursor.chain(cursor.output(residual), { GGML_OP_RMS_NORM, GGML_OP_MUL });
    if (tail.size() != 2) {
        return match_result::no_match();
    }
    owned.insert(residual);
    owned.insert(tail.begin(), tail.end());
    region_match match;
    match.kind                    = region_kind::REGION_KIND_ATTENTION_OUTPUT;
    match.anchor                  = anchor;
    match.operations              = { owned.begin(), owned.end() };
    const op_id attention_reshape = cursor.unique_consumer(cursor.output(flash), GGML_OP_RESHAPE);
    if (attention_reshape == ID_INVALID) {
        return match_result::no_match();
    }
    match.payload = attention_output_region{ cursor.output(attention_reshape), residual_hidden,
                                             cursor.output(tail.back()), selection_ids };
    return match_result::matched(std::move(match));
}

inline match_result router_selection(match_cursor & cursor, op_id anchor) {
    const graph_op * argsort = cursor.op(anchor);
    if (argsort == nullptr || argsort->inputs.size() != 1) {
        return match_result::no_match();
    }
    const op_id      softmax    = cursor.producer(argsort->inputs[0], GGML_OP_SOFT_MAX);
    const graph_op * softmax_op = cursor.op(softmax);
    if (softmax_op == nullptr || softmax_op->inputs.empty()) {
        return match_result::no_match();
    }
    const op_id      projection    = cursor.producer(softmax_op->inputs[0], GGML_OP_MUL_MAT);
    const graph_op * projection_op = cursor.op(projection);
    const op_id      ids           = cursor.unique_consumer(argsort->output, GGML_OP_VIEW);
    const op_id      score_reshape = cursor.unique_consumer(softmax_op->output, GGML_OP_RESHAPE);
    if (projection_op == nullptr || projection_op->inputs.size() < 2 || ids == ID_INVALID ||
        score_reshape == ID_INVALID) {
        return match_result::no_match();
    }
    op_id gathered = ID_INVALID;
    for (op_id candidate : cursor.consumers(cursor.output(score_reshape), GGML_OP_GET_ROWS)) {
        const graph_op * candidate_op = cursor.op(candidate);
        if (candidate_op != nullptr && contains_value(*candidate_op, cursor.output(ids))) {
            gathered = candidate;
            break;
        }
    }
    if (gathered == ID_INVALID) {
        return match_result::no_match();
    }
    const std::vector<op_id> tail =
        cursor.chain(cursor.output(gathered), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP });
    if (tail.size() != 3) {
        return match_result::no_match();
    }
    op_id divide = ID_INVALID;
    for (op_id candidate : cursor.consumers(cursor.output(tail[0]), GGML_OP_DIV)) {
        const graph_op * candidate_op = cursor.op(candidate);
        if (candidate_op != nullptr && contains_value(*candidate_op, cursor.output(tail[2]))) {
            divide = candidate;
            break;
        }
    }
    const op_id weights = cursor.unique_consumer(cursor.output(divide), GGML_OP_RESHAPE);
    if (divide == ID_INVALID || weights == ID_INVALID) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_ROUTER_SELECTION;
    match.anchor     = anchor;
    match.operations = { projection, softmax, anchor,  ids,    score_reshape, gathered,
                         tail[0],    tail[1], tail[2], divide, weights };
    match.payload    = router_selection_region{ projection_op->inputs[1], cursor.output(ids), cursor.output(weights) };
    return match_result::matched(std::move(match));
}

inline match_result expert_gate_up(match_cursor & cursor, op_id anchor) {
    const graph_op * operation = cursor.op(anchor);
    if (operation == nullptr || operation->inputs.size() < 2) {
        return match_result::no_match();
    }
    const op_id      gate    = cursor.producer(operation->inputs[0], GGML_OP_MUL_MAT_ID);
    const op_id      up      = cursor.producer(operation->inputs[1], GGML_OP_MUL_MAT_ID);
    const graph_op * gate_op = cursor.op(gate);
    const graph_op * up_op   = cursor.op(up);
    if (gate_op == nullptr || up_op == nullptr) {
        return match_result::no_match();
    }
    value_id route_ids = ID_INVALID;
    value_id prepared  = ID_INVALID;
    for (value_id gate_input : gate_op->inputs) {
        if (!contains_value(*up_op, gate_input)) {
            continue;
        }
        const graph_value * common = cursor.value(gate_input);
        if (common == nullptr) {
            return match_result::no_match();
        }
        if (common->type == GGML_TYPE_I32) {
            route_ids = gate_input;
        } else {
            prepared = gate_input;
        }
    }
    const op_id reshape = cursor.producer(prepared, GGML_OP_RESHAPE);
    if (route_ids == ID_INVALID || prepared == ID_INVALID || reshape == ID_INVALID) {
        return match_result::no_match();
    }
    const graph_op * reshape_op = cursor.op(reshape);
    if (reshape_op == nullptr || reshape_op->inputs.empty()) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_EXPERT_GATE_UP;
    match.anchor     = anchor;
    match.operations = { reshape, gate, up, anchor };
    match.payload    = expert_gate_up_region{ reshape_op->inputs[0], route_ids, operation->output };
    return match_result::matched(std::move(match));
}

inline match_result expert_down(match_cursor & cursor, op_id anchor) {
    const graph_op * operation = cursor.op(anchor);
    if (operation == nullptr) {
        return match_result::no_match();
    }
    value_id activation    = ID_INVALID;
    value_id route_weights = ID_INVALID;
    for (value_id input : operation->inputs) {
        if (cursor.producer(input, GGML_OP_GLU) != ID_INVALID) {
            activation = input;
        }
    }
    if (activation == ID_INVALID) {
        return match_result::no_match();
    }
    const op_id weighted = cursor.unique_consumer(operation->output, GGML_OP_MUL);
    if (weighted == ID_INVALID) {
        return match_result::no_match();
    }
    const graph_op * weighted_op = cursor.op(weighted);
    if (weighted_op == nullptr || weighted_op->inputs.size() != 2) {
        return match_result::no_match();
    }
    route_weights = weighted_op->inputs[0] == operation->output ? weighted_op->inputs[1] : weighted_op->inputs[0];
    if (route_weights == operation->output) {
        return match_result::no_match();
    }
    const std::vector<op_id> views = cursor.consumers(cursor.output(weighted), GGML_OP_VIEW);
    if (views.empty()) {
        return match_result::no_match();
    }
    std::set<value_id> routed_values;
    for (op_id view : views) {
        routed_values.insert(cursor.output(view));
    }
    std::set<op_id> reductions;
    bool            changed = true;
    while (changed) {
        changed = false;
        const std::vector<value_id> values(routed_values.begin(), routed_values.end());
        for (value_id value : values) {
            for (op_id add : cursor.consumers(value, GGML_OP_ADD)) {
                if (reductions.count(add) != 0) {
                    continue;
                }
                const graph_op * add_op = cursor.op(add);
                if (add_op != nullptr && std::all_of(add_op->inputs.begin(), add_op->inputs.end(),
                                                     [&](value_id input) { return routed_values.count(input) != 0; })) {
                    reductions.insert(add);
                    routed_values.insert(add_op->output);
                    changed = true;
                }
            }
        }
    }
    std::set<op_id> residuals;
    for (value_id value : routed_values) {
        for (op_id add : cursor.consumers(value, GGML_OP_ADD)) {
            if (reductions.count(add) != 0) {
                continue;
            }
            const graph_op * add_op = cursor.op(add);
            if (add_op != nullptr && std::count_if(add_op->inputs.begin(), add_op->inputs.end(), [&](value_id input) {
                                         return routed_values.count(input) != 0;
                                     }) == 1) {
                residuals.insert(add);
            }
        }
    }
    if (reductions.size() + 1 != views.size() || residuals.size() != 1) {
        return match_result::no_match();
    }
    const op_id     residual = *residuals.begin();
    std::set<op_id> owned{ anchor, weighted, residual };
    owned.insert(views.begin(), views.end());
    owned.insert(reductions.begin(), reductions.end());
    region_match match;
    match.kind       = region_kind::REGION_KIND_EXPERT_DOWN;
    match.anchor     = anchor;
    match.operations = { owned.begin(), owned.end() };
    match.payload =
        expert_down_region{ activation, route_weights, cursor.output(residual), static_cast<uint32_t>(views.size()) };
    return match_result::matched(std::move(match));
}

inline match_result endpoint(match_cursor & cursor, op_id anchor) {
    const graph_op * projection = cursor.op(anchor);
    if (projection == nullptr || !cursor.is_root(projection->output)) {
        return match_result::no_match();
    }
    op_id prepared = ID_INVALID;
    for (value_id input : projection->inputs) {
        prepared = cursor.producer(input, GGML_OP_MUL);
        if (prepared != ID_INVALID) {
            break;
        }
    }
    const graph_op * prepared_op = cursor.op(prepared);
    if (prepared_op == nullptr) {
        return match_result::no_match();
    }
    op_id norm = ID_INVALID;
    for (value_id input : prepared_op->inputs) {
        norm = cursor.producer(input, GGML_OP_RMS_NORM);
        if (norm != ID_INVALID) {
            break;
        }
    }
    const graph_op * norm_op = cursor.op(norm);
    if (norm_op == nullptr || norm_op->inputs.empty()) {
        return match_result::no_match();
    }
    region_match match;
    match.kind       = region_kind::REGION_KIND_ENDPOINT;
    match.anchor     = anchor;
    match.operations = { norm, prepared, anchor };
    match.payload    = endpoint_region{ norm_op->inputs[0], projection->output };
    return match_result::matched(std::move(match));
}

inline std::vector<semantic_group_match> routed_blocks(const graph_plan & plan) {
    std::map<value_id, region_id>                                     qkv_by_prepared;
    std::map<std::tuple<value_id, storage_id, storage_id>, region_id> attention_by_key;
    std::map<value_id, region_id>                                     output_by_attention;
    std::map<value_id, region_id>                                     router_by_prepared;
    std::map<value_id, region_id>                                     gate_by_routes;
    std::map<value_id, region_id>                                     down_by_activation;
    std::vector<region_id>                                            prepares;
    for (const graph_region & region : plan.regions()) {
        if (std::holds_alternative<attention_prepare_region>(region.payload)) {
            prepares.push_back(region.id);
        } else if (const auto * value = std::get_if<attention_qkv_region>(&region.payload)) {
            qkv_by_prepared.emplace(value->prepared, region.id);
        } else if (const auto * value = std::get_if<attention_region>(&region.payload)) {
            attention_by_key.emplace(std::make_tuple(value->query, value->key_cache, value->value_cache), region.id);
        } else if (const auto * value = std::get_if<attention_output_region>(&region.payload)) {
            output_by_attention.emplace(value->attention_result, region.id);
        } else if (const auto * value = std::get_if<router_selection_region>(&region.payload)) {
            router_by_prepared.emplace(value->prepared, region.id);
        } else if (const auto * value = std::get_if<expert_gate_up_region>(&region.payload)) {
            gate_by_routes.emplace(value->route_ids, region.id);
        } else if (const auto * value = std::get_if<expert_down_region>(&region.payload)) {
            down_by_activation.emplace(value->activation, region.id);
        }
    }
    std::sort(prepares.begin(), prepares.end(), [&](region_id lhs, region_id rhs) {
        return plan.regions()[lhs].source_ops.front() < plan.regions()[rhs].source_ops.front();
    });
    std::vector<semantic_group_match> result;
    for (region_id prepare_id : prepares) {
        const auto & prepare      = std::get<attention_prepare_region>(plan.regions()[prepare_id].payload);
        const auto   qkv_position = qkv_by_prepared.find(prepare.prepared);
        if (qkv_position == qkv_by_prepared.end()) {
            continue;
        }
        const auto & qkv = std::get<attention_qkv_region>(plan.regions()[qkv_position->second].payload);
        const auto   attention_position =
            attention_by_key.find(std::make_tuple(qkv.query, qkv.key_cache, qkv.value_cache));
        if (attention_position == attention_by_key.end()) {
            continue;
        }
        const region_id attention_id    = attention_position->second;
        const auto &    attention       = std::get<attention_region>(plan.regions()[attention_id].payload);
        const auto      output_position = output_by_attention.find(attention.result);
        if (output_position == output_by_attention.end()) {
            continue;
        }
        const auto & output = std::get<attention_output_region>(plan.regions()[output_position->second].payload);
        const auto   router_position = router_by_prepared.find(output.prepared_ffn);
        if (router_position == router_by_prepared.end()) {
            continue;
        }
        const auto & router        = std::get<router_selection_region>(plan.regions()[router_position->second].payload);
        const auto   gate_position = gate_by_routes.find(router.route_ids);
        if (gate_position == gate_by_routes.end()) {
            continue;
        }
        const auto & gate          = std::get<expert_gate_up_region>(plan.regions()[gate_position->second].payload);
        const auto   down_position = down_by_activation.find(gate.activation);
        if (down_position == down_by_activation.end()) {
            continue;
        }
        semantic_group_match group;
        group.kind    = semantic_group_kind::SEMANTIC_GROUP_KIND_ROUTED_BLOCK;
        group.members = { prepare_id,
                          qkv_position->second,
                          attention_id,
                          output_position->second,
                          router_position->second,
                          gate_position->second,
                          down_position->second };
        result.push_back(std::move(group));
    }
    return result;
}

inline void register_patterns(pattern_registry & registry) {
    registry.add({ "llm.embedding", GGML_OP_GET_ROWS, 32, embedding });
    registry.add({ "llm.attention_prepare", GGML_OP_MUL, 64, attention_prepare });
    registry.add({ "llm.attention_qkv", GGML_OP_ROPE, 384, attention_qkv });
    registry.add({ "llm.attention", GGML_OP_FLASH_ATTN_EXT, 128, attention });
    registry.add({ "llm.attention_output", GGML_OP_MUL_MAT, 192, attention_output });
    registry.add({ "llm.router_selection", GGML_OP_ARGSORT, 192, router_selection });
    registry.add({ "llm.expert_gate_up", GGML_OP_GLU, 128, expert_gate_up });
    registry.add({ "llm.expert_down", GGML_OP_MUL_MAT_ID, 512, expert_down });
    registry.add({ "llm.endpoint", GGML_OP_MUL_MAT, 96, endpoint });
    registry.add(routed_blocks);
}

}  // namespace llm_patterns
}  // namespace ggml::hrx
