#include "transitional-llm-program.h"
#include "transitional-schedule.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace ggml::hrx {
namespace {

op_id unique_kind(const graph_plan &         plan,
                  const graph_region &       region,
                  enum ggml_op               kind,
                  std::vector<std::string> & errors,
                  const char *               role) {
    op_id result = ID_INVALID;
    for (op_id operation : region.source_ops) {
        if (plan.operations()[operation].op != kind) {
            continue;
        }
        if (result != ID_INVALID) {
            errors.push_back(std::string("region has multiple operations for role ") + role);
            return ID_INVALID;
        }
        result = operation;
    }
    if (result == ID_INVALID) {
        errors.push_back(std::string("region is missing operation for role ") + role);
    }
    return result;
}

op_id output_operation(const graph_plan &         plan,
                       const graph_region &       region,
                       value_id                   output,
                       std::vector<std::string> & errors,
                       const char *               role) {
    for (op_id operation : region.source_ops) {
        if (plan.operations()[operation].output == output) {
            return operation;
        }
    }
    errors.push_back(std::string("region is missing output operation for role ") + role);
    return ID_INVALID;
}

RoutedTransformerComponent make_component(const GraphIndex &             index,
                                          LogicalComponentId             id,
                                          RoutedTransformerComponentKind kind,
                                          op_id                          hero,
                                          const graph_region &           region) {
    RoutedTransformerComponent result;
    result.id         = id;
    result.kind       = kind;
    result.hero       = hero;
    result.operations = region.source_ops;
    result.boundary   = index.boundary(result.operations);
    return result;
}

template <typename Payload>
std::vector<std::pair<const graph_region *, const Payload *>> regions_of(const graph_plan & plan) {
    std::vector<std::pair<const graph_region *, const Payload *>> result;
    for (const graph_region & region : plan.regions()) {
        if (const auto * payload = std::get_if<Payload>(&region.payload)) {
            result.emplace_back(&region, payload);
        }
    }
    std::sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.first->source_ops.front() < rhs.first->source_ops.front();
    });
    return result;
}

}  // namespace

RoutedTransformerModel RoutedTransformerModel::from_plan(const graph_plan & plan, const GraphIndex & index) {
    RoutedTransformerModel model;
    model.graph_fingerprint = index.graph().fingerprint;
    if (!plan.valid() || plan.phase() != plan_phase::PLAN_PHASE_PLANNED || !index.valid()) {
        model.errors.push_back("transitional LLM lowering requires a valid planned graph");
        return model;
    }

    const auto   embeddings  = regions_of<embedding_region>(plan);
    const auto   prepares    = regions_of<attention_prepare_region>(plan);
    const auto   qkvs        = regions_of<attention_qkv_region>(plan);
    const auto   attentions  = regions_of<attention_region>(plan);
    const auto   outputs     = regions_of<attention_output_region>(plan);
    const auto   routers     = regions_of<router_selection_region>(plan);
    const auto   gates       = regions_of<expert_gate_up_region>(plan);
    const auto   downs       = regions_of<expert_down_region>(plan);
    const auto   endpoints   = regions_of<endpoint_region>(plan);
    const size_t block_count = prepares.size();
    if (embeddings.size() != 1 || endpoints.size() != 1 || block_count == 0 || qkvs.size() != block_count ||
        attentions.size() != block_count || outputs.size() != block_count || routers.size() != block_count ||
        gates.size() != block_count || downs.size() != block_count) {
        model.errors.push_back(
            "planned LLM regions do not form one preamble, a complete block chain, and one endpoint");
        return model;
    }

    std::map<value_id, std::pair<const graph_region *, const attention_qkv_region *>> qkv_by_prepared;
    std::map<std::tuple<value_id, storage_id, storage_id>, std::pair<const graph_region *, const attention_region *>>
                                                                                         attention_by_key;
    std::map<value_id, std::pair<const graph_region *, const attention_output_region *>> output_by_attention;
    std::map<value_id, std::pair<const graph_region *, const router_selection_region *>> router_by_prepared;
    std::map<value_id, std::pair<const graph_region *, const expert_gate_up_region *>>   gate_by_routes;
    std::map<value_id, std::pair<const graph_region *, const expert_down_region *>>      down_by_activation;
    for (const auto & item : qkvs) {
        qkv_by_prepared.emplace(item.second->prepared, item);
    }
    for (const auto & item : attentions) {
        attention_by_key.emplace(std::make_tuple(item.second->query, item.second->key_cache, item.second->value_cache),
                                 item);
    }
    for (const auto & item : outputs) {
        output_by_attention.emplace(item.second->attention_result, item);
    }
    for (const auto & item : routers) {
        router_by_prepared.emplace(item.second->prepared, item);
    }
    for (const auto & item : gates) {
        gate_by_routes.emplace(item.second->route_ids, item);
    }
    for (const auto & item : downs) {
        down_by_activation.emplace(item.second->activation, item);
    }

    const graph_region &     embedding_source  = *embeddings.front().first;
    const embedding_region & embedding         = *embeddings.front().second;
    model.preamble_operations                  = embedding_source.source_ops;
    model.operations_by_role.program_embedding = embedding_source.source_ops.front();
    model.values_by_role.program_hidden_state  = embedding.hidden;

    LogicalComponentId component_id = 0;
    model.preamble = make_component(index, component_id++, RoutedTransformerComponentKind::ProgramPreamble,
                                    model.operations_by_role.program_embedding, embedding_source);
    for (size_t ordinal = 0; ordinal < prepares.size(); ++ordinal) {
        const graph_region &             prepare_source = *prepares[ordinal].first;
        const attention_prepare_region & prepare        = *prepares[ordinal].second;
        const auto                       qkv_position   = qkv_by_prepared.find(prepare.prepared);
        if (qkv_position == qkv_by_prepared.end()) {
            model.errors.push_back("attention preparation has no QKV publication consumer");
            continue;
        }
        const graph_region &         qkv_source = *qkv_position->second.first;
        const attention_qkv_region & qkv        = *qkv_position->second.second;

        const auto attention_position =
            attention_by_key.find(std::make_tuple(qkv.query, qkv.key_cache, qkv.value_cache));
        if (attention_position == attention_by_key.end()) {
            model.errors.push_back("QKV publication has no attention consumer with matching cache identity");
            continue;
        }
        const graph_region *     attention_source = attention_position->second.first;
        const attention_region * attention        = attention_position->second.second;
        const auto               output_position  = output_by_attention.find(attention->result);
        if (output_position == output_by_attention.end()) {
            model.errors.push_back("attention region has no output preparation consumer");
            continue;
        }
        const graph_region &            output_source   = *output_position->second.first;
        const attention_output_region & output          = *output_position->second.second;
        const auto                      router_position = router_by_prepared.find(output.prepared_ffn);
        if (router_position == router_by_prepared.end()) {
            model.errors.push_back("attention output has no router consumer");
            continue;
        }
        const graph_region &            router_source = *router_position->second.first;
        const router_selection_region & router        = *router_position->second.second;
        const auto                      gate_position = gate_by_routes.find(router.route_ids);
        if (gate_position == gate_by_routes.end()) {
            model.errors.push_back("router selection has no gate/up consumer");
            continue;
        }
        const graph_region &          gate_source   = *gate_position->second.first;
        const expert_gate_up_region & gate          = *gate_position->second.second;
        const auto                    down_position = down_by_activation.find(gate.activation);
        if (down_position == down_by_activation.end()) {
            model.errors.push_back("expert gate/up has no routed-down consumer");
            continue;
        }
        const graph_region &       down_source = *down_position->second.first;
        const expert_down_region & down        = *down_position->second.second;

        RoutedTransformerBlock block;
        block.ordinal                           = ordinal;
        block.values_by_role.attention_prepared = prepare.prepared;
        block.values_by_role.router_route_ids   = router.route_ids;
        block.values_by_role.experts_activation = gate.activation;
        block.operations_by_role.attention_norm =
            unique_kind(plan, prepare_source, GGML_OP_RMS_NORM, model.errors, "attention_norm");
        block.operations_by_role.attention_prepared =
            output_operation(plan, prepare_source, prepare.prepared, model.errors, "attention_prepared");
        block.operations_by_role.attention_query_projection = qkv.query_projection;
        block.operations_by_role.attention_key_projection   = qkv.key_projection;
        block.operations_by_role.attention_value_projection = qkv.value_projection;
        block.operations_by_role.attention_query_rope =
            output_operation(plan, qkv_source, qkv.query, model.errors, "attention_query_rope");
        for (op_id operation : qkv_source.source_ops) {
            const graph_op & op = plan.operations()[operation];
            if (op.op != GGML_OP_SET_ROWS) {
                continue;
            }
            const storage_id storage = plan.values()[op.output].access.storage;
            if (storage == qkv.key_cache) {
                block.operations_by_role.attention_key_cache_writer = operation;
            }
            if (storage == qkv.value_cache) {
                block.operations_by_role.attention_value_cache_writer = operation;
            }
        }
        block.operations_by_role.attention_flash = attention->flash;
        block.operations_by_role.attention_result_reshape =
            output_operation(plan, *attention_source, attention->result, model.errors, "attention_result_reshape");
        block.operations_by_role.attention_output_projection =
            unique_kind(plan, output_source, GGML_OP_MUL_MAT, model.errors, "attention_output_projection");
        block.operations_by_role.feed_forward_prepared =
            output_operation(plan, output_source, output.prepared_ffn, model.errors, "feed_forward_prepared");
        for (op_id operation : output_source.source_ops) {
            const graph_op & op = plan.operations()[operation];
            if (op.op == GGML_OP_ADD) {
                block.operations_by_role.attention_residual = operation;
            }
            if (op.op != GGML_OP_GET_ROWS || op.inputs.empty()) {
                continue;
            }
            if (op.inputs[0] == plan.operations()[block.operations_by_role.attention_output_projection].output) {
                block.operations_by_role.attention_output_selection = operation;
            } else {
                block.operations_by_role.hidden_state_selection = operation;
            }
        }
        block.operations_by_role.router_projection =
            unique_kind(plan, router_source, GGML_OP_MUL_MAT, model.errors, "router_projection");
        block.operations_by_role.router_route_ids =
            output_operation(plan, router_source, router.route_ids, model.errors, "router_route_ids");
        block.operations_by_role.router_route_weights =
            output_operation(plan, router_source, router.route_weights, model.errors, "router_route_weights");
        block.operations_by_role.experts_gate_up =
            unique_kind(plan, gate_source, GGML_OP_GLU, model.errors, "experts_gate_up");
        const graph_op & gate_up_op = plan.operations()[block.operations_by_role.experts_gate_up];
        if (gate_up_op.inputs.size() >= 2) {
            block.operations_by_role.experts_gate_projection = plan.values()[gate_up_op.inputs[0]].producer;
            block.operations_by_role.experts_up_projection   = plan.values()[gate_up_op.inputs[1]].producer;
        }
        block.operations_by_role.experts_routed_down =
            unique_kind(plan, down_source, GGML_OP_MUL_MAT_ID, model.errors, "experts_routed_down");
        block.operations_by_role.hidden_output =
            output_operation(plan, down_source, down.hidden_output, model.errors, "hidden_output");

        block.components.push_back(make_component(index, component_id++,
                                                  RoutedTransformerComponentKind::AttentionPrepare,
                                                  block.operations_by_role.attention_prepared, prepare_source));
        block.components.push_back(make_component(index, component_id++,
                                                  RoutedTransformerComponentKind::AttentionQkvPublication,
                                                  block.operations_by_role.attention_query_projection, qkv_source));
        block.components.push_back(make_component(index, component_id++, RoutedTransformerComponentKind::Attention,
                                                  block.operations_by_role.attention_flash, *attention_source));
        block.components.push_back(make_component(index, component_id++,
                                                  RoutedTransformerComponentKind::AttentionOutputPrepare,
                                                  block.operations_by_role.attention_output_projection, output_source));
        block.components.push_back(make_component(index, component_id++,
                                                  RoutedTransformerComponentKind::RouterSelection,
                                                  block.operations_by_role.router_projection, router_source));
        block.components.push_back(make_component(index, component_id++, RoutedTransformerComponentKind::ExpertGateUp,
                                                  block.operations_by_role.experts_gate_up, gate_source));
        block.components.push_back(make_component(index, component_id++,
                                                  RoutedTransformerComponentKind::ExpertDownPublication,
                                                  block.operations_by_role.experts_routed_down, down_source));
        std::set<op_id> block_operations;
        for (const RoutedTransformerComponent & component : block.components) {
            block_operations.insert(component.operations.begin(), component.operations.end());
        }
        block.operations.assign(block_operations.begin(), block_operations.end());
        model.blocks.push_back(std::move(block));
    }

    const graph_region &    endpoint_source = *endpoints.front().first;
    const endpoint_region & endpoint        = *endpoints.front().second;
    model.endpoint_operations               = endpoint_source.source_ops;
    model.operations_by_role.endpoint_norm =
        unique_kind(plan, endpoint_source, GGML_OP_RMS_NORM, model.errors, "endpoint_norm");
    model.operations_by_role.endpoint_prepared =
        unique_kind(plan, endpoint_source, GGML_OP_MUL, model.errors, "endpoint_prepared");
    model.operations_by_role.endpoint_projection =
        output_operation(plan, endpoint_source, endpoint.logits, model.errors, "endpoint_projection");
    model.endpoint              = make_component(index, component_id++, RoutedTransformerComponentKind::ProgramEndpoint,
                                                 model.operations_by_role.endpoint_projection, endpoint_source);
    model.query_token_count     = plan.facts().at("query_token_count");
    model.output_token_count    = plan.facts().at("output_token_count");
    model.key_value_token_count = plan.facts().at("key_value_token_count");
    model.hidden_size           = plan.facts().at("hidden_size");
    model.query_size            = plan.facts().at("query_size");
    model.key_value_size        = plan.facts().at("key_value_size");
    model.expert_count          = plan.facts().at("expert_count");
    model.route_count           = plan.facts().at("route_count");

    if (model.blocks.size() != block_count) {
        model.errors.push_back("not every planned LLM block was associated");
    }
    const VerificationResult verification = verify(index, model);
    model.errors.insert(model.errors.end(), verification.errors.begin(), verification.errors.end());
    return model;
}

}  // namespace ggml::hrx
