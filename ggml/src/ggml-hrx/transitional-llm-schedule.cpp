#include "transitional-llm-schedule.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace ggml::hrx {
namespace {

static constexpr size_t kAttentionKvTileSize = 64;

class RoutedTransformerProgramImplementation {
  public:
    static enum ggml_type weight_type(const Graph & graph, OperationId operation) {
        if (operation >= graph.operations.size() || graph.operations[operation].inputs.empty()) {
            return GGML_TYPE_COUNT;
        }
        const ValueId weight = graph.operations[operation].inputs[0];
        return weight < graph.values.size() ? graph.values[weight].type : GGML_TYPE_COUNT;
    }

    static std::string type_suffix(enum ggml_type type) {
        return type == GGML_TYPE_COUNT ? "invalid" : ggml_type_name(type);
    }

    static KernelSpecialization kernel(
        const std::string &                 variant,
        int                                 layer,
        size_t                              token_count,
        KernelSpecialization::ExecutionKind kind = KernelSpecialization::ExecutionKind::Native) {
        KernelSpecialization result;
        result.family                            = "qwen3_moe";
        result.variant                           = variant;
        result.kernel_id                         = kernel_catalog_id(result.family.c_str(), result.variant.c_str());
        result.execution_kind                    = kind;
        result.integer_parameters["token_count"] = token_count;
        if (layer >= 0) {
            result.integer_parameters["layer"] = layer;
        }
        return result;
    }

    static KernelSpecialization storage_kernel(const std::string &        q4_variant,
                                               const std::string &        q6_variant,
                                               const std::string &        semantic_role,
                                               enum ggml_type             type,
                                               int                        layer,
                                               size_t                     token_count,
                                               std::vector<std::string> & gaps) {
        if (type == GGML_TYPE_Q4_K && !q4_variant.empty()) {
            KernelSpecialization result              = kernel(q4_variant, layer, token_count);
            result.integer_parameters["weight_type"] = GGML_TYPE_Q4_K;
            return result;
        }
        if (type == GGML_TYPE_Q6_K && !q6_variant.empty()) {
            KernelSpecialization result              = kernel(q6_variant, layer, token_count);
            result.integer_parameters["weight_type"] = GGML_TYPE_Q6_K;
            return result;
        }
        const std::string variant = "missing_" + type_suffix(type) + '_' + semantic_role;
        gaps.push_back("block " + std::to_string(layer) + ": " + variant);
        return kernel(variant, layer, token_count, KernelSpecialization::ExecutionKind::NativeGap);
    }

    static void calculate_boundaries(const GraphIndex & index, Invocation & invocation) {
        const RegionBoundary boundary = index.boundary(invocation.covered_operations);
        for (size_t i = 0; i < boundary.inputs.size(); ++i) {
            invocation.inputs.push_back({ "arg" + std::to_string(i), boundary.inputs[i] });
        }
        for (size_t i = 0; i < boundary.outputs.size(); ++i) {
            invocation.outputs.push_back({ "result" + std::to_string(i), boundary.outputs[i] });
        }
    }

    static uint32_t append_dispatch(Invocation &                  invocation,
                                    KernelSpecialization          specialization,
                                    const std::vector<uint32_t> & dependencies,
                                    size_t &                      ordinal) {
        Dispatch dispatch;
        dispatch.kernel       = std::move(specialization);
        dispatch.dependencies = dependencies;
        invocation.dispatches.push_back(std::move(dispatch));
        return static_cast<uint32_t>(ordinal++);
    }

    static void add_dispatch(Invocation & invocation, KernelSpecialization specialization, size_t & ordinal) {
        std::vector<uint32_t> dependencies;
        if (ordinal != 0) {
            dependencies.push_back(static_cast<uint32_t>(ordinal - 1));
        }
        append_dispatch(invocation, std::move(specialization), dependencies, ordinal);
    }

    static const RoutedTransformerComponent * find_component(const RoutedTransformerModel &  model,
                                                             LogicalComponentId              id,
                                                             const RoutedTransformerBlock ** owner) {
        if (model.preamble.id == id) {
            *owner = nullptr;
            return &model.preamble;
        }
        for (const RoutedTransformerBlock & block : model.blocks) {
            for (const RoutedTransformerComponent & component : block.components) {
                if (component.id == id) {
                    *owner = &block;
                    return &component;
                }
            }
        }
        if (model.endpoint.id == id) {
            *owner = nullptr;
            return &model.endpoint;
        }
        for (const RoutedTransformerComponent & component : model.fallback_components) {
            if (component.id == id) {
                *owner = nullptr;
                return &component;
            }
        }
        *owner = nullptr;
        return nullptr;
    }

    static void emit_prefill_component(const Graph &                  graph,
                                       const RoutedTransformerModel & model,
                                       const RoutedTransformerBlock & block,
                                       RoutedTransformerComponentKind kind,
                                       Invocation &                   invocation,
                                       size_t &                       ordinal,
                                       std::vector<std::string> &     gaps) {
        const int    layer              = static_cast<int>(block.ordinal);
        const size_t token_count        = model.query_token_count;
        const bool   terminal           = block.ordinal + 1 == model.blocks.size();
        const size_t active_token_count = terminal ? model.output_token_count : token_count;
        switch (kind) {
            case RoutedTransformerComponentKind::AttentionPrepare:
                // Preserve the established specialization contract: historically
                // the next attention preparation was published by the preceding
                // block's down recipe and therefore carried that block ordinal.
                add_dispatch(invocation,
                             kernel("qwen3_moe_rmsnorm_f32", block.ordinal == 0 ? layer : layer - 1, token_count),
                             ordinal);
                break;
            case RoutedTransformerComponentKind::AttentionQkvPublication:
                {
                    std::vector<uint32_t> projection_dependencies;
                    if (ordinal != 0) {
                        projection_dependencies.push_back(static_cast<uint32_t>(ordinal - 1));
                    }
                    std::vector<uint32_t> projection_ordinals;
                    for (OperationId projection : { block.operations_by_role.attention_query_projection,
                                                    block.operations_by_role.attention_value_projection,
                                                    block.operations_by_role.attention_key_projection }) {
                        projection_ordinals.push_back(append_dispatch(
                            invocation,
                            storage_kernel("qwen3_moe_dense_linear_q4k_f16_wmma", "qwen3_moe_dense_linear_q6k_f16_wmma",
                                           "dense_attention_projection", weight_type(graph, projection), layer,
                                           token_count, gaps),
                            projection_dependencies, ordinal));
                    }
                    append_dispatch(invocation, kernel("qwen3_moe_attention_postprocess_f32_f16", layer, token_count),
                                    projection_ordinals, ordinal);
                    break;
                }
            case RoutedTransformerComponentKind::Attention:
                add_dispatch(invocation, kernel("qwen3_moe_flash_attention_f32_f16_wmma", layer, token_count), ordinal);
                break;
            case RoutedTransformerComponentKind::AttentionOutputPrepare:
                add_dispatch(invocation,
                             storage_kernel("qwen3_moe_dense_linear_q4k_f16_wmma",
                                            "qwen3_moe_dense_linear_q6k_f16_wmma", "dense_attention_output",
                                            weight_type(graph, block.operations_by_role.attention_output_projection),
                                            layer, token_count, gaps),
                             ordinal);
                if (block.operations_by_role.attention_output_selection != kInvalidId) {
                    add_dispatch(invocation, kernel("ggml_gather_add_f32", layer, active_token_count), ordinal);
                }
                add_dispatch(invocation, kernel("qwen3_moe_rmsnorm_f32", layer, active_token_count), ordinal);
                break;
            case RoutedTransformerComponentKind::RouterSelection:
                add_dispatch(invocation,
                             kernel("qwen3_moe_router_projection_f32_four_row_wave32", layer, active_token_count),
                             ordinal);
                add_dispatch(invocation, kernel("qwen3_moe_router_top8_f32", layer, active_token_count), ordinal);
                if (invocation.recipe == routed_transformer_recipes::kPrefillExpertPartition) {
                    add_dispatch(
                        invocation,
                        kernel("qwen3_moe_build_expert_table_partition_prefill_512", layer, active_token_count),
                        ordinal);
                } else {
                    add_dispatch(invocation, kernel("qwen3_moe_build_expert_table", layer, active_token_count),
                                 ordinal);
                    add_dispatch(invocation,
                                 kernel("qwen3_moe_build_expert_partition_table", layer, active_token_count), ordinal);
                }
                break;
            case RoutedTransformerComponentKind::ExpertGateUp:
                add_dispatch(invocation,
                             kernel("qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma", layer, active_token_count),
                             ordinal);
                break;
            case RoutedTransformerComponentKind::ExpertDownPublication:
                add_dispatch(invocation,
                             storage_kernel("qwen3_moe_routed_down_q4k_f16_wmma_grouped",
                                            "qwen3_moe_routed_down_q6k_f16_wmma_grouped", "routed_down",
                                            weight_type(graph, block.operations_by_role.experts_routed_down), layer,
                                            active_token_count, gaps),
                             ordinal);
                add_dispatch(invocation,
                             kernel(invocation.recipe == routed_transformer_recipes::kPrefillDownNextNorm ?
                                        "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32" :
                                        "qwen3_moe_routed_down_weighted_reduce_f16_f32",
                                    layer, active_token_count),
                             ordinal);
                break;
            default:
                break;
        }
    }

    static void emit_decode_component(const Graph &                  graph,
                                      const RoutedTransformerModel & model,
                                      const RoutedTransformerBlock & block,
                                      RoutedTransformerComponentKind kind,
                                      Invocation &                   invocation,
                                      size_t &                       ordinal,
                                      std::vector<std::string> &     gaps) {
        const int    layer       = static_cast<int>(block.ordinal);
        const size_t token_count = model.query_token_count;
        switch (kind) {
            case RoutedTransformerComponentKind::AttentionPrepare:
                // The preamble publishes block zero's Q8 input; every prior block's
                // down/publication recipe publishes the following block's input.
                break;
            case RoutedTransformerComponentKind::AttentionQkvPublication:
                {
                    const enum ggml_type query_type =
                        weight_type(graph, block.operations_by_role.attention_query_projection);
                    const enum ggml_type key_type =
                        weight_type(graph, block.operations_by_role.attention_key_projection);
                    const enum ggml_type value_type =
                        weight_type(graph, block.operations_by_role.attention_value_projection);
                    if (query_type == GGML_TYPE_Q4_K && key_type == GGML_TYPE_Q4_K &&
                        (value_type == GGML_TYPE_Q4_K || value_type == GGML_TYPE_Q6_K)) {
                        KernelSpecialization qkv = kernel("qwen3_moe_attention_qkv_quantized", layer, token_count);
                        qkv.integer_parameters["query_weight_type"] = query_type;
                        qkv.integer_parameters["key_weight_type"]   = key_type;
                        qkv.integer_parameters["value_weight_type"] = value_type;
                        add_dispatch(invocation, std::move(qkv), ordinal);
                    } else {
                        const std::string variant = "qwen3_moe_attention_qkv_postprocess_fused_decode_missing_" +
                                                    type_suffix(query_type) + '_' + type_suffix(key_type) + '_' +
                                                    type_suffix(value_type);
                        gaps.push_back("block " + std::to_string(layer) + ": " + variant);
                        add_dispatch(
                            invocation,
                            kernel(variant, layer, token_count, KernelSpecialization::ExecutionKind::NativeGap),
                            ordinal);
                    }
                    add_dispatch(invocation, kernel("qwen3_moe_attention_postprocess_f32_f16", layer, token_count),
                                 ordinal);
                    break;
                }
            case RoutedTransformerComponentKind::Attention:
                {
                    KernelSpecialization flash =
                        kernel("qwen3_moe_flash_attention_decode_split_f32_f16_wmma", layer, token_count);
                    flash.integer_parameters["key_value_token_count"] = model.key_value_token_count;
                    flash.integer_parameters["key_value_tile_size"]   = kAttentionKvTileSize;
                    flash.integer_parameters["key_value_block_count"] =
                        (model.key_value_token_count + kAttentionKvTileSize - 1) / kAttentionKvTileSize;
                    add_dispatch(invocation, std::move(flash), ordinal);
                    break;
                }
            case RoutedTransformerComponentKind::AttentionOutputPrepare:
                add_dispatch(invocation, kernel("ggml_quantize_q8_1_x4_f32", layer, token_count), ordinal);
                add_dispatch(invocation,
                             storage_kernel("qwen3_moe_dense_linear_q4k_q8_1_x4", "", "dense_attention_output_q8",
                                            weight_type(graph, block.operations_by_role.attention_output_projection),
                                            layer, token_count, gaps),
                             ordinal);
                add_dispatch(invocation, kernel("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4", layer, token_count), ordinal);
                break;
            case RoutedTransformerComponentKind::RouterSelection:
                add_dispatch(invocation, kernel("qwen3_moe_router_projection_f32_one_row_wave64", layer, token_count),
                             ordinal);
                add_dispatch(invocation, kernel("qwen3_moe_router_top8_f32", layer, token_count), ordinal);
                break;
            case RoutedTransformerComponentKind::ExpertGateUp:
                add_dispatch(invocation, kernel("qwen3_moe_routed_gate_up_swiglu_q4k_q8", layer, token_count), ordinal);
                add_dispatch(invocation, kernel("ggml_quantize_q8_1_x4_f32", layer, token_count * model.route_count),
                             ordinal);
                break;
            case RoutedTransformerComponentKind::ExpertDownPublication:
                add_dispatch(
                    invocation,
                    storage_kernel("qwen3_moe_routed_down_q4k_q8_1_x4", "qwen3_moe_routed_down_q6k_f32_wave64",
                                   "routed_down", weight_type(graph, block.operations_by_role.experts_routed_down),
                                   layer, token_count, gaps),
                    ordinal);
                add_dispatch(
                    invocation,
                    kernel(block.ordinal + 1 == model.blocks.size() ? "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4" :
                                                                      "qwen3_moe_attention_rmsnorm_quantize_q8_1_x4",
                           layer, token_count),
                    ordinal);
                break;
            default:
                break;
        }
    }

    static bool emit_decode_recipe(const Graph &                  graph,
                                   const RoutedTransformerModel & model,
                                   const RoutedTransformerBlock & block,
                                   const std::string &            recipe,
                                   Invocation &                   invocation,
                                   size_t &                       ordinal,
                                   std::vector<std::string> &     gaps) {
        const int    layer       = static_cast<int>(block.ordinal);
        const size_t token_count = model.query_token_count;
        if (recipe == routed_transformer_recipes::kDecodeQkvPostprocess) {
            const enum ggml_type query_type = weight_type(graph, block.operations_by_role.attention_query_projection);
            const enum ggml_type key_type   = weight_type(graph, block.operations_by_role.attention_key_projection);
            const enum ggml_type value_type = weight_type(graph, block.operations_by_role.attention_value_projection);
            if (query_type == GGML_TYPE_Q4_K && key_type == GGML_TYPE_Q4_K &&
                (value_type == GGML_TYPE_Q4_K || value_type == GGML_TYPE_Q6_K)) {
                KernelSpecialization qkv =
                    kernel("qwen3_moe_attention_qkv_postprocess_fused_decode", layer, token_count);
                qkv.integer_parameters["query_weight_type"] = query_type;
                qkv.integer_parameters["key_weight_type"]   = key_type;
                qkv.integer_parameters["value_weight_type"] = value_type;
                add_dispatch(invocation, std::move(qkv), ordinal);
            } else {
                const std::string variant = "qwen3_moe_attention_qkv_postprocess_fused_decode_missing_" +
                                            type_suffix(query_type) + '_' + type_suffix(key_type) + '_' +
                                            type_suffix(value_type);
                gaps.push_back("block " + std::to_string(layer) + ": " + variant);
                add_dispatch(invocation,
                             kernel(variant, layer, token_count, KernelSpecialization::ExecutionKind::NativeGap),
                             ordinal);
            }
            return true;
        }
        if (recipe == routed_transformer_recipes::kDecodeAttentionNextQ8) {
            KernelSpecialization flash =
                kernel("qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8", layer, token_count);
            flash.integer_parameters["key_value_token_count"] = model.key_value_token_count;
            flash.integer_parameters["key_value_tile_size"]   = kAttentionKvTileSize;
            flash.integer_parameters["key_value_block_count"] =
                (model.key_value_token_count + kAttentionKvTileSize - 1) / kAttentionKvTileSize;
            add_dispatch(invocation, std::move(flash), ordinal);
            add_dispatch(
                invocation,
                storage_kernel("qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8", "", "dense_attention_output_next_q8",
                               weight_type(graph, block.operations_by_role.attention_output_projection), layer,
                               token_count, gaps),
                ordinal);
            return true;
        }
        if (recipe == routed_transformer_recipes::kDecodeRouterTopK) {
            add_dispatch(invocation, kernel("qwen3_moe_router_projection_top8_fused_decode_f32", layer, token_count),
                         ordinal);
            return true;
        }
        if (recipe == routed_transformer_recipes::kDecodeGateUpNextQ8) {
            const enum ggml_type down_type = weight_type(graph, block.operations_by_role.experts_routed_down);
            if (down_type == GGML_TYPE_Q4_K) {
                add_dispatch(invocation,
                             kernel("qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8", layer, token_count),
                             ordinal);
            } else if (down_type == GGML_TYPE_Q6_K) {
                // The Q6 down path consumes F32 activations, so gate/up preserves
                // its F32 output instead of publishing an unused Q8 packet.
                add_dispatch(invocation, kernel("qwen3_moe_routed_gate_up_swiglu_q4k_q8", layer, token_count), ordinal);
            } else {
                const std::string variant = "qwen3_moe_routed_gate_up_missing_down_" + type_suffix(down_type);
                gaps.push_back("block " + std::to_string(layer) + ": " + variant);
                add_dispatch(invocation,
                             kernel(variant, layer, token_count, KernelSpecialization::ExecutionKind::NativeGap),
                             ordinal);
            }
            return true;
        }
        if (recipe == routed_transformer_recipes::kDecodeDownNextQ8) {
            add_dispatch(invocation,
                         storage_kernel("qwen3_moe_routed_down_q4k_q8_1_x4_next_q8",
                                        "qwen3_moe_routed_down_q6k_f32_wave64_next_q8", "routed_down_next_q8",
                                        weight_type(graph, block.operations_by_role.experts_routed_down), layer,
                                        token_count, gaps),
                         ordinal);
            return true;
        }
        return false;
    }
};

static RoutedTransformerRecipeCatalog implemented_recipe_catalog(const RoutedTransformerModel & model) {
    RoutedTransformerRecipeCatalog catalog;
    catalog.available = {
        routed_transformer_recipes::kDecodeQkvPostprocess, routed_transformer_recipes::kDecodeAttentionNextQ8,
        routed_transformer_recipes::kDecodeRouterTopK,     routed_transformer_recipes::kDecodeGateUpNextQ8,
        routed_transformer_recipes::kDecodeDownNextQ8,
    };
    if (model.query_token_count != 1) {
        catalog.available.insert(routed_transformer_recipes::kPrefillDownNextNorm);
        if (model.query_token_count == 512) {
            catalog.available.insert(routed_transformer_recipes::kPrefillExpertPartition);
        }
    }
    return catalog;
}

}  // namespace

RoutedTransformerProgramProof RoutedTransformerProgramProof::recover(const Graph & graph) {
    RoutedTransformerProgramProof proof;
    const GraphIndex              index(graph);
    if (!index.valid()) {
        proof.errors = index.errors();
        return proof;
    }
    auto model = std::make_shared<const RoutedTransformerModel>(RoutedTransformerModel::analyze(index));
    if (!model->valid()) {
        proof.errors = model->errors;
        return proof;
    }
    proof.logical_program         = model;
    proof.structurally_recognized = true;
    SearchOptions search_options;
    search_options.require_complete_coverage = true;
    search_options.record_trace              = true;
    proof.search                             = SearchResult::search(
        index, RoutedTransformerProvider::make_planner(implemented_recipe_catalog(*model), model), search_options);
    if (!proof.search.valid()) {
        proof.errors = proof.search.errors;
        return proof;
    }

    Schedule & schedule        = proof.schedule;
    schedule.graph_fingerprint = graph.fingerprint;
    const bool prefill         = model->query_token_count != 1;
    schedule.workload          = prefill ? "prefill-" + std::to_string(model->query_token_count) :
                                           "decode-" + std::to_string(model->key_value_token_count);
    schedule.oracle_revision   = "routed-transformer-search-v1";
    for (ValueId root : graph.roots) {
        const Value &     value = graph.values[root];
        const std::string materialization =
            value.access.shape[0] > model->hidden_size ? "full_f32_logits" : "normalized_f32_hidden_state";
        schedule.roots.push_back({ root, RootDisposition::Materialized, materialization });
    }

    size_t                   dispatch_ordinal = 0;
    std::vector<OperationId> deferred_operations;
    std::vector<uint32_t>    deferred_components;
    for (const FusionCandidate & selected : proof.search.selected) {
        if (selected.logical_components.empty()) {
            proof.errors.push_back("selected recipe " + selected.family + " has no logical component provenance");
            return proof;
        }
        const RoutedTransformerBlock *     block = nullptr;
        const RoutedTransformerComponent * component =
            RoutedTransformerProgramImplementation::find_component(*model, selected.logical_components.front(), &block);
        if (component == nullptr) {
            proof.errors.push_back("selected recipe " + selected.family + " references an unknown logical component");
            return proof;
        }

        Invocation invocation;
        invocation.recipe             = selected.family;
        invocation.logical_components = selected.logical_components;
        invocation.covered_operations = selected.operations;
        invocation.stage              = RoutedTransformerModel::component_kind_name(component->kind);
        invocation.layer              = block == nullptr ? -1 : static_cast<int32_t>(block->ordinal);
        invocation.kernel             = RoutedTransformerProgramImplementation::kernel(
            selected.family, invocation.layer,
            component->kind == RoutedTransformerComponentKind::ProgramEndpoint ? model->output_token_count :
                                                                                             model->query_token_count);

        if (component->kind == RoutedTransformerComponentKind::ProgramPreamble) {
            KernelSpecialization embedding = RoutedTransformerProgramImplementation::kernel(
                "qwen_token_embedding_q4k_bringup_workaround", -1, model->query_token_count);
            embedding.integer_parameters["vocabulary_count"] =
                graph.values[graph.operations[model->preamble_operations.front()].inputs[0]].access.shape[1];
            embedding.integer_parameters["hidden_size"] = model->hidden_size;
            RoutedTransformerProgramImplementation::add_dispatch(invocation, std::move(embedding), dispatch_ordinal);
            if (prefill) {
                KernelSpecialization context_capture;
                context_capture.family  = "qwen3_moe";
                context_capture.variant = "qwen_attention_context_base_capture";
                context_capture.kernel_id =
                    kernel_catalog_id(context_capture.family.c_str(), context_capture.variant.c_str());
                RoutedTransformerProgramImplementation::add_dispatch(invocation, std::move(context_capture),
                                                                     dispatch_ordinal);
                KernelSpecialization metadata = RoutedTransformerProgramImplementation::kernel(
                    "qwen_attention_metadata_bringup_workaround", -1, model->query_token_count);
                metadata.integer_parameters["context_capacity"] = model->key_value_token_count;
                RoutedTransformerProgramImplementation::add_dispatch(invocation, std::move(metadata), dispatch_ordinal);
            } else {
                KernelSpecialization state;
                state.family    = "qwen3_moe";
                state.variant   = "qwen_attention_decode_state_initialize";
                state.kernel_id = kernel_catalog_id(state.family.c_str(), state.variant.c_str());
                RoutedTransformerProgramImplementation::add_dispatch(invocation, std::move(state), dispatch_ordinal);
                RoutedTransformerProgramImplementation::add_dispatch(
                    invocation,
                    RoutedTransformerProgramImplementation::kernel("qwen3_moe_attention_rmsnorm_quantize_q8_1_x4", -1,
                                                                   model->query_token_count),
                    dispatch_ordinal);
            }
        } else if (component->kind == RoutedTransformerComponentKind::ProgramEndpoint) {
            if (prefill) {
                RoutedTransformerProgramImplementation::add_dispatch(
                    invocation,
                    RoutedTransformerProgramImplementation::kernel("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4", -1,
                                                                   model->output_token_count),
                    dispatch_ordinal);
            }
            RoutedTransformerProgramImplementation::add_dispatch(
                invocation,
                RoutedTransformerProgramImplementation::kernel("ggml_linear_q6k_q8_1_x4", -1,
                                                               model->output_token_count),
                dispatch_ordinal);
        } else if (component->kind == RoutedTransformerComponentKind::Atom) {
            KernelSpecialization atom;
            atom.family         = "hrx_atom";
            atom.variant        = ggml_op_name(graph.operations[component->hero].op);
            atom.kernel_id      = kernel_catalog_id(atom.family.c_str(), atom.variant.c_str());
            atom.execution_kind = KernelSpecialization::ExecutionKind::NativeEager;
            invocation.kernel   = atom;
            RoutedTransformerProgramImplementation::add_dispatch(invocation, std::move(atom), dispatch_ordinal);
        } else if (block != nullptr &&
                   (selected.correctness_baseline ||
                    (prefill && (selected.family == routed_transformer_recipes::kPrefillExpertPartition ||
                                 selected.family == routed_transformer_recipes::kPrefillDownNextNorm)))) {
            if (prefill) {
                RoutedTransformerProgramImplementation::emit_prefill_component(
                    graph, *model, *block, component->kind, invocation, dispatch_ordinal, proof.native_gaps);
            } else {
                RoutedTransformerProgramImplementation::emit_decode_component(
                    graph, *model, *block, component->kind, invocation, dispatch_ordinal, proof.native_gaps);
            }
        } else if (block != nullptr && !prefill &&
                   RoutedTransformerProgramImplementation::emit_decode_recipe(
                       graph, *model, *block, selected.family, invocation, dispatch_ordinal, proof.native_gaps)) {
            // The selected fused recipe populated the invocation.
        } else {
            proof.errors.push_back("selected recipe " + selected.family + " has no registered emitter");
            return proof;
        }

        if (invocation.dispatches.empty()) {
            deferred_operations.insert(deferred_operations.end(), invocation.covered_operations.begin(),
                                       invocation.covered_operations.end());
            deferred_components.insert(deferred_components.end(), invocation.logical_components.begin(),
                                       invocation.logical_components.end());
            continue;
        }
        invocation.covered_operations.insert(invocation.covered_operations.end(), deferred_operations.begin(),
                                             deferred_operations.end());
        invocation.logical_components.insert(invocation.logical_components.end(), deferred_components.begin(),
                                             deferred_components.end());
        deferred_operations.clear();
        deferred_components.clear();
        std::sort(invocation.covered_operations.begin(), invocation.covered_operations.end());
        invocation.covered_operations.erase(
            std::unique(invocation.covered_operations.begin(), invocation.covered_operations.end()),
            invocation.covered_operations.end());
        std::sort(invocation.logical_components.begin(), invocation.logical_components.end());
        invocation.logical_components.erase(
            std::unique(invocation.logical_components.begin(), invocation.logical_components.end()),
            invocation.logical_components.end());
        RoutedTransformerProgramImplementation::calculate_boundaries(index, invocation);
        schedule.invocations.push_back(std::move(invocation));
    }
    if (!deferred_operations.empty()) {
        proof.errors.push_back("logical program ends in a component with no executable recipe");
        return proof;
    }
    // The count is a derived schedule contract, not a model identity. Recipe
    // additions are expected to change it while preserving full graph coverage.
    schedule.expected_dispatch_count = schedule_dispatch_count(schedule);
    return proof;
}

}  // namespace ggml::hrx
