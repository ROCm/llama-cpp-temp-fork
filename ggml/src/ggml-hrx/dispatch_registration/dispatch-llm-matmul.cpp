#include "dispatch-llm-matmul.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kLlmDenseLinearQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ4KQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_q8_1_x4");
static constexpr KernelCatalogRef kLlmQuantizeQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
static constexpr KernelCatalogRef kLlmQuantizeActU4AsymKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_quant_act_u4asym");
static constexpr KernelCatalogRef kLlmQuantizeActI4Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_quant_act_i4");
static constexpr KernelCatalogRef kLlmDenseLinearQ4KU4AsymKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x128x64");
static constexpr KernelCatalogRef kLlmDenseLinearQ4KU4AsymLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x16x64");
static constexpr KernelCatalogRef kLlmDenseLinearQ4KU4AsymLowRowSplitK2Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x16x64_splitk2");
static constexpr KernelCatalogRef kLlmDenseLinearSymmetricI4AdjacentLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_symi4_i4_adjacent_m16n16_wg64");
static constexpr KernelCatalogRef kLlmDenseLinearSymmetricI4AdjacentLowRowSplitK2Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_symi4_i4_adjacent_m16n16_wg64_splitk2");
static constexpr KernelCatalogRef kLlmDenseLinearQ5KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q5k_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ5KF16WmmaLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q5k_f16_wmma_lowrow");
static constexpr KernelCatalogRef kLlmDenseLinearQ5KF16InputWmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q5k_f16_input_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ5KQ8_1X4WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q5k_q8_1_x4_wmmai8");
static constexpr KernelCatalogRef kLlmDenseLinearQ5KQ8_1X4WmmaToken256Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q5k_q8_1_x4_wmmai8_token256");
static constexpr KernelCatalogRef kLlmDenseLinearIQ4XSQ8_1X4WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_iq4xs_q8_1_x4_wmmai8");
static constexpr KernelCatalogRef kLlmDenseLinearIQ4XSQ8_1X4WmmaToken256Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_iq4xs_q8_1_x4_wmmai8_token256");
static constexpr KernelCatalogRef kLlmDenseLinearIQ4XSF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_iq4xs_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearIQ4XSF16WmmaLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_iq4xs_f16_wmma_lowrow");
static constexpr KernelCatalogRef kLlmDenseLinearIQ4XSF16InputWmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_iq4xs_f16_input_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KI8PrepackedF16WmmaLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_i8_prepacked_f16_wmma_64x16");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KPackedRawF16WmmaLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_packed_raw_f16_wmma_token1_64x16");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KPackedRawF16WmmaScaleRowOneWaveKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_packed_raw_f16_wmma_scalerow_16x16");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KF16WmmaPpWave32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_wmma_pp_wave32");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KF16InputWmmaPpWave32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_input_wmma_pp_wave32");
static constexpr KernelCatalogRef kLlmLinearQ6KQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_linear_q6k_q8_1_x4");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

enum class LlmDenseMatmulRoute : uint8_t {
    Q4K,
    Q5K,
    Q6K,
    IQ4XS,
};

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

struct LlmDenseMatmulMatch {
    const Value *    input       = nullptr;
    const Value *    weight      = nullptr;
    const Value *    output      = nullptr;
    KernelCatalogRef kernel      = {};
    ValueId          input_value = {};
    size_t           input_bytes = 0;
    int64_t          input_size  = 0;
    int64_t          output_size = 0;
    int64_t          token_count = 0;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

static LlmDenseMatmulMatch match_llm_dense_matmul(const Graph &       graph,
                                                  const GraphNode *   node,
                                                  const CommandPlan & plan,
                                                  LlmDenseMatmulRoute route) {
    LlmDenseMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !is_llm_supported_query_length(kActiveLlmMoeDispatchProfile, token_count) ||
        !is_supported_dense_input_size(input_size) || !is_supported_dense_output_size(output_size)) {
        return {};
    }

    if (route == LlmDenseMatmulRoute::Q4K && weight->type == GGML_TYPE_Q4_K) {
        match.kernel = kLlmDenseLinearQ4KF16WmmaKernel;
    } else if (route == LlmDenseMatmulRoute::Q5K && weight->type == GGML_TYPE_Q5_K) {
        match.kernel = token_count <= 16 && output_size % 64 == 0 ? kLlmDenseLinearQ5KF16WmmaLowRowKernel :
                                                                    kLlmDenseLinearQ5KF16WmmaKernel;
    } else if (route == LlmDenseMatmulRoute::Q6K && weight->type == GGML_TYPE_Q6_K) {
        match.kernel = token_count >= 128 ? kLlmDenseLinearQ6KF16WmmaPpWave32Kernel : kLlmDenseLinearQ6KF16WmmaKernel;
    } else if (route == LlmDenseMatmulRoute::IQ4XS && weight->type == GGML_TYPE_IQ4_XS) {
        match.kernel = token_count <= 16 && output_size % 64 == 0 ? kLlmDenseLinearIQ4XSF16WmmaLowRowKernel :
                                                                    kLlmDenseLinearIQ4XSF16WmmaKernel;
    } else {
        return {};
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_value = input->id;
    match.input_bytes = input->byte_count;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;

    if ((route == LlmDenseMatmulRoute::Q5K || route == LlmDenseMatmulRoute::IQ4XS) && token_count >= 128 &&
        token_count % 128 == 0 && output_size % 64 == 0) {
        const size_t q8_input_bytes = static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
        const CommandPlanAlternateValue * alternate =
            find_alternate_value(graph, plan, input->id, GGML_TYPE_Q8_1, q8_input_bytes);
        if (alternate != nullptr) {
            match.input_value = alternate->alternate_value;
            match.input_bytes = alternate->byte_count;
            if (route == LlmDenseMatmulRoute::Q5K) {
                match.kernel = token_count % 256 == 0 ? kLlmDenseLinearQ5KQ8_1X4WmmaToken256Kernel :
                                                        kLlmDenseLinearQ5KQ8_1X4WmmaKernel;
            } else {
                match.kernel = token_count % 256 == 0 ? kLlmDenseLinearIQ4XSQ8_1X4WmmaToken256Kernel :
                                                        kLlmDenseLinearIQ4XSQ8_1X4WmmaKernel;
            }
            return match;
        }
    }

    if (token_count >= 128 && (route == LlmDenseMatmulRoute::Q5K || route == LlmDenseMatmulRoute::Q6K ||
                               route == LlmDenseMatmulRoute::IQ4XS)) {
        const size_t f16_input_bytes = static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_F16, input_size);
        const CommandPlanAlternateValue * alternate =
            find_alternate_value(graph, plan, input->id, GGML_TYPE_F16, f16_input_bytes);
        if (alternate != nullptr) {
            match.input_value = alternate->alternate_value;
            match.input_bytes = alternate->byte_count;
            if (route == LlmDenseMatmulRoute::Q5K) {
                match.kernel = kLlmDenseLinearQ5KF16InputWmmaKernel;
            } else if (route == LlmDenseMatmulRoute::Q6K) {
                match.kernel = kLlmDenseLinearQ6KF16InputWmmaPpWave32Kernel;
            } else {
                match.kernel = kLlmDenseLinearIQ4XSF16InputWmmaKernel;
            }
        }
    }
    return match;
}

}  // namespace

static void build_llm_dense_matmul_dispatch(const LlmDenseMatmulMatch & match,
                                            DispatchMatch &             dispatch_match,
                                            size_t                      root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static void build_llm_dense_q4k_u4asym_dispatch(const LlmDenseMatmulMatch &  match,
                                                DispatchMatch &              dispatch_match,
                                                const DispatchMatchContext & context,
                                                KernelCatalogRef             contraction_kernel) {
    const size_t   element_count = static_cast<size_t>(match.input_size) * static_cast<size_t>(match.token_count);
    const size_t   qact_bytes    = element_count / 2;
    const size_t   scale_bytes   = element_count / 8;
    const size_t   meta_bytes    = element_count / 4;
    const bool     split_k       = contraction_kernel.id == kLlmDenseLinearQ4KU4AsymLowRowSplitK2Kernel.id;
    const size_t   partial_bytes = match.output->byte_count;
    const uint32_t completion_counter_count =
        static_cast<uint32_t>(((match.output_size + 15) / 16) * ((match.token_count + 15) / 16));
    const ValueId qact(context.next_plan_value.value);
    const ValueId scales(context.next_plan_value.value + 1);
    const ValueId metadata(context.next_plan_value.value + 2);
    const ValueId partial             = split_k ? ValueId(context.next_plan_value.value + 3) : ValueId{};
    const ValueId completion_counters = split_k ? ValueId(context.next_plan_value.value + 4) : ValueId{};

    dispatch_match.transients.push_back({ qact, "llm.dense_q4.u4_qs", qact_bytes, 256 });
    dispatch_match.transients.push_back({ scales, "llm.dense_q4.u4_ds", scale_bytes, 256 });
    dispatch_match.transients.push_back({ metadata, "llm.dense_q4.u4_meta", meta_bytes, 256 });
    if (split_k) {
        dispatch_match.transients.push_back({ partial, "llm.dense_q4.split_k_partial", partial_bytes, 256 });
        dispatch_match.completion_counter_requests.push_back({
            completion_counters,
            "llm.dense_q4.split_k_completion_counters",
            completion_counter_count,
        });
    }

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kLlmQuantizeActU4AsymKernel);
    quantize.kernel.compile_parameters.emplace("qwen3.qact_u4asym.shape_k", to_config_value(match.input_size));
    quantize.kernel.compile_parameters.emplace("qwen3.qact_u4asym.shape_cols", to_config_value(match.token_count));
    quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize.bindings.push_back({ qact, 0, qact_bytes });
    quantize.bindings.push_back({ scales, 0, scale_bytes });
    quantize.bindings.push_back({ metadata, 0, meta_bytes });

    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(contraction_kernel);
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_u4asym.shape_k", to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_u4asym.shape_rows",
                                                  to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_u4asym.shape_cols",
                                                  to_config_value(match.token_count));
    contraction.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, kQ4KI4K32Row64Layout,
                                     GGML_TYPE_Q4_K, match.input_size, match.output_size, match.weight->byte_count });
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ qact, 0, qact_bytes });
    contraction.bindings.push_back({ scales, 0, scale_bytes });
    contraction.bindings.push_back({ metadata, 0, meta_bytes });
    if (split_k) {
        contraction.bindings.push_back({ partial, 0, partial_bytes });
        contraction.bindings.push_back({ completion_counters, 0, completion_counter_count * sizeof(int32_t) });
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(contraction));
}

static void build_llm_dense_symmetric_i4_dispatch(const LlmDenseMatmulMatch &  match,
                                                  DispatchMatch &              dispatch_match,
                                                  const DispatchMatchContext & context) {
    const LlmSymmetricI4ActivationLayout activation_layout =
        llm_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const size_t logical_output_size = static_cast<size_t>(match.output_size);
    const size_t row_group_size      = ((logical_output_size + 255) / 256) * 32;
    const size_t padded_output_size  = (logical_output_size + row_group_size - 1) / row_group_size * row_group_size;
    const size_t materialized_weight_bytes = padded_output_size * static_cast<size_t>(match.input_size / 256) * 132;
    // Split large contractions only when output-row parallelism is limited.
    const bool   split_k = match.input_size % 128 == 0 && materialized_weight_bytes >= size_t{ 16 } * 1024 * 1024 &&
                         match.output_size <= 2 * match.input_size;
    const uint32_t completion_counter_count =
        static_cast<uint32_t>(((match.output_size + 15) / 16) * ((match.token_count + 15) / 16));
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kLlmSymmetricI4ActivationAlternateName) {
        alternate = nullptr;
    }
    const ValueId activation          = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    const int32_t split_value_base    = context.next_plan_value.value + (alternate == nullptr ? 1 : 0);
    const ValueId partial             = split_k ? ValueId(split_value_base) : ValueId{};
    const ValueId completion_counters = split_k ? ValueId(split_value_base + 1) : ValueId{};

    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kLlmSymmetricI4ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kLlmQuantizeActI4Kernel);
        quantize.kernel.compile_parameters.emplace("qwen3.qact_i4.shape_k", to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("qwen3.qact_i4.shape_cols", to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));
    }

    if (split_k) {
        dispatch_match.transients.push_back(
            { partial, "llm.dense_symi4.split_k_partial", match.output->byte_count, 256 });
        dispatch_match.completion_counter_requests.push_back({
            completion_counters,
            "llm.dense_symi4.split_k_completion_counters",
            completion_counter_count,
        });
    }
    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(split_k ? kLlmDenseLinearSymmetricI4AdjacentLowRowSplitK2Kernel :
                                                              kLlmDenseLinearSymmetricI4AdjacentLowRowKernel);
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_i4.lowrow.shape_k",
                                                  to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_i4.lowrow.shape_rows",
                                                  to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_i4.lowrow.shape_cols",
                                                  to_config_value(match.token_count));
    contraction.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes,
                                     kSymmetricI4K32EightGroupsShared4Layout, match.weight->type, match.input_size,
                                     match.output_size, match.weight->byte_count });
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    contraction.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    contraction.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
    if (split_k) {
        contraction.bindings.push_back({ partial, 0, match.output->byte_count });
        contraction.bindings.push_back({ completion_counters, 0, completion_counter_count * sizeof(int32_t) });
    }
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(contraction));
}

static void build_llm_dense_q4k_q8_1_x4_dispatch(const LlmDenseMatmulMatch &       match,
                                                 DispatchMatch &                   dispatch_match,
                                                 const DispatchMatchContext &      context,
                                                 const CommandPlanAlternateValue * q8_alternate) {
    const size_t  q8_bytes = static_cast<size_t>(match.token_count) * ggml_row_size(GGML_TYPE_Q8_1, match.input_size);
    const int64_t q8_group_count = match.token_count * match.input_size / 128;
    const ValueId q8_input =
        q8_alternate != nullptr ? q8_alternate->alternate_value : ValueId(context.next_plan_value.value);

    if (q8_alternate == nullptr) {
        dispatch_match.transients.push_back({ q8_input, "llm.dense_q4.q8_input", q8_bytes, 256 });
        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kLlmQuantizeQ8_1X4Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", match.token_count);
        quantize.kernel.integer_parameters.emplace("input_size", match.input_size);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                   to_config_value(q8_group_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ q8_input, 0, q8_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));
    }

    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kLlmDenseLinearQ4KQ8_1X4Kernel);
    contraction.kernel.integer_parameters.emplace("token_count", match.token_count);
    contraction.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                                  to_config_value(match.token_count));
    contraction.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                                  to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                                  to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    contraction.bindings.push_back({ q8_input, 0, q8_bytes });
    contraction.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(contraction));
}

static void build_llm_dense_q6k_q8_1_x4_dispatch(const LlmDenseMatmulMatch &       match,
                                                 DispatchMatch &                   dispatch_match,
                                                 const DispatchMatchContext &      context,
                                                 const CommandPlanAlternateValue * q8_alternate) {
    const size_t  q8_bytes = static_cast<size_t>(match.token_count) * ggml_row_size(GGML_TYPE_Q8_1, match.input_size);
    const int64_t q8_group_count = match.token_count * match.input_size / 128;
    const ValueId q8_input =
        q8_alternate != nullptr ? q8_alternate->alternate_value : ValueId(context.next_plan_value.value);

    if (q8_alternate == nullptr) {
        dispatch_match.transients.push_back({ q8_input, "llm.dense_q6.q8_input", q8_bytes, 256 });
        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kLlmQuantizeQ8_1X4Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", match.token_count);
        quantize.kernel.integer_parameters.emplace("input_size", match.input_size);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                   to_config_value(q8_group_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ q8_input, 0, q8_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));
    }

    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kLlmLinearQ6KQ8_1X4Kernel);
    contraction.kernel.integer_parameters.emplace("token_count", match.token_count);
    contraction.kernel.integer_parameters.emplace("input_size", match.input_size);
    contraction.kernel.integer_parameters.emplace("output_size", match.output_size);
    contraction.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.token_capacity",
                                                  to_config_value(match.token_count));
    contraction.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.output_capacity",
                                                  to_config_value(match.output_size));
    contraction.bindings.push_back({ q8_input, 0, q8_bytes });
    contraction.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(contraction));
}

static void build_llm_dense_q6k_i8_prepacked_dispatch(const LlmDenseMatmulMatch & match,
                                                      DispatchMatch &             dispatch_match,
                                                      size_t                      root_index) {
    const size_t block_count               = static_cast<size_t>(match.input_size / ggml_blck_size(GGML_TYPE_Q6_K));
    const size_t materialized_weight_bytes = static_cast<size_t>(match.output_size) * block_count * size_t{ 274 };

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kLlmDenseLinearQ6KI8PrepackedF16WmmaLowRowKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kQ6KI8K32Row64Layout, GGML_TYPE_Q6_K,
                                  match.input_size, match.output_size, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static void build_llm_dense_field16_dispatch(const LlmDenseMatmulMatch & match,
                                             DispatchMatch &             dispatch_match,
                                             size_t                      root_index,
                                             KernelCatalogRef            kernel,
                                             const char *                layout,
                                             ggml_type                   source_type) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, layout, source_type, match.input_size,
                                  match.output_size, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static void build_llm_dense_q6k_packed_raw_dispatch(const LlmDenseMatmulMatch & match,
                                                    DispatchMatch &             dispatch_match,
                                                    size_t                      root_index,
                                                    KernelCatalogRef            kernel,
                                                    const char *                layout) {
    const size_t block_count = static_cast<size_t>(match.input_size / ggml_blck_size(GGML_TYPE_Q6_K));
    const size_t materialized_weight_bytes =
        static_cast<size_t>(match.output_size) * block_count * ggml_type_size(GGML_TYPE_Q6_K);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, layout, GGML_TYPE_Q6_K,
                                  match.input_size, match.output_size, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_llm_dense_q4k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, context.plan, LlmDenseMatmulRoute::Q4K);
    if (!match.matched()) {
        return false;
    }
    const bool uses_low_row_q8 = match.token_count >= 2 && match.token_count <= 16 && match.output_size <= 64;
    if (uses_low_row_q8) {
        const size_t q8_bytes =
            static_cast<size_t>(match.token_count) * ggml_row_size(GGML_TYPE_Q8_1, match.input_size);
        const CommandPlanAlternateValue * q8_alternate =
            find_alternate_value(context.graph, context.plan, match.input->id, GGML_TYPE_Q8_1, q8_bytes);
        build_llm_dense_q4k_q8_1_x4_dispatch(match, dispatch_match, context, q8_alternate);
        return true;
    }
    const bool uses_low_row_u4 = match.token_count >= 1 && match.token_count <= 16;
    const bool uses_prefill_u4 = match.token_count % 128 == 0;
    if (match.input_size % 64 == 0 && match.output_size % 64 == 0 && (uses_low_row_u4 || uses_prefill_u4)) {
        const GraphNode * input_producer = context.graph.index().producer(match.input->id);
        const bool        uses_large_contracting_symmetric_i4 =
            uses_low_row_u4 && match.input_size >= 2 * match.output_size && input_producer != nullptr &&
            input_producer->op == GGML_OP_GLU && match.weight->byte_count >= size_t{ 16 } * 1024 * 1024;
        const bool uses_large_mild_expanding_symmetric_i4 = uses_low_row_u4 && match.output_size > match.input_size &&
                                                            match.output_size < 2 * match.input_size &&
                                                            match.weight->byte_count >= size_t{ 16 } * 1024 * 1024;
        const bool uses_symmetric_i4_low_row =
            uses_low_row_u4 && (match.output_size >= 2 * match.input_size || uses_large_contracting_symmetric_i4 ||
                                uses_large_mild_expanding_symmetric_i4);
        if (uses_symmetric_i4_low_row) {
            build_llm_dense_symmetric_i4_dispatch(match, dispatch_match, context);
            return true;
        }
        const bool uses_large_low_parallelism_split_k =
            match.weight->byte_count >= size_t{ 16 } * 1024 * 1024 && match.output_size <= 2 * match.input_size;
        const bool uses_low_row_split_k =
            uses_low_row_u4 && (match.input_size >= 2 * match.output_size || uses_large_low_parallelism_split_k);
        build_llm_dense_q4k_u4asym_dispatch(match, dispatch_match, context,
                                            uses_low_row_split_k ? kLlmDenseLinearQ4KU4AsymLowRowSplitK2Kernel :
                                            uses_low_row_u4      ? kLlmDenseLinearQ4KU4AsymLowRowKernel :
                                                                   kLlmDenseLinearQ4KU4AsymKernel);
    } else {
        build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    }
    return true;
}

static bool match_llm_dense_q6k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, context.plan, LlmDenseMatmulRoute::Q6K);
    if (!match.matched()) {
        return false;
    }
    if (match.token_count == 2) {
        const size_t q8_bytes =
            static_cast<size_t>(match.token_count) * ggml_row_size(GGML_TYPE_Q8_1, match.input_size);
        const CommandPlanAlternateValue * q8_alternate =
            find_alternate_value(context.graph, context.plan, match.input->id, GGML_TYPE_Q8_1, q8_bytes);
        build_llm_dense_q6k_q8_1_x4_dispatch(match, dispatch_match, context, q8_alternate);
    } else if (match.token_count >= 1 && match.token_count <= 16 && match.output_size % 64 == 0) {
        const bool    uses_expanding_packed_raw = match.token_count >= 3 && match.output_size / match.input_size >= 2;
        const int64_t contraction_ratio         = match.input_size / match.output_size;
        const bool    uses_contracting_packed_raw =
            match.token_count >= 3 && contraction_ratio >= 2 && contraction_ratio <= 4;
        if (uses_expanding_packed_raw) {
            build_llm_dense_q6k_packed_raw_dispatch(match, dispatch_match, context.root_index,
                                                    kLlmDenseLinearQ6KPackedRawF16WmmaLowRowKernel,
                                                    kQ6KPackedK256Row64Layout);
        } else if (uses_contracting_packed_raw) {
            build_llm_dense_q6k_packed_raw_dispatch(match, dispatch_match, context.root_index,
                                                    kLlmDenseLinearQ6KPackedRawF16WmmaScaleRowOneWaveKernel,
                                                    kQ6KPackedScaleRowLayout);
        } else {
            build_llm_dense_q6k_i8_prepacked_dispatch(match, dispatch_match, context.root_index);
        }
    } else {
        build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    }
    return true;
}

static bool match_llm_dense_q5k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, context.plan, LlmDenseMatmulRoute::Q5K);
    if (!match.matched()) {
        return false;
    }
    if (match.token_count >= 1 && match.token_count <= 16 && match.input_size % 64 == 0 &&
        match.output_size % 64 == 0) {
        build_llm_dense_symmetric_i4_dispatch(match, dispatch_match, context);
    } else {
        build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    }
    return true;
}

static bool match_llm_dense_iq4xs_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, context.plan, LlmDenseMatmulRoute::IQ4XS);
    if (!match.matched()) {
        return false;
    }
    if (match.token_count >= 1 && match.token_count <= 16 && match.input_size % 64 == 0 &&
        match.output_size % 64 == 0) {
        build_llm_dense_symmetric_i4_dispatch(match, dispatch_match, context);
    } else {
        build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    }
    return true;
}

void register_llm_matmul_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.matmul.dense_q4k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q4k_dispatch,
    });
    registry.add({
        "llm.matmul.dense_q5k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q5k_dispatch,
    });
    registry.add({
        "llm.matmul.dense_q6k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q6k_dispatch,
    });
    registry.add({
        "llm.matmul.dense_iq4xs_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_iq4xs_dispatch,
    });
}

}  // namespace ggml::hrx
