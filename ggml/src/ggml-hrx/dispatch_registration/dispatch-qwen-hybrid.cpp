#include "dispatch-qwen-hybrid.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenConcatWindowTailF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_concat_window_tail");
static constexpr KernelCatalogRef kQwenSsmConvF32PrefillKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_ssm_conv_f32_chan_concat_silu_regblock_wg1024");
static constexpr KernelCatalogRef kQwenSsmConvF32DecodeKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_ssm_conv_f32_state_materialized_decode_silu");
static constexpr KernelCatalogRef kQwenSsmConvF32RollbackKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_ssm_conv_f32_state_materialized_rollback_silu");
static constexpr KernelCatalogRef kQwenConcatDim0F32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_concat_dim0_f32");
static constexpr KernelCatalogRef kQwenGdnProjectionEpilogueF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_gdn_projection_epilogue_f32");
static constexpr KernelCatalogRef kQwenGdnF32PrefillKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head_rms_scale_fixed");
static constexpr KernelCatalogRef kQwenGdnF32StateCacheKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head_rms_scale_state_cache_fixed");
static constexpr KernelCatalogRef kQwenGdnF32SnapshotRollbackKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe",
                        "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head_rms_scale_snapshot_rollback_fixed");
static constexpr KernelCatalogRef kCopyF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "copy_f32_f32_contiguous_1d");
static constexpr KernelCatalogRef kQwenRecurrentRmsGateF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_recurrent_rms_raw_gate_silu_mul_f32");
static constexpr KernelCatalogRef kQwenRecurrentRmsGateF32F16Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_recurrent_rms_raw_gate_silu_mul_f32_f16");
static constexpr KernelCatalogRef kQwenRecurrentRmsGateF32I4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_recurrent_rms_raw_gate_silu_mul_f32_i4");
static constexpr KernelCatalogRef kQwenSwiGluSplitF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_swiglu_split_f32");
static constexpr KernelCatalogRef kQwenSwiGluSplitF32F16Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_swiglu_split_f32_f16");
static constexpr KernelCatalogRef kQwenSwiGluSplitF32I4ParallelKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_swiglu_split_f32_i4_parallel");
static constexpr KernelCatalogRef kQuantizeQ8_1X4Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
static constexpr KernelCatalogRef kQuantizeActI4Kernel  = GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_quant_act_i4");
static constexpr KernelCatalogRef kQuantizeActU4AsymKernel = GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_quant_act_u4asym");
static constexpr KernelCatalogRef kDenseSymmetricI4DualGateUpSwiGluF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_symi4_i4_dual_gate_up_swiglu_m32n32_f32out");
static constexpr KernelCatalogRef kDenseSymmetricI4DualGateUpSwiGluU4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_i4_dual_gate_up_swiglu_m32n32_u4out");
static constexpr KernelCatalogRef kDenseSymmetricI4DualGateUpSwiGluF16Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_i4_dual_gate_up_swiglu_m32n32_f16out");
static constexpr KernelCatalogRef kDenseQ4KU4AsymKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_u4asym_prepacked_wmmai4_64x128x64");
static constexpr KernelCatalogRef kDenseQ4KU4AsymDualGridLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_u4asym_prepacked_dual_grid_64x16x64");
static constexpr KernelCatalogRef kDenseSymmetricI4AdjacentDualGridLowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_symi4_i4_adjacent_dual_grid_m16n16_wg64");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 8192 && hidden_size <= 10240 && hidden_size % 32 == 0;
}

static bool is_f32(const Value * value) {
    return value != nullptr && value->type == GGML_TYPE_F32;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    return lhs.ne == rhs.ne;
}

static const Value * root_alias_value(const Graph & graph, const Value * value) {
    for (size_t depth = 0; value != nullptr && value->alias_source.value >= 0 && depth < graph.values().size();
         ++depth) {
        value = graph_value(graph, value->alias_source);
    }
    return value;
}

static bool distinct_storage(const Value & lhs, const Value & rhs) {
    return lhs.storage != rhs.storage;
}

static bool ranges_overlap(const Value & lhs, const Value & rhs) {
    if (lhs.storage != rhs.storage || lhs.byte_count == 0 || rhs.byte_count == 0) {
        return false;
    }
    if (lhs.storage_offset <= rhs.storage_offset) {
        return rhs.storage_offset - lhs.storage_offset < lhs.byte_count;
    }
    return lhs.storage_offset - rhs.storage_offset < rhs.byte_count;
}

static bool has_prefill_f16_dense_consumers_only(const Graph & graph, const Value & value) {
    if (!graph.has_index() || value.ne[0] <= 0 || value.element_count <= 0 || value.element_count % value.ne[0] != 0) {
        return false;
    }
    const int64_t token_count = value.element_count / value.ne[0];
    if (token_count < 128) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value.id);
    if (consumers.empty()) {
        return false;
    }
    for (const GraphNode * consumer : consumers) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
            consumer->inputs[1] != value.id) {
            return false;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        const Value * output = graph_value(graph, consumer->output);
        if (weight == nullptr || output == nullptr ||
            (weight->type != GGML_TYPE_Q5_K && weight->type != GGML_TYPE_IQ4_XS) || output->type != GGML_TYPE_F32 ||
            !weight->contiguous || !output->contiguous || weight->ne[0] != value.ne[0] || weight->ne[2] != 1 ||
            weight->ne[3] != 1 || output->ne[0] != weight->ne[1] || output->ne[1] != token_count ||
            output->ne[2] != 1 || output->ne[3] != 1) {
            return false;
        }
    }
    return true;
}

static bool has_prefill_q5_dense_consumer(const Graph & graph, const Value & value) {
    if (!has_prefill_f16_dense_consumers_only(graph, value)) {
        return false;
    }
    const int64_t token_count = value.element_count / value.ne[0];
    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        const Value * weight =
            consumer != nullptr && !consumer->inputs.empty() ? graph_value(graph, consumer->inputs[0]) : nullptr;
        if (weight != nullptr && weight->type == GGML_TYPE_Q5_K && token_count % 128 == 0 && weight->ne[1] % 64 == 0) {
            return true;
        }
    }
    return false;
}

static bool has_direct_low_row_symmetric_i4_consumer(const Graph & graph, const Value & value) {
    if (!graph.has_index() || value.ne[0] <= 0 || value.ne[0] > 32768 || value.ne[0] % 64 != 0 ||
        value.element_count <= 0 || value.element_count % value.ne[0] != 0) {
        return false;
    }
    const int64_t token_count = value.element_count / value.ne[0];
    if (token_count < 1 || token_count > 16) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
            consumer->inputs[1] != value.id) {
            continue;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        const Value * output = graph_value(graph, consumer->output);
        if (weight == nullptr || output == nullptr || output->type != GGML_TYPE_F32 || !weight->contiguous ||
            !output->contiguous || weight->ne[0] != value.ne[0] || weight->ne[1] != output->ne[0] ||
            output->ne[1] != token_count || output->ne[2] != 1 || output->ne[3] != 1 || output->ne[0] <= 0 ||
            output->ne[0] > 262144 || output->ne[0] % 64 != 0) {
            continue;
        }
        if (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_IQ4_XS ||
            (weight->type == GGML_TYPE_Q4_K && output->ne[0] >= 2 * value.ne[0])) {
            return true;
        }
    }
    return false;
}

static bool has_low_row_symmetric_i4_consumer(const Graph & graph, const Value & value) {
    if (has_direct_low_row_symmetric_i4_consumer(graph, value)) {
        return true;
    }
    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        if (consumer == nullptr || consumer->op != GGML_OP_RESHAPE || consumer->inputs.size() != 1 ||
            consumer->inputs.front() != value.id) {
            continue;
        }
        const Value * reshaped = graph_value(graph, consumer->output);
        if (reshaped != nullptr && reshaped->type == GGML_TYPE_F32 && reshaped->contiguous &&
            reshaped->storage_root == value.storage_root && reshaped->element_count == value.element_count &&
            reshaped->byte_count == value.byte_count && has_direct_low_row_symmetric_i4_consumer(graph, *reshaped)) {
            return true;
        }
    }
    return false;
}

static size_t q8_1_x4_byte_count(const Value & value) {
    const int64_t hidden_size = value.ne[0];
    const int64_t token_count = value.element_count / hidden_size;
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

struct QwenHybridSsmCacheUpdate {
    const GraphNode * state_tail_view = nullptr;
    const GraphNode * cache_view      = nullptr;
    const GraphNode * cache_copy      = nullptr;
    const Value *     state_tail      = nullptr;
    const Value *     cache           = nullptr;
    int64_t           source_row      = 0;
};

struct QwenHybridSsmPrefillMatch {
    const GraphNode *                     concat = nullptr;
    const GraphNode *                     ssm    = nullptr;
    const GraphNode *                     silu   = nullptr;
    const Value *                         state  = nullptr;
    const Value *                         x      = nullptr;
    const Value *                         filter = nullptr;
    const Value *                         output = nullptr;
    std::vector<QwenHybridSsmCacheUpdate> cache_updates;
    int64_t                               hidden_size = 0;
    int64_t                               token_count = 0;

    bool matched() const {
        return concat != nullptr && ssm != nullptr && silu != nullptr && state != nullptr && x != nullptr &&
               filter != nullptr && output != nullptr && !cache_updates.empty();
    }
};

static QwenHybridSsmPrefillMatch match_qwen_hybrid_ssm_prefill(const Graph & graph, const GraphNode * node) {
    QwenHybridSsmPrefillMatch match;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * state        = graph_value(graph, node->inputs[0]);
    const Value * x_transposed = graph_value(graph, node->inputs[1]);
    const Value * window       = graph_value(graph, node->output);
    if (!is_f32(state) || !is_f32(x_transposed) || !is_f32(window)) {
        return {};
    }

    const GraphNode * transpose = graph.index().producer(x_transposed->id);
    if (transpose == nullptr || transpose->op != GGML_OP_TRANSPOSE || transpose->inputs.size() != 1) {
        return {};
    }
    const Value * x = root_alias_value(graph, graph_value(graph, transpose->inputs[0]));
    if (!is_f32(x) || !x->contiguous) {
        return {};
    }

    const int64_t hidden_size = x->ne[0];
    const int64_t token_count = x->ne[1];
    if (!is_supported_hidden_size(hidden_size) || token_count < 1 || token_count > 512 ||
        !is_shape(*x, hidden_size, token_count, 1, 1) || !is_shape(*x_transposed, token_count, hidden_size, 1, 1) ||
        !is_shape(*state, 3, hidden_size, 1, 1) || !is_shape(*window, token_count + 3, hidden_size, 1, 1)) {
        return {};
    }
    if (x->nb[0] != sizeof(float) || x->nb[1] != static_cast<size_t>(hidden_size) * sizeof(float) ||
        x_transposed->nb[0] != static_cast<size_t>(hidden_size) * sizeof(float) ||
        x_transposed->nb[1] != sizeof(float) || state->nb[0] != sizeof(float) || state->nb[1] != 3 * sizeof(float) ||
        window->nb[0] != sizeof(float) || window->nb[1] != static_cast<size_t>(token_count + 3) * sizeof(float)) {
        return {};
    }

    const std::vector<const GraphNode *> & window_consumers = graph.index().consumers(window->id);
    if (window_consumers.size() < 2 || window_consumers.size() > 6) {
        return {};
    }
    std::vector<const GraphNode *> state_tail_views;
    const GraphNode *              ssm = nullptr;
    for (const GraphNode * consumer : window_consumers) {
        if (consumer != nullptr && consumer->op == GGML_OP_VIEW) {
            state_tail_views.push_back(consumer);
        } else if (consumer != nullptr && consumer->op == GGML_OP_SSM_CONV) {
            if (ssm != nullptr) {
                return {};
            }
            ssm = consumer;
        } else {
            return {};
        }
    }
    if (state_tail_views.empty() || state_tail_views.size() > 5 || ssm == nullptr || ssm->inputs.size() != 2 ||
        ssm->inputs[0] != window->id) {
        return {};
    }

    const Value * filter     = graph_value(graph, ssm->inputs[1]);
    const Value * ssm_output = graph_value(graph, ssm->output);
    if (!is_f32(filter) || !is_f32(ssm_output) || !filter->contiguous || !ssm_output->contiguous ||
        !is_shape(*filter, 4, hidden_size, 1, 1) || !is_shape(*ssm_output, hidden_size, token_count, 1, 1) ||
        filter->nb[0] != sizeof(float) || filter->nb[1] != 4 * sizeof(float)) {
        return {};
    }

    std::vector<QwenHybridSsmCacheUpdate> cache_updates;
    for (const GraphNode * state_tail_view : state_tail_views) {
        if (state_tail_view == nullptr || state_tail_view->inputs.size() != 1) {
            return {};
        }
        const Value * state_tail = graph_value(graph, state_tail_view->output);
        if (!is_f32(state_tail) || state_tail->alias_source != window->id ||
            state_tail->storage_offset % sizeof(float) != 0 || !is_shape(*state_tail, 3, hidden_size, 1, 1)) {
            return {};
        }
        const int64_t source_row = static_cast<int64_t>(state_tail->storage_offset / sizeof(float));
        if (source_row < 0 || source_row > token_count) {
            return {};
        }
        const std::vector<const GraphNode *> & tail_consumers = graph.index().consumers(state_tail->id);
        if (tail_consumers.size() != 1 || tail_consumers.front() == nullptr ||
            tail_consumers.front()->op != GGML_OP_CPY || tail_consumers.front()->inputs.size() != 2 ||
            tail_consumers.front()->inputs[0] != state_tail->id) {
            return {};
        }
        const GraphNode * cache_copy   = tail_consumers.front();
        const Value *     cache_target = graph_value(graph, cache_copy->inputs[1]);
        const Value *     cache        = graph_value(graph, cache_copy->output);
        const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
        if (!is_f32(cache_target) || !is_f32(cache) || cache_view == nullptr || cache_view->op != GGML_OP_VIEW ||
            cache_view->inputs.size() != 1 || !same_full_value_range(*cache_target, *cache) ||
            cache->byte_count != static_cast<size_t>(3 * hidden_size) * sizeof(float)) {
            return {};
        }
        cache_updates.push_back({ state_tail_view, cache_view, cache_copy, state_tail, cache, source_row });
    }

    std::sort(cache_updates.begin(), cache_updates.end(),
              [](const auto & lhs, const auto & rhs) { return lhs.cache->storage_offset < rhs.cache->storage_offset; });
    for (size_t slot = 0; slot < cache_updates.size(); ++slot) {
        const int64_t expected_source_row = std::max<int64_t>(0, token_count - static_cast<int64_t>(slot));
        if (cache_updates[slot].source_row != expected_source_row ||
            cache_updates[slot].cache->storage != cache_updates.front().cache->storage) {
            return {};
        }
        for (size_t prior = 0; prior < slot; ++prior) {
            if (ranges_overlap(*cache_updates[slot].cache, *cache_updates[prior].cache)) {
                return {};
            }
        }
    }

    const std::vector<const GraphNode *> & ssm_consumers = graph.index().consumers(ssm_output->id);
    if (ssm_consumers.size() != 1 || ssm_consumers.front() == nullptr || ssm_consumers.front()->op != GGML_OP_UNARY ||
        ssm_consumers.front()->inputs.size() != 1) {
        return {};
    }
    const GraphNode *   silu   = ssm_consumers.front();
    const UnaryParams * unary  = op_params_as<UnaryParams>(silu->params);
    const Value *       output = graph_value(graph, silu->output);
    if (unary == nullptr || unary->op != GGML_UNARY_OP_SILU || !is_f32(output) || !output->contiguous ||
        !is_shape(*output, hidden_size, token_count, 1, 1)) {
        return {};
    }

    if (!distinct_storage(*state, *x) || !distinct_storage(*state, *filter) || !distinct_storage(*x, *filter) ||
        (!distinct_storage(*output, *x) && !same_full_value_range(*output, *x)) || !distinct_storage(*output, *state) ||
        !distinct_storage(*output, *filter)) {
        return {};
    }
    for (const QwenHybridSsmCacheUpdate & update : cache_updates) {
        if (!distinct_storage(*state, *update.cache) || !distinct_storage(*x, *update.cache) ||
            !distinct_storage(*filter, *update.cache) || !distinct_storage(*output, *update.cache)) {
            return {};
        }
    }

    match.concat        = node;
    match.ssm           = ssm;
    match.silu          = silu;
    match.state         = state;
    match.x             = x;
    match.filter        = filter;
    match.output        = output;
    match.cache_updates = std::move(cache_updates);
    match.hidden_size   = hidden_size;
    match.token_count   = token_count;
    return match;
}

static const GraphNode * producer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && producer->op == op ? producer : nullptr;
}

static const GraphNode * single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    return consumers.size() == 1 && consumers.front() != nullptr && consumers.front()->op == op ? consumers.front() :
                                                                                                  nullptr;
}

static bool has_unary_op(const GraphNode * node, ggml_unary_op op) {
    const UnaryParams * params = node != nullptr ? op_params_as<UnaryParams>(node->params) : nullptr;
    return node != nullptr && node->op == GGML_OP_UNARY && params != nullptr && params->op == op;
}

struct QwenHybridGdnPrefillMatch {
    std::vector<const GraphNode *> covered;
    const Value *                  alpha_raw      = nullptr;
    const Value *                  beta_raw       = nullptr;
    const Value *                  bias           = nullptr;
    const Value *                  a_scale        = nullptr;
    const Value *                  gate           = nullptr;
    const Value *                  gate_flat      = nullptr;
    const Value *                  beta           = nullptr;
    const Value *                  raw_q          = nullptr;
    const Value *                  raw_k          = nullptr;
    const Value *                  v              = nullptr;
    const Value *                  state          = nullptr;
    const Value *                  gdn_output     = nullptr;
    const Value *                  new_state      = nullptr;
    const Value *                  cache          = nullptr;
    int64_t                        width          = 0;
    int64_t                        q_head_count   = 0;
    int64_t                        head_count     = 0;
    int64_t                        token_count    = 0;
    int64_t                        sequence_count = 0;
    int64_t                        snapshot_count = 0;
    float                          l2_epsilon      = 0.0f;

    bool matched() const {
        return !covered.empty() && alpha_raw != nullptr && beta_raw != nullptr && bias != nullptr &&
               a_scale != nullptr && gate != nullptr && gate_flat != nullptr && beta != nullptr && raw_q != nullptr &&
               raw_k != nullptr && v != nullptr && state != nullptr && gdn_output != nullptr && new_state != nullptr &&
               cache != nullptr;
    }
};

struct QwenHybridGdnProjectionPairMatch {
    const GraphNode * alpha_node   = nullptr;
    const GraphNode * beta_node    = nullptr;
    const Value *     alpha_weight = nullptr;
    const Value *     beta_weight  = nullptr;
    const Value *     input        = nullptr;
    const Value *     alpha_output = nullptr;
    const Value *     beta_output  = nullptr;
    ValueId           activation;
    int64_t           input_size  = 0;
    int64_t           output_size = 0;
    int64_t           token_count = 0;

    bool matched() const {
        return alpha_node != nullptr && beta_node != nullptr && alpha_weight != nullptr && beta_weight != nullptr &&
               input != nullptr && alpha_output != nullptr && beta_output != nullptr && activation.value >= 0;
    }
};

struct QwenHybridRecurrentRmsMatch {
    std::vector<const GraphNode *> covered;
    const Value *                  input        = nullptr;
    const Value *                  weight       = nullptr;
    const Value *                  raw_gate     = nullptr;
    const Value *                  output       = nullptr;
    const Value *                  f16_target   = nullptr;
    int64_t                        column_count = 0;
    int64_t                        row_count    = 0;

    bool matched() const {
        return !covered.empty() && input != nullptr && weight != nullptr && raw_gate != nullptr && output != nullptr;
    }
};

struct QwenDenseMixedGateUpMatch {
    const GraphNode * gate_node   = nullptr;
    const GraphNode * up_node     = nullptr;
    const GraphNode * glu_node    = nullptr;
    const Value *     gate_weight = nullptr;
    const Value *     up_weight   = nullptr;
    const Value *     input       = nullptr;
    const Value *     gate_output = nullptr;
    const Value *     up_output   = nullptr;
    const Value *     output      = nullptr;
    int64_t           input_size  = 0;
    int64_t           output_size = 0;
    int64_t           token_count = 0;

    bool matched() const {
        return gate_node != nullptr && up_node != nullptr && glu_node != nullptr && gate_weight != nullptr &&
               up_weight != nullptr && input != nullptr && gate_output != nullptr && up_output != nullptr &&
               output != nullptr;
    }
};

struct QwenDenseDownMatch {
    const GraphNode * node        = nullptr;
    const Value *     weight      = nullptr;
    const Value *     output      = nullptr;
    int64_t           output_size = 0;

    bool matched() const { return node != nullptr && weight != nullptr && output != nullptr; }
};

static bool is_dense_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_symmetric_i4_source_type(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS;
}

static bool is_supported_symmetric_i4_pair(const Value & lhs, const Value & rhs) {
    return (lhs.type == GGML_TYPE_Q4_K && rhs.type == GGML_TYPE_Q4_K) ||
           (lhs.type == GGML_TYPE_Q5_K && rhs.type == GGML_TYPE_IQ4_XS) ||
           (lhs.type == GGML_TYPE_IQ4_XS && rhs.type == GGML_TYPE_Q5_K);
}

static bool has_mixed_quant_symmetric_i4_consumer(const Graph &     graph,
                                                  const Value &     input,
                                                  const GraphNode * alpha_node,
                                                  const GraphNode * beta_node) {
    for (const GraphNode * consumer : graph.index().consumers(input.id)) {
        if (consumer == nullptr || consumer == alpha_node || consumer == beta_node || consumer->op != GGML_OP_MUL_MAT ||
            consumer->inputs.size() != 2 || consumer->inputs[1] != input.id) {
            continue;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        if (weight != nullptr && (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_IQ4_XS)) {
            return true;
        }
    }
    return false;
}

static QwenDenseMixedGateUpMatch match_qwen_dense_mixed_gate_up(const DispatchMatchContext & context) {
    QwenDenseMixedGateUpMatch match;
    const GraphNode *         root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT || root->inputs.size() != 2 || !context.graph.has_index()) {
        return match;
    }

    const std::vector<const GraphNode *> & root_consumers = context.graph.index().consumers(root->output);
    if (root_consumers.size() != 1 || root_consumers.front() == nullptr || root_consumers.front()->op != GGML_OP_GLU ||
        root_consumers.front()->inputs.size() != 2) {
        return {};
    }
    const GraphNode * glu_node   = root_consumers.front();
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root) ||
        gate_node->inputs.size() != 2 || up_node->inputs.size() != 2 || gate_node->inputs[1] != up_node->inputs[1]) {
        return {};
    }

    const Value * gate_weight = graph_value(context.graph, gate_node->inputs[0]);
    const Value * up_weight   = graph_value(context.graph, up_node->inputs[0]);
    const Value * input       = graph_value(context.graph, gate_node->inputs[1]);
    const Value * gate_output = graph_value(context.graph, gate_node->output);
    const Value * up_output   = graph_value(context.graph, up_node->output);
    const Value * output      = graph_value(context.graph, glu_node->output);
    if (gate_weight == nullptr || up_weight == nullptr || input == nullptr || gate_output == nullptr ||
        up_output == nullptr || output == nullptr || !is_supported_symmetric_i4_pair(*gate_weight, *up_weight) ||
        !is_symmetric_i4_source_type(gate_weight->type) || !is_symmetric_i4_source_type(up_weight->type) ||
        !is_dense_2d(*gate_weight) || !is_dense_2d(*up_weight) || !is_dense_2d(*input) || !is_dense_2d(*gate_output) ||
        !is_dense_2d(*up_output) || !is_dense_2d(*output) || !gate_weight->contiguous || !up_weight->contiguous ||
        !input->contiguous || !gate_output->contiguous || !up_output->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || gate_output->type != GGML_TYPE_F32 || up_output->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || gate_weight->alias_source.value >= 0 || up_weight->alias_source.value >= 0) {
        return {};
    }

    const int64_t input_size            = gate_weight->ne[0];
    const int64_t output_size           = gate_weight->ne[1];
    const int64_t token_count           = input->ne[1];
    const bool    supported_token_count = token_count == 512 || (token_count >= 1 && token_count <= 16);
    if (input_size < 256 || input_size > 32768 || input_size % 256 != 0 || output_size < 32 || output_size > 262144 ||
        output_size % 64 != 0 || !supported_token_count || input->ne[0] != input_size ||
        up_weight->ne[0] != input_size || up_weight->ne[1] != output_size ||
        !is_shape(*gate_output, output_size, token_count, 1, 1) ||
        !is_shape(*up_output, output_size, token_count, 1, 1) || !is_shape(*output, output_size, token_count, 1, 1)) {
        return {};
    }
    if (context.graph.index().consumers(gate_output->id).size() != 1 ||
        context.graph.index().consumers(up_output->id).size() != 1 || !distinct_storage(*gate_weight, *up_weight) ||
        !distinct_storage(*gate_weight, *input) || !distinct_storage(*up_weight, *input) ||
        !distinct_storage(*gate_output, *up_output) || !distinct_storage(*gate_output, *output) ||
        !distinct_storage(*up_output, *output)) {
        return {};
    }

    match.gate_node   = gate_node;
    match.up_node     = up_node;
    match.glu_node    = glu_node;
    match.gate_weight = gate_weight;
    match.up_weight   = up_weight;
    match.input       = input;
    match.gate_output = gate_output;
    match.up_output   = up_output;
    match.output      = output;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    return match;
}

static size_t symmetric_i4_weight_byte_count(int64_t input_size, int64_t output_size) {
    const size_t logical_output_size = static_cast<size_t>(output_size);
    const size_t row_group_size      = ((logical_output_size + 255) / 256) * 32;
    const size_t padded_output_size  = (logical_output_size + row_group_size - 1) / row_group_size * row_group_size;
    return padded_output_size * static_cast<size_t>(input_size / 256) * 132;
}

static DispatchBinding symmetric_i4_weight_binding(const Value & weight, int64_t input_size, int64_t output_size) {
    DispatchBinding binding;
    binding.value         = weight.id;
    binding.length        = symmetric_i4_weight_byte_count(input_size, output_size);
    binding.layout        = kSymmetricI4K32EightGroupsShared4Layout;
    binding.source_type   = weight.type;
    binding.input_size    = input_size;
    binding.output_size   = output_size;
    binding.source_length = weight.byte_count;
    return binding;
}

static DispatchBinding symmetric_i4_prefill_weight_binding(const Value & weight,
                                                           int64_t       input_size,
                                                           int64_t       output_size) {
    DispatchBinding binding;
    binding.value         = weight.id;
    binding.length        = static_cast<size_t>(output_size) * static_cast<size_t>(input_size / 256) * 144;
    binding.layout        = kSymmetricI4K64Row64Layout;
    binding.source_type   = weight.type;
    binding.input_size    = input_size;
    binding.output_size   = output_size;
    binding.source_length = weight.byte_count;
    return binding;
}

static DispatchBinding q4_k_i4_k32_weight_binding(const Value & weight, int64_t input_size, int64_t output_size) {
    DispatchBinding binding;
    binding.value         = weight.id;
    binding.length        = weight.byte_count;
    binding.layout        = kQ4KI4K32Row64Layout;
    binding.source_type   = GGML_TYPE_Q4_K;
    binding.input_size    = input_size;
    binding.output_size   = output_size;
    binding.source_length = weight.byte_count;
    return binding;
}

static QwenDenseDownMatch match_qwen_dense_down(const Graph &                     graph,
                                                const QwenDenseMixedGateUpMatch & gate_up,
                                                ggml_type                         weight_type) {
    QwenDenseDownMatch                     match;
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(gate_up.output->id);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->op != GGML_OP_MUL_MAT ||
        consumers.front()->inputs.size() != 2 || consumers.front()->inputs[1] != gate_up.output->id) {
        return match;
    }
    const GraphNode * node   = consumers.front();
    const Value *     weight = graph_value(graph, node->inputs[0]);
    const Value *     output = graph_value(graph, node->output);
    if (weight == nullptr || output == nullptr || weight->type != weight_type || output->type != GGML_TYPE_F32 ||
        !is_dense_2d(*weight) || !is_dense_2d(*output) || !weight->contiguous || !output->contiguous ||
        weight->alias_source.value >= 0 || weight->ne[0] != gate_up.output_size || weight->ne[1] <= 0 ||
        weight->ne[1] > 262144 || weight->ne[1] % 64 != 0 ||
        !is_shape(*output, weight->ne[1], gate_up.token_count, 1, 1) || !distinct_storage(*weight, *gate_up.output) ||
        !distinct_storage(*weight, *output) || !distinct_storage(*gate_up.output, *output)) {
        return {};
    }
    match.node        = node;
    match.weight      = weight;
    match.output      = output;
    match.output_size = weight->ne[1];
    return match;
}

static QwenHybridRecurrentRmsMatch match_qwen_hybrid_recurrent_rms(const Graph & graph, const GraphNode * node) {
    QwenHybridRecurrentRmsMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1) {
        return match;
    }
    const RmsNormParams * params = op_params_as<RmsNormParams>(node->params);
    const Value *         input  = graph_value(graph, node->inputs[0]);
    const Value *         rms    = graph_value(graph, node->output);
    if (params == nullptr || std::fabs(params->eps - 1.0e-6f) > 1.0e-12f || !is_f32(input) || !is_f32(rms) ||
        !input->contiguous || !rms->contiguous || !same_shape(*input, *rms)) {
        return {};
    }

    const std::vector<const GraphNode *> & rms_consumers = graph.index().consumers(rms->id);
    if (rms_consumers.size() != 1 || rms_consumers.front() == nullptr || rms_consumers.front()->op != GGML_OP_MUL ||
        rms_consumers.front()->inputs.size() != 2) {
        return {};
    }
    const GraphNode * rms_mul = rms_consumers.front();
    const Value * weight = graph_value(graph, rms_mul->inputs[0] == rms->id ? rms_mul->inputs[1] : rms_mul->inputs[0]);
    const Value * side   = graph_value(graph, rms_mul->output);
    if (!is_f32(weight) || !is_f32(side) || !weight->contiguous || !side->contiguous || !same_shape(*input, *side) ||
        !is_shape(*weight, input->ne[0], 1, 1, 1)) {
        return {};
    }

    const std::vector<const GraphNode *> & side_consumers = graph.index().consumers(side->id);
    if (side_consumers.size() != 1 || side_consumers.front() == nullptr || side_consumers.front()->op != GGML_OP_MUL ||
        side_consumers.front()->inputs.size() != 2) {
        return {};
    }
    const GraphNode * terminal = side_consumers.front();
    const Value *     activated =
        graph_value(graph, terminal->inputs[0] == side->id ? terminal->inputs[1] : terminal->inputs[0]);
    const GraphNode * silu = activated != nullptr ? producer_with_op(graph, activated->id, GGML_OP_UNARY) : nullptr;
    if (!has_unary_op(silu, GGML_UNARY_OP_SILU) || silu->inputs.size() != 1) {
        return {};
    }
    const Value *     gate_view = graph_value(graph, silu->inputs[0]);
    const GraphNode * gate_reshape =
        gate_view != nullptr ? producer_with_op(graph, gate_view->id, GGML_OP_RESHAPE) : nullptr;
    const Value *     raw_gate = gate_reshape != nullptr && gate_reshape->inputs.size() == 1 ?
                                     graph_value(graph, gate_reshape->inputs[0]) :
                                     nullptr;
    const GraphNode * gate_projection =
        raw_gate != nullptr ? producer_with_op(graph, raw_gate->id, GGML_OP_MUL_MAT) : nullptr;
    const Value * gate_projection_weight = gate_projection != nullptr && !gate_projection->inputs.empty() ?
                                               graph_value(graph, gate_projection->inputs[0]) :
                                               nullptr;
    const Value * output                 = graph_value(graph, terminal->output);
    if (!is_f32(gate_view) || !is_f32(raw_gate) || !is_f32(activated) || !is_f32(output) ||
        gate_projection_weight == nullptr ||
        (gate_projection_weight->type != GGML_TYPE_Q4_K && gate_projection_weight->type != GGML_TYPE_Q5_K) ||
        !raw_gate->contiguous || !activated->contiguous || !output->contiguous || !same_shape(*input, *gate_view) ||
        !same_shape(*input, *activated) || !same_shape(*input, *output) ||
        raw_gate->element_count != input->element_count) {
        return {};
    }

    const int64_t column_count = input->ne[0];
    if (column_count <= 0 || column_count > 65536 || column_count % 4 != 0 || input->element_count <= 0 ||
        input->element_count % column_count != 0 || !distinct_storage(*input, *weight) ||
        !distinct_storage(*input, *raw_gate) || !distinct_storage(*input, *output) ||
        !distinct_storage(*weight, *raw_gate) || !distinct_storage(*weight, *output) ||
        !distinct_storage(*raw_gate, *output)) {
        return {};
    }

    match.covered      = { node, rms_mul, gate_reshape, silu, terminal };
    match.input        = input;
    match.weight       = weight;
    match.raw_gate     = raw_gate;
    match.output       = output;
    match.column_count = column_count;
    match.row_count    = input->element_count / column_count;
    if (match.row_count > 1048576) {
        return {};
    }
    if (has_prefill_f16_dense_consumers_only(graph, *output)) {
        match.f16_target = output;
    } else {
        const std::vector<const GraphNode *> & output_consumers = graph.index().consumers(output->id);
        const GraphNode * output_reshape = output_consumers.size() == 1 && output_consumers.front() != nullptr &&
                                                   output_consumers.front()->op == GGML_OP_RESHAPE &&
                                                   output_consumers.front()->inputs.size() == 1 &&
                                                   output_consumers.front()->inputs.front() == output->id ?
                                               output_consumers.front() :
                                               nullptr;
        const Value *     reshaped = output_reshape != nullptr ? graph_value(graph, output_reshape->output) : nullptr;
        if (reshaped != nullptr && reshaped->type == GGML_TYPE_F32 && reshaped->contiguous &&
            reshaped->storage_root == output->storage_root && reshaped->element_count == output->element_count &&
            reshaped->byte_count == output->byte_count && has_prefill_f16_dense_consumers_only(graph, *reshaped)) {
            match.covered.push_back(output_reshape);
            match.f16_target = reshaped;
        }
    }
    return match;
}

static QwenHybridGdnPrefillMatch match_qwen_hybrid_gdn_prefill(const Graph & graph, const GraphNode * node) {
    QwenHybridGdnPrefillMatch match;
    if (node == nullptr || node->op != GGML_OP_L2_NORM || node->inputs.size() != 1) {
        return match;
    }

    const Value * raw_q  = graph_value(graph, node->inputs[0]);
    const Value * q_norm = graph_value(graph, node->output);
    if (!is_f32(raw_q) || !is_f32(q_norm)) {
        return {};
    }
    const std::vector<const GraphNode *> & q_consumers = graph.index().consumers(q_norm->id);
    if (q_consumers.size() != 1 || q_consumers.front() == nullptr ||
        q_consumers.front()->op != GGML_OP_GATED_DELTA_NET || q_consumers.front()->inputs.size() != 6 ||
        q_consumers.front()->inputs[0] != q_norm->id) {
        return {};
    }
    const GraphNode * gdn = q_consumers.front();

    const Value *     k_norm      = graph_value(graph, gdn->inputs[1]);
    const Value *     v           = graph_value(graph, gdn->inputs[2]);
    const Value *     gate        = graph_value(graph, gdn->inputs[3]);
    const Value *     beta        = graph_value(graph, gdn->inputs[4]);
    const Value *     state       = graph_value(graph, gdn->inputs[5]);
    const Value *     gdn_output  = graph_value(graph, gdn->output);
    const GraphNode * k_norm_node = k_norm != nullptr ? producer_with_op(graph, k_norm->id, GGML_OP_L2_NORM) : nullptr;
    const Value *     raw_k       = k_norm_node != nullptr && k_norm_node->inputs.size() == 1 ?
                                        graph_value(graph, k_norm_node->inputs[0]) :
                                        nullptr;
    if (!is_f32(k_norm) || !is_f32(raw_k) || !is_f32(v) || !is_f32(gate) || !is_f32(beta) || !is_f32(state) ||
        !is_f32(gdn_output)) {
        return {};
    }
    const L2NormParams * l2_params = op_params_as<L2NormParams>(node->params);
    if (l2_params == nullptr || !std::isfinite(l2_params->eps) || l2_params->eps <= 0.0f ||
        !op_params_equivalent(GGML_OP_L2_NORM, node->params, k_norm_node->params)) {
        return {};
    }

    const int64_t width          = raw_q->ne[0];
    const int64_t q_head_count   = raw_q->ne[1];
    const int64_t token_count    = raw_q->ne[2];
    const int64_t sequence_count = raw_q->ne[3];
    const int64_t head_count     = v->ne[1];
    if (width != 128 || q_head_count <= 0 || q_head_count > 4096 || head_count <= 0 || head_count > 4096 ||
        token_count < 1 || token_count > 512 || sequence_count != 1) {
        return {};
    }
    const int64_t hidden_size = width * (2 * q_head_count + head_count);
    if (!is_shape(*raw_k, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*q_norm, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*k_norm, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*v, width, head_count, token_count, sequence_count) ||
        !is_shape(*gate, 1, head_count, token_count, sequence_count) ||
        !is_shape(*beta, 1, head_count, token_count, sequence_count) ||
        !is_shape(*state, width, width, head_count, sequence_count) || gdn_output->ne[0] != width * head_count ||
        gdn_output->ne[2] != 1 || gdn_output->ne[3] != 1 || gdn_output->ne[1] <= token_count ||
        (gdn_output->ne[1] - token_count) % width != 0) {
        return {};
    }
    const int64_t snapshot_count = (gdn_output->ne[1] - token_count) / width;
    if (snapshot_count < 1 || snapshot_count > 5) {
        return {};
    }
    if (!q_norm->contiguous || !k_norm->contiguous || !state->contiguous || !gdn_output->contiguous ||
        raw_q->storage != raw_k->storage || raw_q->storage != v->storage || raw_q->storage_offset != 0 ||
        raw_k->storage_offset != static_cast<size_t>(width * q_head_count) * sizeof(float) ||
        v->storage_offset != static_cast<size_t>(2 * width * q_head_count) * sizeof(float) ||
        raw_q->nb[0] != sizeof(float) || raw_q->nb[1] != static_cast<size_t>(width) * sizeof(float) ||
        raw_q->nb[2] != static_cast<size_t>(hidden_size) * sizeof(float) || raw_k->nb != raw_q->nb ||
        v->nb[0] != sizeof(float) || v->nb[1] != static_cast<size_t>(width) * sizeof(float) ||
        v->nb[2] != static_cast<size_t>(hidden_size) * sizeof(float)) {
        return {};
    }

    const GraphNode * gate_reshape = producer_with_op(graph, gate->id, GGML_OP_RESHAPE);
    const Value *     gate_flat    = gate_reshape != nullptr && gate_reshape->inputs.size() == 1 ?
                                         graph_value(graph, gate_reshape->inputs[0]) :
                                         nullptr;
    const GraphNode * gate_mul = gate_flat != nullptr ? producer_with_op(graph, gate_flat->id, GGML_OP_MUL) : nullptr;
    if (gate_mul == nullptr || gate_mul->inputs.size() != 2) {
        return {};
    }
    const Value *     alpha_softplus = graph_value(graph, gate_mul->inputs[0]);
    const Value *     a_scale        = graph_value(graph, gate_mul->inputs[1]);
    const GraphNode * softplus =
        alpha_softplus != nullptr ? producer_with_op(graph, alpha_softplus->id, GGML_OP_UNARY) : nullptr;
    if (!has_unary_op(softplus, GGML_UNARY_OP_SOFTPLUS) || softplus->inputs.size() != 1) {
        return {};
    }
    const Value *     alpha_biased = graph_value(graph, softplus->inputs[0]);
    const GraphNode * add_bias =
        alpha_biased != nullptr ? producer_with_op(graph, alpha_biased->id, GGML_OP_ADD) : nullptr;
    if (add_bias == nullptr || add_bias->inputs.size() != 2) {
        return {};
    }
    const Value *     alpha         = graph_value(graph, add_bias->inputs[0]);
    const Value *     bias          = graph_value(graph, add_bias->inputs[1]);
    const GraphNode * alpha_reshape = alpha != nullptr ? producer_with_op(graph, alpha->id, GGML_OP_RESHAPE) : nullptr;
    const Value *     alpha_raw     = alpha_reshape != nullptr && alpha_reshape->inputs.size() == 1 ?
                                          graph_value(graph, alpha_reshape->inputs[0]) :
                                          nullptr;

    const GraphNode * sigmoid = producer_with_op(graph, beta->id, GGML_OP_UNARY);
    if (!has_unary_op(sigmoid, GGML_UNARY_OP_SIGMOID) || sigmoid->inputs.size() != 1) {
        return {};
    }
    const Value *     beta_pre = graph_value(graph, sigmoid->inputs[0]);
    const GraphNode * beta_reshape =
        beta_pre != nullptr ? producer_with_op(graph, beta_pre->id, GGML_OP_RESHAPE) : nullptr;
    const Value * beta_raw = beta_reshape != nullptr && beta_reshape->inputs.size() == 1 ?
                                 graph_value(graph, beta_reshape->inputs[0]) :
                                 nullptr;
    if (!is_f32(alpha_raw) || !is_f32(beta_raw) || !is_f32(alpha) || !is_f32(alpha_biased) || !is_f32(alpha_softplus) ||
        !is_f32(a_scale) || !is_f32(bias) || !is_f32(gate_flat) || !is_f32(beta_pre) ||
        !is_shape(*alpha_raw, head_count, token_count, 1, 1) || !is_shape(*beta_raw, head_count, token_count, 1, 1) ||
        !is_shape(*gate_flat, head_count, token_count, 1, 1) || !is_shape(*bias, head_count, 1, 1, 1) ||
        !is_shape(*a_scale, head_count, 1, 1, 1) || !alpha_raw->contiguous || !beta_raw->contiguous ||
        !bias->contiguous || !a_scale->contiguous || !gate_flat->contiguous || !beta->contiguous) {
        return {};
    }

    const std::array<const Value *, 6> epilogue_values = { alpha_raw, beta_raw, bias, a_scale, gate_flat, beta };
    for (size_t lhs = 0; lhs < epilogue_values.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < epilogue_values.size(); ++rhs) {
            if (!distinct_storage(*epilogue_values[lhs], *epilogue_values[rhs])) {
                return {};
            }
        }
    }

    const GraphNode * state_reshape = producer_with_op(graph, state->id, GGML_OP_RESHAPE);
    if (state_reshape == nullptr || state_reshape->inputs.size() != 1) {
        return {};
    }
    const std::vector<const GraphNode *> & gdn_consumers = graph.index().consumers(gdn_output->id);
    if (gdn_consumers.size() != 2) {
        return {};
    }
    const GraphNode * new_state_view  = nullptr;
    const GraphNode * attention_view  = nullptr;
    const size_t      attention_bytes = static_cast<size_t>(width * head_count * token_count) * sizeof(float);
    for (const GraphNode * consumer : gdn_consumers) {
        const Value * output =
            consumer != nullptr && consumer->op == GGML_OP_VIEW ? graph_value(graph, consumer->output) : nullptr;
        if (output != nullptr && output->storage == gdn_output->storage &&
            output->storage_offset == gdn_output->storage_offset + attention_bytes) {
            new_state_view = consumer;
        } else if (output != nullptr && output->storage == gdn_output->storage &&
                   output->storage_offset == gdn_output->storage_offset &&
                   is_shape(*output, width, head_count, token_count, sequence_count)) {
            attention_view = consumer;
        } else {
            return {};
        }
    }
    const Value * new_state = new_state_view != nullptr ? graph_value(graph, new_state_view->output) : nullptr;
    if (new_state == nullptr || attention_view == nullptr) {
        return {};
    }
    const int64_t written_snapshot_count = std::min(token_count, snapshot_count);
    const size_t  state_bytes            = static_cast<size_t>(width * width * head_count) * sizeof(float);
    const bool    single_snapshot_state =
        snapshot_count == 1 && is_shape(*new_state, width, width, head_count, sequence_count);
    const bool rollback_snapshot_state =
        is_shape(*new_state, width * width * head_count, sequence_count, written_snapshot_count, 1);
    if ((!single_snapshot_state && !rollback_snapshot_state) || !new_state->contiguous ||
        new_state->storage != gdn_output->storage ||
        new_state->storage_offset != gdn_output->storage_offset + attention_bytes ||
        new_state->byte_count != state_bytes * static_cast<size_t>(written_snapshot_count)) {
        return {};
    }
    const std::vector<const GraphNode *> & state_consumers = graph.index().consumers(new_state->id);
    if (state_consumers.size() != 1 || state_consumers.front() == nullptr ||
        state_consumers.front()->op != GGML_OP_CPY || state_consumers.front()->inputs.size() != 2 ||
        state_consumers.front()->inputs[0] != new_state->id) {
        return {};
    }
    const GraphNode * cache_copy   = state_consumers.front();
    const Value *     cache_target = graph_value(graph, cache_copy->inputs[1]);
    const Value *     cache        = graph_value(graph, cache_copy->output);
    const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
    if (!is_f32(cache_target) || !is_f32(cache) || cache_view == nullptr) {
        return {};
    }
    const size_t snapshot_stride_count = static_cast<size_t>(written_snapshot_count - 1);
    if (snapshot_stride_count != 0 &&
        cache->nb[2] > (std::numeric_limits<size_t>::max() - state_bytes) / snapshot_stride_count) {
        return {};
    }
    const size_t required_cache_bytes = state_bytes + snapshot_stride_count * cache->nb[2];
    if (cache_view->op != GGML_OP_VIEW || cache_view->inputs.size() != 1 ||
        !same_full_value_range(*cache_target, *cache) || cache->ne[0] != width * width * head_count ||
        cache->ne[1] != sequence_count || cache->ne[2] != written_snapshot_count || cache->ne[3] != 1 ||
        cache->nb[0] != sizeof(float) || cache->nb[1] != state_bytes ||
        (written_snapshot_count > 1 && cache->nb[2] < state_bytes) || cache->byte_count < required_cache_bytes ||
        !distinct_storage(*new_state, *cache)) {
        return {};
    }

    if (!distinct_storage(*gdn_output, *state) || !distinct_storage(*gdn_output, *gate_flat) ||
        !distinct_storage(*gdn_output, *beta)) {
        return {};
    }

    match.covered        = { alpha_reshape, add_bias,       softplus,   gate_mul,    gate_reshape,
                             beta_reshape,  sigmoid,        node,       k_norm_node, state_reshape,
                             gdn,           new_state_view, cache_view, cache_copy,  attention_view };
    match.alpha_raw      = alpha_raw;
    match.beta_raw       = beta_raw;
    match.bias           = bias;
    match.a_scale        = a_scale;
    match.gate           = gate;
    match.gate_flat      = gate_flat;
    match.beta           = beta;
    match.raw_q          = raw_q;
    match.raw_k          = raw_k;
    match.v              = v;
    match.state          = state;
    match.gdn_output     = gdn_output;
    match.new_state      = new_state;
    match.cache          = cache;
    match.width          = width;
    match.q_head_count   = q_head_count;
    match.head_count     = head_count;
    match.token_count    = token_count;
    match.sequence_count = sequence_count;
    match.snapshot_count = snapshot_count;
    match.l2_epsilon      = l2_params->eps;
    return match;
}

static QwenHybridGdnProjectionPairMatch match_qwen_hybrid_gdn_projection_pair(const DispatchMatchContext & context) {
    QwenHybridGdnProjectionPairMatch match;
    const GraphNode *                alpha_node = context.root_node;
    if (alpha_node == nullptr || alpha_node->op != GGML_OP_MUL_MAT || alpha_node->inputs.size() != 2 ||
        !context.graph.has_index()) {
        return match;
    }

    const GraphNode * alpha_reshape = single_consumer_with_op(context.graph, alpha_node->output, GGML_OP_RESHAPE);
    const GraphNode * add_bias =
        alpha_reshape != nullptr ? single_consumer_with_op(context.graph, alpha_reshape->output, GGML_OP_ADD) : nullptr;
    const GraphNode * softplus =
        add_bias != nullptr ? single_consumer_with_op(context.graph, add_bias->output, GGML_OP_UNARY) : nullptr;
    const GraphNode * gate_mul = softplus != nullptr && has_unary_op(softplus, GGML_UNARY_OP_SOFTPLUS) ?
                                     single_consumer_with_op(context.graph, softplus->output, GGML_OP_MUL) :
                                     nullptr;
    const GraphNode * gate_reshape =
        gate_mul != nullptr ? single_consumer_with_op(context.graph, gate_mul->output, GGML_OP_RESHAPE) : nullptr;
    const GraphNode * gdn = gate_reshape != nullptr ?
                                single_consumer_with_op(context.graph, gate_reshape->output, GGML_OP_GATED_DELTA_NET) :
                                nullptr;
    if (gdn == nullptr || gdn->inputs.size() != 6) {
        return {};
    }

    const GraphNode *               q_norm    = producer_with_op(context.graph, gdn->inputs[0], GGML_OP_L2_NORM);
    const QwenHybridGdnPrefillMatch gdn_match = match_qwen_hybrid_gdn_prefill(context.graph, q_norm);
    if (!gdn_match.matched() || gdn_match.alpha_raw->id != alpha_node->output) {
        return {};
    }

    const GraphNode * beta_node = producer_with_op(context.graph, gdn_match.beta_raw->id, GGML_OP_MUL_MAT);
    if (beta_node == nullptr || beta_node == alpha_node || beta_node->inputs.size() != 2 ||
        beta_node->inputs[1] != alpha_node->inputs[1]) {
        return {};
    }

    const Value * alpha_weight = graph_value(context.graph, alpha_node->inputs[0]);
    const Value * beta_weight  = graph_value(context.graph, beta_node->inputs[0]);
    const Value * input        = graph_value(context.graph, alpha_node->inputs[1]);
    const Value * alpha_output = graph_value(context.graph, alpha_node->output);
    const Value * beta_output  = graph_value(context.graph, beta_node->output);
    if (alpha_weight == nullptr || beta_weight == nullptr || input == nullptr || alpha_output == nullptr ||
        beta_output == nullptr || alpha_weight->type != GGML_TYPE_Q4_K || beta_weight->type != GGML_TYPE_Q4_K ||
        input->type != GGML_TYPE_F32 || alpha_output->type != GGML_TYPE_F32 || beta_output->type != GGML_TYPE_F32 ||
        !is_dense_2d(*alpha_weight) || !is_dense_2d(*beta_weight) || !is_dense_2d(*input) ||
        !is_dense_2d(*alpha_output) || !is_dense_2d(*beta_output) || !alpha_weight->contiguous ||
        !beta_weight->contiguous || !input->contiguous || !alpha_output->contiguous || !beta_output->contiguous ||
        alpha_weight->alias_source.value >= 0 || beta_weight->alias_source.value >= 0) {
        return {};
    }

    const int64_t input_size  = alpha_weight->ne[0];
    const int64_t output_size = alpha_weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input_size < 256 || input_size > 32768 || input_size % 256 != 0 || output_size < 1 || output_size > 256 ||
        token_count < 1 || token_count > 16 || input->ne[0] != input_size || beta_weight->ne[0] != input_size ||
        beta_weight->ne[1] != output_size || output_size != gdn_match.head_count ||
        token_count != gdn_match.token_count || !is_shape(*alpha_output, output_size, token_count, 1, 1) ||
        !is_shape(*beta_output, output_size, token_count, 1, 1) || !distinct_storage(*alpha_weight, *beta_weight) ||
        !distinct_storage(*alpha_weight, *input) || !distinct_storage(*beta_weight, *input) ||
        !distinct_storage(*alpha_output, *beta_output) ||
        !has_mixed_quant_symmetric_i4_consumer(context.graph, *input, alpha_node, beta_node)) {
        return {};
    }

    const LlmSymmetricI4ActivationLayout activation_layout =
        llm_symmetric_i4_activation_layout(input_size, token_count);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(context.graph, context.plan, input->id, GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate == nullptr || alternate->name != kLlmSymmetricI4ActivationAlternateName) {
        return {};
    }

    match.alpha_node   = alpha_node;
    match.beta_node    = beta_node;
    match.alpha_weight = alpha_weight;
    match.beta_weight  = beta_weight;
    match.input        = input;
    match.alpha_output = alpha_output;
    match.beta_output  = beta_output;
    match.activation   = alternate->alternate_value;
    match.input_size   = input_size;
    match.output_size  = output_size;
    match.token_count  = token_count;
    return match;
}

static void set_compile_parameter(KernelSpecialization & kernel, const char * name, int64_t value) {
    kernel.compile_parameters.emplace(name, std::to_string(value));
}

static void set_float_compile_parameter(KernelSpecialization & kernel, const char * name, float value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    kernel.compile_parameters.emplace(name, stream.str());
}

static void configure_qwen_gdn_kernel(Dispatch &                        dispatch,
                                      const QwenHybridGdnPrefillMatch & match,
                                      int64_t                           token_count) {
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_s_v", match.width);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_n_heads", match.head_count);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_n_tokens", token_count);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_n_seqs", match.sequence_count);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_qk_scale_s1", match.raw_q->nb[1] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_qk_scale_s2", match.raw_q->nb[2] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_qk_scale_s3", match.raw_q->nb[3] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_sv1", match.v->nb[1] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_sv2", match.v->nb[2] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_sv3", match.v->nb[3] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_sb1", match.beta->nb[1] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_sb2", match.beta->nb[2] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_sb3", match.beta->nb[3] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_neqk1", match.q_head_count);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_gdn_rq3", match.sequence_count);
    set_float_compile_parameter(dispatch.kernel, "hrx2_gdn_l2_epsilon", match.l2_epsilon);
    set_compile_parameter(dispatch.kernel, "hrx2_tuning_gdn_workgroup_size", 256);
}

}  // namespace

static bool match_qwen_dense_mixed_gate_up_dispatch(const DispatchMatchContext & context,
                                                    DispatchMatch &              dispatch_match) {
    const QwenDenseMixedGateUpMatch match = match_qwen_dense_mixed_gate_up(context);
    if (!match.matched()) {
        return false;
    }
    const bool q4_gate_up = match.gate_weight->type == GGML_TYPE_Q4_K && match.up_weight->type == GGML_TYPE_Q4_K;
    if (match.token_count >= 1 && match.token_count <= 16 && match.input_size % 64 == 0) {
        if (!append_covered_node(context, match.gate_node, dispatch_match) ||
            !append_covered_node(context, match.up_node, dispatch_match)) {
            return false;
        }

        const LlmSymmetricI4ActivationLayout activation_layout =
            llm_symmetric_i4_activation_layout(match.input_size, match.token_count);
        const CommandPlanAlternateValue * alternate = find_alternate_value(
            context.graph, context.plan, match.input->id, GGML_TYPE_COUNT, activation_layout.total_bytes);
        if (alternate != nullptr && alternate->name != kLlmSymmetricI4ActivationAlternateName) {
            alternate = nullptr;
        }
        const ValueId activation = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
        if (alternate == nullptr) {
            dispatch_match.transients.push_back(
                { activation, kLlmSymmetricI4ActivationAlternateName, activation_layout.total_bytes, 256 });

            Dispatch quantize;
            quantize.kernel = make_kernel_specialization(kQuantizeActI4Kernel);
            set_compile_parameter(quantize.kernel, "qwen3.qact_i4.shape_k", match.input_size);
            set_compile_parameter(quantize.kernel, "qwen3.qact_i4.shape_cols", match.token_count);
            quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
            quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
            quantize.bindings.push_back(
                { activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
            quantize.bindings.push_back(
                { activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
            dispatch_match.dispatches.push_back(std::move(quantize));
        }

        Dispatch gate_up;
        gate_up.kernel = make_kernel_specialization(kDenseSymmetricI4AdjacentDualGridLowRowKernel);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.lowrow.shape_k", match.input_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.lowrow.shape_rows", match.output_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.lowrow.shape_cols", match.token_count);
        gate_up.bindings.push_back(
            symmetric_i4_weight_binding(*match.gate_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back(symmetric_i4_weight_binding(*match.up_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back({ match.gate_output->id, 0, match.gate_output->byte_count });
        gate_up.bindings.push_back({ match.up_output->id, 0, match.up_output->byte_count });
        gate_up.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        gate_up.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        gate_up.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

        dispatch_match.dispatches.push_back(std::move(gate_up));
        return true;
    }
    if (q4_gate_up && match.token_count >= 1 && match.token_count <= 16) {
        if (!append_covered_node(context, match.gate_node, dispatch_match) ||
            !append_covered_node(context, match.up_node, dispatch_match)) {
            return false;
        }

        const size_t  input_elements = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size);
        const size_t  qact_bytes     = input_elements / 2;
        const size_t  scale_bytes    = input_elements / 8;
        const size_t  meta_bytes     = input_elements / 4;
        const ValueId qact(context.next_plan_value.value);
        const ValueId scales(context.next_plan_value.value + 1);
        const ValueId metadata(context.next_plan_value.value + 2);
        dispatch_match.transients.push_back({ qact, "qwen.decode.u4_gate_up_qs", qact_bytes, 256 });
        dispatch_match.transients.push_back({ scales, "qwen.decode.u4_gate_up_ds", scale_bytes, 256 });
        dispatch_match.transients.push_back({ metadata, "qwen.decode.u4_gate_up_meta", meta_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeActU4AsymKernel);
        set_compile_parameter(quantize.kernel, "qwen3.qact_u4asym.shape_k", match.input_size);
        set_compile_parameter(quantize.kernel, "qwen3.qact_u4asym.shape_cols", match.token_count);
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ qact, 0, qact_bytes });
        quantize.bindings.push_back({ scales, 0, scale_bytes });
        quantize.bindings.push_back({ metadata, 0, meta_bytes });

        Dispatch gate_up;
        gate_up.kernel = make_kernel_specialization(kDenseQ4KU4AsymDualGridLowRowKernel);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_u4asym.shape_k", match.input_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_u4asym.shape_rows", match.output_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_u4asym.shape_cols", match.token_count);
        gate_up.bindings.push_back(q4_k_i4_k32_weight_binding(*match.gate_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back(q4_k_i4_k32_weight_binding(*match.up_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back({ match.gate_output->id, 0, match.gate_output->byte_count });
        gate_up.bindings.push_back({ match.up_output->id, 0, match.up_output->byte_count });
        gate_up.bindings.push_back({ qact, 0, qact_bytes });
        gate_up.bindings.push_back({ scales, 0, scale_bytes });
        gate_up.bindings.push_back({ metadata, 0, meta_bytes });

        dispatch_match.dispatches.push_back(std::move(quantize));
        dispatch_match.dispatches.push_back(std::move(gate_up));
        return true;
    }
    if (match.token_count < 128 || match.token_count % 128 != 0) {
        return false;
    }
    const QwenDenseDownMatch q4_down =
        q4_gate_up ? match_qwen_dense_down(context.graph, match, GGML_TYPE_Q4_K) : QwenDenseDownMatch{};
    if (q4_down.matched()) {
        if (!append_covered_node(context, match.gate_node, dispatch_match) ||
            !append_covered_node(context, match.up_node, dispatch_match) ||
            !append_covered_node(context, match.glu_node, dispatch_match) ||
            !append_covered_node(context, q4_down.node, dispatch_match)) {
            return false;
        }

        const size_t  input_elements   = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size);
        const size_t  i4_payload_bytes = input_elements / 2;
        const size_t  i4_metadata_bytes = input_elements / 8;
        const size_t  output_elements = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.output_size);
        const size_t  u4_payload_bytes = output_elements / 2;
        const size_t  u4_scale_bytes   = output_elements / 8;
        const size_t  u4_meta_bytes    = output_elements / 4;
        const ValueId i4_payload(context.next_plan_value.value);
        const ValueId i4_scales(context.next_plan_value.value + 1);
        const ValueId i4_sums(context.next_plan_value.value + 2);
        const ValueId u4_payload(context.next_plan_value.value + 3);
        const ValueId u4_scales(context.next_plan_value.value + 4);
        const ValueId u4_metadata(context.next_plan_value.value + 5);
        dispatch_match.transients.push_back({ i4_payload, "qwen.prefill.i4_input", i4_payload_bytes, 256 });
        dispatch_match.transients.push_back({ i4_scales, "qwen.prefill.i4_scales", i4_metadata_bytes, 256 });
        dispatch_match.transients.push_back({ i4_sums, "qwen.prefill.i4_sums", i4_metadata_bytes, 256 });
        dispatch_match.transients.push_back({ u4_payload, "qwen.prefill.u4_swiglu_qs", u4_payload_bytes, 256 });
        dispatch_match.transients.push_back({ u4_scales, "qwen.prefill.u4_swiglu_ds", u4_scale_bytes, 256 });
        dispatch_match.transients.push_back({ u4_metadata, "qwen.prefill.u4_swiglu_meta", u4_meta_bytes, 256 });

        Dispatch quantize_i4;
        quantize_i4.kernel = make_kernel_specialization(kQuantizeActI4Kernel);
        set_compile_parameter(quantize_i4.kernel, "qwen3.qact_i4.shape_k", match.input_size);
        set_compile_parameter(quantize_i4.kernel, "qwen3.qact_i4.shape_cols", match.token_count);
        quantize_i4.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize_i4.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
        quantize_i4.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
        quantize_i4.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });

        Dispatch gate_up;
        gate_up.kernel = make_kernel_specialization(kDenseSymmetricI4DualGateUpSwiGluU4Kernel);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_k", match.input_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_rows", match.output_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_cols", match.token_count);
        gate_up.bindings.push_back(
            symmetric_i4_prefill_weight_binding(*match.gate_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back(
            symmetric_i4_prefill_weight_binding(*match.up_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        gate_up.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        gate_up.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
        gate_up.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
        gate_up.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });
        gate_up.bindings.push_back({ u4_payload, 0, u4_payload_bytes });
        gate_up.bindings.push_back({ u4_scales, 0, u4_scale_bytes });
        gate_up.bindings.push_back({ u4_metadata, 0, u4_meta_bytes });

        Dispatch down;
        down.kernel = make_kernel_specialization(kDenseQ4KU4AsymKernel);
        set_compile_parameter(down.kernel, "qwen3.dense_q4_u4asym.shape_k", match.output_size);
        set_compile_parameter(down.kernel, "qwen3.dense_q4_u4asym.shape_rows", q4_down.output_size);
        set_compile_parameter(down.kernel, "qwen3.dense_q4_u4asym.shape_cols", match.token_count);
        down.bindings.push_back(q4_k_i4_k32_weight_binding(*q4_down.weight, match.output_size, q4_down.output_size));
        down.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        down.bindings.push_back({ q4_down.output->id, 0, q4_down.output->byte_count });
        down.bindings.push_back({ u4_payload, 0, u4_payload_bytes });
        down.bindings.push_back({ u4_scales, 0, u4_scale_bytes });
        down.bindings.push_back({ u4_metadata, 0, u4_meta_bytes });

        dispatch_match.dispatches.push_back(std::move(quantize_i4));
        dispatch_match.dispatches.push_back(std::move(gate_up));
        dispatch_match.dispatches.push_back(std::move(down));
        return true;
    }
    const QwenDenseDownMatch q6_down =
        q4_gate_up ? match_qwen_dense_down(context.graph, match, GGML_TYPE_Q6_K) : QwenDenseDownMatch{};
    if (q6_down.matched()) {
        if (!append_covered_node(context, match.gate_node, dispatch_match) ||
            !append_covered_node(context, match.up_node, dispatch_match) ||
            !append_covered_node(context, match.glu_node, dispatch_match)) {
            return false;
        }

        const size_t input_elements    = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size);
        const size_t i4_payload_bytes  = input_elements / 2;
        const size_t i4_metadata_bytes = input_elements / 8;
        const size_t f16_output_bytes =
            static_cast<size_t>(match.token_count) * static_cast<size_t>(match.output_size) * sizeof(uint16_t);
        const ValueId i4_payload(context.next_plan_value.value);
        const ValueId i4_scales(context.next_plan_value.value + 1);
        const ValueId i4_sums(context.next_plan_value.value + 2);
        const ValueId f16_output(context.next_plan_value.value + 3);
        dispatch_match.transients.push_back({ i4_payload, "qwen.prefill.i4_input", i4_payload_bytes, 256 });
        dispatch_match.transients.push_back({ i4_scales, "qwen.prefill.i4_scales", i4_metadata_bytes, 256 });
        dispatch_match.transients.push_back({ i4_sums, "qwen.prefill.i4_sums", i4_metadata_bytes, 256 });
        dispatch_match.transients.push_back({ f16_output, "qwen.prefill.f16_swiglu", f16_output_bytes, 256 });

        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, "qwen.prefill.f16_swiglu" },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }

        Dispatch quantize_i4;
        quantize_i4.kernel = make_kernel_specialization(kQuantizeActI4Kernel);
        set_compile_parameter(quantize_i4.kernel, "qwen3.qact_i4.shape_k", match.input_size);
        set_compile_parameter(quantize_i4.kernel, "qwen3.qact_i4.shape_cols", match.token_count);
        quantize_i4.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize_i4.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
        quantize_i4.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
        quantize_i4.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });

        Dispatch gate_up;
        gate_up.kernel = make_kernel_specialization(kDenseSymmetricI4DualGateUpSwiGluF16Kernel);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_k", match.input_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_rows", match.output_size);
        set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_cols", match.token_count);
        gate_up.bindings.push_back(
            symmetric_i4_prefill_weight_binding(*match.gate_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back(
            symmetric_i4_prefill_weight_binding(*match.up_weight, match.input_size, match.output_size));
        gate_up.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        gate_up.bindings.push_back({ f16_output, 0, f16_output_bytes });
        gate_up.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
        gate_up.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
        gate_up.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });

        dispatch_match.dispatches.push_back(std::move(quantize_i4));
        dispatch_match.dispatches.push_back(std::move(gate_up));
        return true;
    }
    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }

    const size_t i4_payload_bytes = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size) / 2;
    const size_t i4_metadata_bytes =
        static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size / 32) * sizeof(uint32_t);
    const size_t  q8_output_bytes = q8_1_x4_byte_count(*match.output);
    const ValueId i4_payload      = context.next_plan_value;
    const ValueId i4_scales(context.next_plan_value.value + 1);
    const ValueId i4_sums(context.next_plan_value.value + 2);
    const ValueId q8_output(context.next_plan_value.value + 3);
    dispatch_match.transients.push_back({ i4_payload, "qwen.prefill.i4_input", i4_payload_bytes, 256 });
    dispatch_match.transients.push_back({ i4_scales, "qwen.prefill.i4_scales", i4_metadata_bytes, 256 });
    dispatch_match.transients.push_back({ i4_sums, "qwen.prefill.i4_sums", i4_metadata_bytes, 256 });
    dispatch_match.transients.push_back({ q8_output, "qwen.prefill.q8_swiglu", q8_output_bytes, 256 });

    Dispatch quantize_i4;
    quantize_i4.kernel = make_kernel_specialization(kQuantizeActI4Kernel);
    set_compile_parameter(quantize_i4.kernel, "qwen3.qact_i4.shape_k", match.input_size);
    set_compile_parameter(quantize_i4.kernel, "qwen3.qact_i4.shape_cols", match.token_count);
    quantize_i4.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize_i4.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
    quantize_i4.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
    quantize_i4.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });

    Dispatch gate_up;
    gate_up.kernel = make_kernel_specialization(kDenseSymmetricI4DualGateUpSwiGluF32Kernel);
    set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_k", match.input_size);
    set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_rows", match.output_size);
    set_compile_parameter(gate_up.kernel, "qwen3.dense_q4_i4.dual_m32n32.shape_cols", match.token_count);
    gate_up.bindings.push_back(
        symmetric_i4_prefill_weight_binding(*match.gate_weight, match.input_size, match.output_size));
    gate_up.bindings.push_back(
        symmetric_i4_prefill_weight_binding(*match.up_weight, match.input_size, match.output_size));
    gate_up.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    gate_up.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    gate_up.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
    gate_up.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
    gate_up.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });

    Dispatch quantize_q8;
    quantize_q8.kernel = make_kernel_specialization(kQuantizeQ8_1X4Kernel);
    quantize_q8.kernel.integer_parameters.emplace("token_count", match.token_count);
    quantize_q8.kernel.integer_parameters.emplace("input_size", match.output_size);
    set_compile_parameter(quantize_q8.kernel, "ggml.quantize_q8_1_x4.group_capacity",
                          match.token_count * ((match.output_size + 127) / 128));
    quantize_q8.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    quantize_q8.bindings.push_back({ q8_output, 0, q8_output_bytes });

    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes, "qwen.prefill.q8_swiglu" },
            metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(quantize_i4));
    dispatch_match.dispatches.push_back(std::move(gate_up));
    dispatch_match.dispatches.push_back(std::move(quantize_q8));
    return true;
}

static bool match_qwen_hybrid_gdn_projection_pair_dispatch(const DispatchMatchContext & context,
                                                           DispatchMatch &              dispatch_match) {
    const QwenHybridGdnProjectionPairMatch match = match_qwen_hybrid_gdn_projection_pair(context);
    if (!match.matched() || !append_covered_node(context, match.alpha_node, dispatch_match) ||
        !append_covered_node(context, match.beta_node, dispatch_match)) {
        return false;
    }

    const LlmSymmetricI4ActivationLayout activation_layout =
        llm_symmetric_i4_activation_layout(match.input_size, match.token_count);
    Dispatch projections;
    projections.kernel = make_kernel_specialization(kDenseSymmetricI4AdjacentDualGridLowRowKernel);
    set_compile_parameter(projections.kernel, "qwen3.dense_q4_i4.lowrow.shape_k", match.input_size);
    set_compile_parameter(projections.kernel, "qwen3.dense_q4_i4.lowrow.shape_rows", match.output_size);
    set_compile_parameter(projections.kernel, "qwen3.dense_q4_i4.lowrow.shape_cols", match.token_count);
    projections.bindings.push_back(
        symmetric_i4_weight_binding(*match.alpha_weight, match.input_size, match.output_size));
    projections.bindings.push_back(
        symmetric_i4_weight_binding(*match.beta_weight, match.input_size, match.output_size));
    projections.bindings.push_back({ match.alpha_output->id, 0, match.alpha_output->byte_count });
    projections.bindings.push_back({ match.beta_output->id, 0, match.beta_output->byte_count });
    projections.bindings.push_back({ match.activation, 0, activation_layout.payload_bytes });
    projections.bindings.push_back(
        { match.activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    projections.bindings.push_back(
        { match.activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
    dispatch_match.dispatches.push_back(std::move(projections));
    return true;
}

static bool match_qwen_concat_dim0_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2) {
        return false;
    }
    const Value * lhs    = graph_value(context.graph, node->inputs[0]);
    const Value * rhs    = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (!is_f32(lhs) || !is_f32(rhs) || !is_f32(output) || !lhs->contiguous || !rhs->contiguous ||
        !output->contiguous || lhs->ne[0] <= 0 || lhs->ne[0] > 65536 || rhs->ne[0] <= 0 || rhs->ne[0] > 65536 ||
        output->ne[0] != lhs->ne[0] + rhs->ne[0] || output->element_count <= 0 || output->element_count > 1073741824 ||
        !distinct_storage(*lhs, *rhs) || !distinct_storage(*lhs, *output) || !distinct_storage(*rhs, *output)) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs->ne[dim] != rhs->ne[dim] || lhs->ne[dim] != output->ne[dim]) {
            return false;
        }
    }

    const int64_t row_count = output->element_count / output->ne[0];
    if (row_count < 1 || row_count > 1048576) {
        return false;
    }
    Dispatch concat;
    concat.kernel = make_kernel_specialization(kQwenConcatDim0F32Kernel);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_dim0_lhs_width", lhs->ne[0]);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_dim0_rhs_width", rhs->ne[0]);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_dim0_row_count", row_count);
    set_compile_parameter(concat.kernel, "hrx2_tuning_concat_dim0_workgroup_size", 256);
    concat.bindings.push_back({ lhs->id, 0, lhs->byte_count });
    concat.bindings.push_back({ rhs->id, 0, rhs->byte_count });
    concat.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(concat));
    return true;
}

static bool match_qwen_hybrid_ssm_prefill_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match) {
    const QwenHybridSsmPrefillMatch match = match_qwen_hybrid_ssm_prefill(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    if (!append_covered_node(context, match.concat, dispatch_match) ||
        !append_covered_node(context, match.ssm, dispatch_match) ||
        !append_covered_node(context, match.silu, dispatch_match)) {
        return false;
    }
    for (const QwenHybridSsmCacheUpdate & update : match.cache_updates) {
        if (!append_covered_node(context, update.state_tail_view, dispatch_match) ||
            !append_covered_node(context, update.cache_view, dispatch_match) ||
            !append_covered_node(context, update.cache_copy, dispatch_match)) {
            return false;
        }
    }

    if (match.token_count == 1 && match.cache_updates.size() == 1) {
        Dispatch ssm;
        ssm.kernel = make_kernel_specialization(kQwenSsmConvF32DecodeKernel);
        set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_decode_d_inner", match.hidden_size);
        set_compile_parameter(ssm.kernel, "hrx2_tuning_ssm_decode_workgroup_size", 256);
        ssm.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        ssm.bindings.push_back({ match.x->id, 0, match.x->byte_count });
        ssm.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
        ssm.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        ssm.bindings.push_back(
            { match.cache_updates.front().cache->id, 0, match.cache_updates.front().cache->byte_count });
        dispatch_match.dispatches.push_back(std::move(ssm));
        return true;
    }

    if (match.token_count != 512 || match.cache_updates.size() != 1) {
        Dispatch ssm;
        ssm.kernel = make_kernel_specialization(kQwenSsmConvF32RollbackKernel);
        set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_rollback_d_inner", match.hidden_size);
        set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_rollback_n_t", match.token_count);
        set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_rollback_cache_count", match.cache_updates.size());
        set_compile_parameter(ssm.kernel, "hrx2_tuning_ssm_rollback_workgroup_size", 256);
        ssm.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        ssm.bindings.push_back({ match.x->id, 0, match.x->byte_count });
        ssm.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
        ssm.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        for (size_t slot = 0; slot < 5; ++slot) {
            const Value * cache = match.cache_updates[std::min(slot, match.cache_updates.size() - 1)].cache;
            ssm.bindings.push_back({ cache->id, 0, cache->byte_count });
        }
        dispatch_match.dispatches.push_back(std::move(ssm));
        return true;
    }

    const size_t  snapshot_bytes = static_cast<size_t>(64 * match.hidden_size) * sizeof(float);
    const ValueId snapshot       = context.next_plan_value;
    dispatch_match.transients.push_back({ snapshot, "qwen_hybrid_ssm_window", snapshot_bytes, 256 });

    Dispatch concat;
    concat.kernel = make_kernel_specialization(kQwenConcatWindowTailF32Kernel);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_d_conv", 4);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_d_inner", match.hidden_size);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_n_t", match.token_count);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_n_s", 1);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_state_row_stride", 1);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_state_channel_stride", 3);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_x_row_stride", match.hidden_size);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_dst_row_stride", match.hidden_size);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_cache_row_stride", 1);
    set_compile_parameter(concat.kernel, "hrx2_shape_concat_tail_cache_channel_stride", 3);
    set_compile_parameter(concat.kernel, "hrx2_tuning_concat_tail_workgroup_size", 256);
    concat.bindings.push_back({ match.state->id, 0, match.state->byte_count });
    concat.bindings.push_back({ match.x->id, 0, match.x->byte_count });
    concat.bindings.push_back({ snapshot, 0, snapshot_bytes });
    concat.bindings.push_back(
        { match.cache_updates.front().cache->id, 0, match.cache_updates.front().cache->byte_count });

    Dispatch ssm;
    ssm.kernel = make_kernel_specialization(kQwenSsmConvF32PrefillKernel);
    set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_conv_d_conv", 4);
    set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_conv_d_inner", match.hidden_size);
    set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_conv_n_t", match.token_count);
    set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_conv_n_s", 1);
    set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_conv_state_row_stride", match.hidden_size);
    set_compile_parameter(ssm.kernel, "hrx2_shape_ssm_conv_x_row_stride", match.hidden_size);
    ssm.bindings.push_back({ snapshot, 0, snapshot_bytes });
    ssm.bindings.push_back({ match.x->id, 0, match.x->byte_count });
    ssm.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
    ssm.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.dispatches.push_back(std::move(concat));
    dispatch_match.dispatches.push_back(std::move(ssm));
    return true;
}

static bool match_qwen_hybrid_gdn_prefill_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match) {
    const QwenHybridGdnPrefillMatch match = match_qwen_hybrid_gdn_prefill(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    for (const GraphNode * node : match.covered) {
        if (!append_covered_node(context, node, dispatch_match)) {
            return false;
        }
    }

    const size_t  rms_scales_bytes = static_cast<size_t>(match.head_count * match.token_count) * sizeof(float);
    const ValueId rms_scales       = context.next_plan_value;
    dispatch_match.transients.push_back({ rms_scales, "qwen_hybrid_gdn_rms_scales", rms_scales_bytes, 256 });

    Dispatch epilogue;
    epilogue.kernel = make_kernel_specialization(kQwenGdnProjectionEpilogueF32Kernel);
    set_compile_parameter(epilogue.kernel, "hrx2_gdn_epilogue_head_count", match.head_count);
    set_compile_parameter(epilogue.kernel, "hrx2_gdn_epilogue_element_count", match.head_count * match.token_count);
    set_compile_parameter(epilogue.kernel, "hrx2_gdn_epilogue_workgroup_size", 256);
    epilogue.bindings.push_back({ match.alpha_raw->id, 0, match.alpha_raw->byte_count });
    epilogue.bindings.push_back({ match.beta_raw->id, 0, match.beta_raw->byte_count });
    epilogue.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
    epilogue.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
    epilogue.bindings.push_back({ match.gate_flat->id, 0, match.gate_flat->byte_count });
    epilogue.bindings.push_back({ match.beta->id, 0, match.beta->byte_count });
    dispatch_match.dispatches.push_back(std::move(epilogue));

    if (match.snapshot_count == 1) {
        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(kQwenGdnF32PrefillKernel);
        configure_qwen_gdn_kernel(gdn, match, match.token_count);
        gdn.bindings.push_back({ match.raw_q->id, 0, match.raw_q->byte_count });
        gdn.bindings.push_back({ match.raw_k->id, 0, match.raw_k->byte_count });
        gdn.bindings.push_back({ match.v->id, 0, match.v->byte_count });
        gdn.bindings.push_back({ match.gate->id, 0, match.gate->byte_count });
        gdn.bindings.push_back({ match.beta->id, 0, match.beta->byte_count });
        gdn.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        gdn.bindings.push_back({ match.gdn_output->id, 0, match.gdn_output->byte_count });
        gdn.bindings.push_back({ rms_scales, 0, rms_scales_bytes });

        Dispatch cache_copy;
        cache_copy.kernel = make_kernel_specialization(kCopyF32Kernel);
        set_compile_parameter(cache_copy.kernel, "shape_copy_n", match.new_state->element_count);
        set_compile_parameter(cache_copy.kernel, "tuning_copy_workgroup_size", 256);
        cache_copy.bindings.push_back({ match.new_state->id, 0, match.new_state->byte_count });
        cache_copy.bindings.push_back({ match.cache->id, 0, match.cache->byte_count });

        dispatch_match.dispatches.push_back(std::move(gdn));
        dispatch_match.dispatches.push_back(std::move(cache_copy));
        return true;
    }

    const int64_t written_snapshot_count = std::min(match.token_count, match.snapshot_count);
    const int64_t prefix_token_count     = match.token_count - written_snapshot_count;
    const size_t  q_token_bytes          = static_cast<size_t>(match.width * match.q_head_count) * sizeof(float);
    const size_t  v_token_bytes          = static_cast<size_t>(match.width * match.head_count) * sizeof(float);
    const size_t  gate_token_bytes       = static_cast<size_t>(match.head_count) * sizeof(float);
    const size_t  attention_token_bytes  = v_token_bytes;
    const size_t  state_bytes  = static_cast<size_t>(match.width * match.width * match.head_count) * sizeof(float);
    const size_t  cache_stride = match.cache->nb[2];

    if (prefix_token_count == 0 && written_snapshot_count == match.token_count && match.token_count >= 2 &&
        match.token_count <= 5 && cache_stride % sizeof(float) == 0) {
        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(kQwenGdnF32SnapshotRollbackKernel);
        configure_qwen_gdn_kernel(gdn, match, match.token_count);
        set_compile_parameter(gdn.kernel, "hrx2_shape_gdn_snapshot_stride", cache_stride / sizeof(float));
        gdn.bindings.push_back({ match.raw_q->id, 0, match.raw_q->byte_count });
        gdn.bindings.push_back({ match.raw_k->id, 0, match.raw_k->byte_count });
        gdn.bindings.push_back({ match.v->id, 0, match.v->byte_count });
        gdn.bindings.push_back({ match.gate->id, 0, match.gate->byte_count });
        gdn.bindings.push_back({ match.beta->id, 0, match.beta->byte_count });
        gdn.bindings.push_back({ match.state->id, 0, state_bytes });
        gdn.bindings.push_back(
            { match.cache->id, 0, state_bytes + static_cast<size_t>(written_snapshot_count - 1) * cache_stride });
        gdn.bindings.push_back(
            { match.gdn_output->id, 0, static_cast<size_t>(match.token_count) * attention_token_bytes });
        gdn.bindings.push_back({ rms_scales, 0, rms_scales_bytes });
        dispatch_match.dispatches.push_back(std::move(gdn));
        return true;
    }

    auto append_state_copy = [&](ValueId source, size_t source_offset, ValueId target, size_t target_offset) {
        Dispatch copy;
        copy.kernel = make_kernel_specialization(kCopyF32Kernel);
        set_compile_parameter(copy.kernel, "shape_copy_n", state_bytes / sizeof(float));
        set_compile_parameter(copy.kernel, "tuning_copy_workgroup_size", 256);
        copy.bindings.push_back({ source, source_offset, state_bytes });
        copy.bindings.push_back({ target, target_offset, state_bytes });
        dispatch_match.dispatches.push_back(std::move(copy));
    };

    if (prefix_token_count > 0) {
        const size_t q_span    = static_cast<size_t>(prefix_token_count - 1) * match.raw_q->nb[2] + q_token_bytes;
        const size_t k_span    = static_cast<size_t>(prefix_token_count - 1) * match.raw_k->nb[2] + q_token_bytes;
        const size_t v_span    = static_cast<size_t>(prefix_token_count - 1) * match.v->nb[2] + v_token_bytes;
        const size_t gate_span = static_cast<size_t>(prefix_token_count - 1) * match.gate->nb[2] + gate_token_bytes;
        const size_t beta_span = static_cast<size_t>(prefix_token_count - 1) * match.beta->nb[2] + gate_token_bytes;
        const size_t prefix_attention_bytes = static_cast<size_t>(prefix_token_count) * attention_token_bytes;
        const size_t prefix_rms_bytes       = static_cast<size_t>(prefix_token_count) * gate_token_bytes;

        Dispatch prefix;
        prefix.kernel = make_kernel_specialization(kQwenGdnF32PrefillKernel);
        configure_qwen_gdn_kernel(prefix, match, prefix_token_count);
        prefix.bindings.push_back({ match.raw_q->id, 0, q_span });
        prefix.bindings.push_back({ match.raw_k->id, 0, k_span });
        prefix.bindings.push_back({ match.v->id, 0, v_span });
        prefix.bindings.push_back({ match.gate->id, 0, gate_span });
        prefix.bindings.push_back({ match.beta->id, 0, beta_span });
        prefix.bindings.push_back({ match.state->id, 0, state_bytes });
        prefix.bindings.push_back({ match.gdn_output->id, 0, prefix_attention_bytes + state_bytes });
        prefix.bindings.push_back({ rms_scales, 0, prefix_rms_bytes });
        dispatch_match.dispatches.push_back(std::move(prefix));
        append_state_copy(match.gdn_output->id, prefix_attention_bytes, match.cache->id,
                          static_cast<size_t>(written_snapshot_count - 1) * cache_stride);
    } else {
        append_state_copy(match.state->id, 0, match.cache->id,
                          static_cast<size_t>(written_snapshot_count - 1) * cache_stride);
    }

    for (int64_t snapshot = 0; snapshot < written_snapshot_count; ++snapshot) {
        const int64_t token = prefix_token_count + snapshot;
        const int64_t slot  = written_snapshot_count - 1 - snapshot;
        if (snapshot > 0) {
            append_state_copy(match.cache->id, static_cast<size_t>(slot + 1) * cache_stride, match.cache->id,
                              static_cast<size_t>(slot) * cache_stride);
        }

        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(kQwenGdnF32StateCacheKernel);
        configure_qwen_gdn_kernel(gdn, match, 1);
        gdn.bindings.push_back({ match.raw_q->id, static_cast<size_t>(token) * match.raw_q->nb[2], q_token_bytes });
        gdn.bindings.push_back({ match.raw_k->id, static_cast<size_t>(token) * match.raw_k->nb[2], q_token_bytes });
        gdn.bindings.push_back({ match.v->id, static_cast<size_t>(token) * match.v->nb[2], v_token_bytes });
        gdn.bindings.push_back({ match.gate->id, static_cast<size_t>(token) * match.gate->nb[2], gate_token_bytes });
        gdn.bindings.push_back({ match.beta->id, static_cast<size_t>(token) * match.beta->nb[2], gate_token_bytes });
        gdn.bindings.push_back({ match.cache->id, static_cast<size_t>(slot) * cache_stride, state_bytes });
        gdn.bindings.push_back(
            { match.gdn_output->id, static_cast<size_t>(token) * attention_token_bytes, attention_token_bytes });
        gdn.bindings.push_back({ rms_scales, static_cast<size_t>(token) * gate_token_bytes, gate_token_bytes });
        dispatch_match.dispatches.push_back(std::move(gdn));
    }

    return true;
}

static bool match_qwen_hybrid_recurrent_rms_dispatch(const DispatchMatchContext & context,
                                                     DispatchMatch &              dispatch_match) {
    const QwenHybridRecurrentRmsMatch match = match_qwen_hybrid_recurrent_rms(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }
    for (const GraphNode * node : match.covered) {
        if (!append_covered_node(context, node, dispatch_match)) {
            return false;
        }
    }

    const bool publish_i4 =
        match.column_count % 64 == 0 && has_low_row_symmetric_i4_consumer(context.graph, *match.output);
    const bool publish_q8 =
        !publish_i4 && match.f16_target != nullptr && has_prefill_q5_dense_consumer(context.graph, *match.f16_target);
    const bool    publish_f16 = match.f16_target != nullptr && !publish_q8 && !publish_i4;
    const size_t  f16_bytes   = publish_f16 ? match.f16_target->byte_count / 2 : 0;
    const ValueId f16_value   = publish_f16 ? context.next_plan_value : ValueId{};
    const size_t  q8_bytes    = publish_q8 ? q8_1_x4_byte_count(*match.f16_target) : 0;
    const ValueId q8_value    = publish_q8 ? ValueId(context.next_plan_value.value + (publish_f16 ? 1 : 0)) : ValueId{};
    const LlmSymmetricI4ActivationLayout i4_layout =
        llm_symmetric_i4_activation_layout(match.column_count, match.row_count);
    const ValueId i4_value = publish_i4 ? context.next_plan_value : ValueId{};

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(publish_i4  ? kQwenRecurrentRmsGateF32I4Kernel :
                                                 publish_f16 ? kQwenRecurrentRmsGateF32F16Kernel :
                                                               kQwenRecurrentRmsGateF32Kernel);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    set_compile_parameter(dispatch.kernel, "hrx2_shape_rms_norm_mul_ncols", match.column_count);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_rms_norm_mul_nrows", match.row_count);
    set_compile_parameter(dispatch.kernel, "hrx2_tuning_rms_norm_mul_workgroup_size", 256);
    set_compile_parameter(dispatch.kernel, "hrx2_tuning_rms_norm_mul_vector_width", 4);
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.raw_gate->id, 0, match.raw_gate->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    if (publish_f16) {
        dispatch.bindings.push_back({ f16_value, 0, f16_bytes });
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.f16_target->id, f16_value, GGML_TYPE_F16, f16_bytes, "qwen.prefill.f16_recurrent_gate" },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
        dispatch_match.transients.push_back({ f16_value, "qwen.prefill.f16_recurrent_gate", f16_bytes, 256 });
    }
    if (publish_i4) {
        dispatch.bindings.push_back({ i4_value, 0, i4_layout.payload_bytes });
        dispatch.bindings.push_back({ i4_value, i4_layout.scales_offset, i4_layout.metadata_bytes });
        dispatch.bindings.push_back({ i4_value, i4_layout.sums_offset, i4_layout.metadata_bytes });
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.output->id, i4_value, GGML_TYPE_COUNT, i4_layout.total_bytes,
                  kLlmSymmetricI4ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
        dispatch_match.transients.push_back(
            { i4_value, kLlmSymmetricI4ActivationAlternateName, i4_layout.total_bytes, 256 });
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    if (publish_q8) {
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.f16_target->id, q8_value, GGML_TYPE_Q8_1, q8_bytes, "qwen.prefill.q8_recurrent_gate" },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
        dispatch_match.transients.push_back({ q8_value, "qwen.prefill.q8_recurrent_gate", q8_bytes, 256 });

        const int64_t q8_hidden_size = match.f16_target->ne[0];
        const int64_t q8_token_count = match.f16_target->element_count / q8_hidden_size;
        Dispatch      quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeQ8_1X4Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", q8_token_count);
        quantize.kernel.integer_parameters.emplace("input_size", q8_hidden_size);
        set_compile_parameter(quantize.kernel, "ggml.quantize_q8_1_x4.group_capacity",
                              q8_token_count * ((q8_hidden_size + 127) / 128));
        quantize.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        quantize.bindings.push_back({ q8_value, 0, q8_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));
    }
    return true;
}

static bool match_qwen_swiglu_split_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node   = context.root_node;
    const GluParams * params = node != nullptr ? op_params_as<GluParams>(node->params) : nullptr;
    if (node == nullptr || node->op != GGML_OP_GLU || node->inputs.size() != 2 || params == nullptr ||
        params->op != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    const Value * gate   = graph_value(context.graph, node->inputs[0]);
    const Value * up     = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (!is_f32(gate) || !is_f32(up) || !is_f32(output) || !gate->contiguous || !up->contiguous ||
        !output->contiguous || !same_shape(*gate, *up) || !same_shape(*gate, *output) || gate->element_count <= 0 ||
        gate->element_count > 1073741824 || !distinct_storage(*gate, *up) || !distinct_storage(*gate, *output) ||
        !distinct_storage(*up, *output)) {
        return false;
    }

    const bool publish_i4 =
        output->element_count % 64 == 0 && has_low_row_symmetric_i4_consumer(context.graph, *output);
    const bool has_prefill_dense_consumers = has_prefill_f16_dense_consumers_only(context.graph, *output);
    const bool publish_q8 =
        !publish_i4 && has_prefill_dense_consumers && has_prefill_q5_dense_consumer(context.graph, *output);
    const bool    publish_f16 = !publish_i4 && has_prefill_dense_consumers && !publish_q8;
    const size_t  f16_bytes   = publish_f16 ? output->byte_count / 2 : 0;
    const ValueId f16_value   = publish_f16 ? context.next_plan_value : ValueId{};
    const size_t  q8_bytes    = publish_q8 ? q8_1_x4_byte_count(*output) : 0;
    const ValueId q8_value    = publish_q8 ? ValueId(context.next_plan_value.value + (publish_f16 ? 1 : 0)) : ValueId{};
    const int64_t i4_input_size                    = publish_i4 ? output->ne[0] : 64;
    const int64_t i4_token_count                   = publish_i4 ? output->element_count / i4_input_size : 1;
    const LlmSymmetricI4ActivationLayout i4_layout = llm_symmetric_i4_activation_layout(i4_input_size, i4_token_count);
    const ValueId                        i4_value  = publish_i4 ? context.next_plan_value : ValueId{};

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(publish_i4  ? kQwenSwiGluSplitF32I4ParallelKernel :
                                                 publish_f16 ? kQwenSwiGluSplitF32F16Kernel :
                                                               kQwenSwiGluSplitF32Kernel);
    set_compile_parameter(dispatch.kernel, "hrx2_shape_swiglu_element_count", gate->element_count);
    set_compile_parameter(dispatch.kernel, "hrx2_tuning_swiglu_workgroup_size", 256);
    dispatch.bindings.push_back({ gate->id, 0, gate->byte_count });
    dispatch.bindings.push_back({ up->id, 0, up->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    if (publish_f16) {
        dispatch.bindings.push_back({ f16_value, 0, f16_bytes });
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { output->id, f16_value, GGML_TYPE_F16, f16_bytes, "qwen.prefill.f16_swiglu" }, metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
        dispatch_match.transients.push_back({ f16_value, "qwen.prefill.f16_swiglu", f16_bytes, 256 });
    }
    if (publish_i4) {
        dispatch.bindings.push_back({ i4_value, 0, i4_layout.payload_bytes });
        dispatch.bindings.push_back({ i4_value, i4_layout.scales_offset, i4_layout.metadata_bytes });
        dispatch.bindings.push_back({ i4_value, i4_layout.sums_offset, i4_layout.metadata_bytes });
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { output->id, i4_value, GGML_TYPE_COUNT, i4_layout.total_bytes,
                  kLlmSymmetricI4ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
        dispatch_match.transients.push_back(
            { i4_value, kLlmSymmetricI4ActivationAlternateName, i4_layout.total_bytes, 256 });
    }
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    if (publish_q8) {
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { output->id, q8_value, GGML_TYPE_Q8_1, q8_bytes, "qwen.prefill.q8_swiglu" }, metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
        dispatch_match.transients.push_back({ q8_value, "qwen.prefill.q8_swiglu", q8_bytes, 256 });

        const int64_t token_count = output->element_count / output->ne[0];
        Dispatch      quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeQ8_1X4Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", token_count);
        quantize.kernel.integer_parameters.emplace("input_size", output->ne[0]);
        set_compile_parameter(quantize.kernel, "ggml.quantize_q8_1_x4.group_capacity",
                              token_count * ((output->ne[0] + 127) / 128));
        quantize.bindings.push_back({ output->id, 0, output->byte_count });
        quantize.bindings.push_back({ q8_value, 0, q8_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));
    }
    return true;
}

void register_qwen_hybrid_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.hybrid.gdn_projection_pair_decode",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        1400,
        DispatchSource::Qwen,
        match_qwen_hybrid_gdn_projection_pair_dispatch,
    });
    registry.add({
        "qwen.dense.symmetric_i4_gate_up_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        1300,
        DispatchSource::Qwen,
        match_qwen_dense_mixed_gate_up_dispatch,
    });
    registry.add({
        "qwen.hybrid.ssm_prefill",
        GGML_OP_CONCAT,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Qwen,
        match_qwen_hybrid_ssm_prefill_dispatch,
    });
    registry.add({
        "qwen.concat_dim0_f32",
        GGML_OP_CONCAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Qwen,
        match_qwen_concat_dim0_f32_dispatch,
    });
    registry.add({
        "qwen.hybrid.gdn_prefill",
        GGML_OP_L2_NORM,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Qwen,
        match_qwen_hybrid_gdn_prefill_dispatch,
    });
    registry.add({
        "qwen.hybrid.recurrent_rms_gate_prefill",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1200,
        DispatchSource::Qwen,
        match_qwen_hybrid_recurrent_rms_dispatch,
    });
    registry.add({
        "qwen.swiglu_split_f32",
        GGML_OP_GLU,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_swiglu_split_dispatch,
    });
}

}  // namespace ggml::hrx
