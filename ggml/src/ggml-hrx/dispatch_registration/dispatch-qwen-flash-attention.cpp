#include "dispatch-qwen-flash-attention.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenFlashAttentionF32F16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_flash_attention_f32_f16_wmma");
static constexpr KernelCatalogRef kQwenFlashAttentionDecodeSplitNextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8");
static constexpr KernelCatalogRef kQwenFlashAttentionDirectGateKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "hrx2_flash_attn_ext_f32_f16_direct_kv64_f32acc_gate");
static constexpr int64_t kQwenAttentionHeadSize = 128;
static constexpr int64_t kQwenQueryHeadCount    = 32;
static constexpr int64_t kQwenKeyValueHeadCount = 4;
static constexpr int64_t kQwenDecodeRowCapacity = 16;
static constexpr int64_t kQwenDecodeKvTileSize  = 64;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

static bool is_supported_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 32768;
}

static bool is_supported_decode_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_qwen_flash_decode_query_length(int64_t query_length) {
    return query_length >= 1 && query_length < kQwenDecodeRowCapacity;
}

static bool is_supported_head_count(int64_t head_count) {
    return head_count >= 1 && head_count <= 64;
}

static bool has_query_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(query_head_count * kQwenAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size);
}

static bool has_key_value_layout(const Value & value, int64_t key_value_head_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(key_value_head_count * kQwenAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size);
}

static bool has_mask_layout(const Value & value, int64_t key_value_token_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(key_value_token_count) * element_size;
}

static bool has_output_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size &&
           value.nb[2] == static_cast<size_t>(query_head_count * kQwenAttentionHeadSize) * element_size;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t attention_mask_byte_count(int64_t query_token_count, int64_t key_value_token_count) {
    if (query_token_count <= 0 || key_value_token_count <= 0) {
        return 0;
    }
    return static_cast<size_t>(query_token_count) * static_cast<size_t>(key_value_token_count) * sizeof(ggml_fp16_t);
}

static size_t q8_1_x4_byte_count(int64_t row_count, int64_t hidden_size) {
    if (row_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(row_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

static ValueId match_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match, int32_t offset) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()) + offset);
}

struct QwenFlashAttentionMatch {
    const Value *     query         = nullptr;
    const Value *     key           = nullptr;
    const Value *     value         = nullptr;
    const Value *     mask          = nullptr;
    const Value *     output        = nullptr;
    const GraphNode * output_layout = nullptr;
    ValueId           mask_binding_value;
    size_t            mask_binding_bytes    = 0;
    int64_t           query_token_count     = 0;
    int64_t           key_value_token_count = 0;
    int64_t           query_head_count      = 0;
    int64_t           key_value_head_count  = 0;

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

struct QwenDecodeSplitFlashAttentionMatch {
    const Value *     query                 = nullptr;
    const Value *     key                   = nullptr;
    const Value *     value                 = nullptr;
    const Value *     mask                  = nullptr;
    const Value *     output                = nullptr;
    const GraphNode * output_layout         = nullptr;
    int64_t           query_token_count     = 0;
    int64_t           key_value_token_count = 0;
    int64_t           key_value_capacity    = 0;
    int64_t           query_head_count      = 0;
    int64_t           key_value_head_count  = 0;

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

static bool has_qwen_flash_attention_params(const GraphNode & node) {
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    const float expected_scale = 1.0f / std::sqrt(static_cast<float>(kQwenAttentionHeadSize));
    return nearly_equal(params->scale, expected_scale) && nearly_equal(params->max_bias, 0.0f) &&
           nearly_equal(params->logit_softcap, 0.0f) &&
           (params->prec == GGML_PREC_DEFAULT || params->prec == GGML_PREC_F32);
}

static QwenFlashAttentionMatch match_qwen_flash_attention(const Graph &       graph,
                                                          const CommandPlan & plan,
                                                          const GraphNode *   node) {
    QwenFlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4 ||
        !has_qwen_flash_attention_params(*node)) {
        return match;
    }

    const Value * query  = graph_value(graph, node->inputs[0]);
    const Value * key    = graph_value(graph, node->inputs[1]);
    const Value * value  = graph_value(graph, node->inputs[2]);
    const Value * mask   = graph_value(graph, node->inputs[3]);
    const Value * output = graph_value(graph, node->output);
    if (query == nullptr || key == nullptr || value == nullptr || mask == nullptr || output == nullptr) {
        return {};
    }
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 || value->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32) {
        return {};
    }
    if (query->ne[0] != kQwenAttentionHeadSize || key->ne[0] != kQwenAttentionHeadSize ||
        value->ne[0] != kQwenAttentionHeadSize || output->ne[0] != kQwenAttentionHeadSize) {
        return {};
    }
    if (query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1 || output->ne[3] != 1 || mask->ne[2] != 1 ||
        mask->ne[3] != 1) {
        return {};
    }

    const int64_t query_token_count    = query->ne[1];
    const int64_t query_head_count     = query->ne[2];
    const int64_t key_value_capacity   = key->ne[1];
    const int64_t key_value_head_count = key->ne[2];
    if (!is_qwen_prefill_query_length(query_token_count) || key_value_capacity < query_token_count ||
        !is_supported_head_count(query_head_count) || !is_supported_head_count(key_value_head_count) ||
        query_head_count % key_value_head_count != 0) {
        return {};
    }
    if (value->ne[1] != key_value_capacity || value->ne[2] != key_value_head_count ||
        mask->ne[0] > key_value_capacity || mask->ne[1] != query_token_count || output->ne[1] != query_head_count ||
        output->ne[2] != query_token_count) {
        return {};
    }

    int64_t      key_value_token_count = key_value_capacity;
    ValueId      mask_binding_value    = mask->id;
    size_t       mask_binding_bytes    = mask->byte_count;
    const size_t compact_mask_bytes    = attention_mask_byte_count(query_token_count, query_token_count);
    const auto * compact_mask          = find_alternate_value(plan, mask->id, GGML_TYPE_F16, compact_mask_bytes);
    const bool   mask_is_compact       = mask->ne[0] == query_token_count;
    const bool   mask_is_capacity      = mask->ne[0] == key_value_capacity;
    if (compact_mask != nullptr && mask_is_capacity && mask->ne[0] > query_token_count) {
        key_value_token_count = query_token_count;
        mask_binding_value    = compact_mask->alternate_value;
        mask_binding_bytes    = compact_mask->byte_count;
    } else if (!mask_is_compact && !mask_is_capacity) {
        return {};
    }
    if (!is_supported_key_value_token_count(key_value_token_count)) {
        return {};
    }

    if (!has_query_layout(*query, query_head_count) || !has_key_value_layout(*key, key_value_head_count) ||
        !has_key_value_layout(*value, key_value_head_count) || !has_output_layout(*output, query_head_count)) {
        return {};
    }
    if (mask_binding_value == mask->id && !has_mask_layout(*mask, key_value_token_count)) {
        return {};
    }

    match.query                 = query;
    match.key                   = key;
    match.value                 = value;
    match.mask                  = mask;
    match.output                = output;
    match.output_layout         = find_single_layout_alias_consumer(graph, output->id);
    match.mask_binding_value    = mask_binding_value;
    match.mask_binding_bytes    = mask_binding_bytes;
    match.query_token_count     = query_token_count;
    match.key_value_token_count = key_value_token_count;
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    return match;
}

static QwenDecodeSplitFlashAttentionMatch match_qwen_decode_split_flash_attention(const Graph &     graph,
                                                                                  const GraphNode * node) {
    QwenDecodeSplitFlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4 ||
        !has_qwen_flash_attention_params(*node)) {
        return match;
    }

    const Value * query  = graph_value(graph, node->inputs[0]);
    const Value * key    = graph_value(graph, node->inputs[1]);
    const Value * value  = graph_value(graph, node->inputs[2]);
    const Value * mask   = graph_value(graph, node->inputs[3]);
    const Value * output = graph_value(graph, node->output);
    if (query == nullptr || key == nullptr || value == nullptr || mask == nullptr || output == nullptr) {
        return {};
    }
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 || value->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32) {
        return {};
    }
    if (query->ne[0] != kQwenAttentionHeadSize || key->ne[0] != kQwenAttentionHeadSize ||
        value->ne[0] != kQwenAttentionHeadSize || output->ne[0] != kQwenAttentionHeadSize) {
        return {};
    }
    if (query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1 || output->ne[3] != 1 || mask->ne[2] != 1 ||
        mask->ne[3] != 1) {
        return {};
    }

    const int64_t query_token_count     = query->ne[1];
    const int64_t query_head_count      = query->ne[2];
    const int64_t key_value_capacity    = key->ne[1];
    const int64_t key_value_head_count  = key->ne[2];
    const int64_t key_value_token_count = mask->ne[0];
    if (!is_qwen_flash_decode_query_length(query_token_count) ||
        !is_supported_decode_key_value_token_count(key_value_token_count) || query_head_count != kQwenQueryHeadCount ||
        key_value_head_count != kQwenKeyValueHeadCount || key_value_capacity < key_value_token_count ||
        value->ne[1] != key_value_capacity || value->ne[2] != key_value_head_count ||
        mask->ne[1] != query_token_count || output->ne[1] != query_head_count || output->ne[2] != query_token_count) {
        return {};
    }
    if (!has_query_layout(*query, query_head_count) || !has_key_value_layout(*key, key_value_head_count) ||
        !has_key_value_layout(*value, key_value_head_count) || !has_mask_layout(*mask, key_value_token_count) ||
        !has_output_layout(*output, query_head_count)) {
        return {};
    }

    match.query                 = query;
    match.key                   = key;
    match.value                 = value;
    match.mask                  = mask;
    match.output                = output;
    match.output_layout         = find_single_layout_alias_consumer(graph, output->id);
    match.query_token_count     = query_token_count;
    match.key_value_token_count = key_value_token_count;
    match.key_value_capacity    = ceil_div(key_value_token_count, kQwenDecodeKvTileSize) * kQwenDecodeKvTileSize;
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    return match;
}

}  // namespace

static const GraphNode * single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    return consumers.size() == 1 && consumers.front() != nullptr && consumers.front()->op == op ? consumers.front() :
                                                                                                  nullptr;
}

static bool match_qwen_flash_attention_direct_gate_dispatch(const DispatchMatchContext & context,
                                                            DispatchMatch &              dispatch_match) {
    const GraphNode * flash = context.root_node;
    if (flash == nullptr || flash->op != GGML_OP_FLASH_ATTN_EXT || flash->inputs.size() != 4) {
        return false;
    }
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(flash->params);
    const Value *              query  = graph_value(context.graph, flash->inputs[0]);
    const Value *              key    = graph_value(context.graph, flash->inputs[1]);
    const Value *              value  = graph_value(context.graph, flash->inputs[2]);
    const Value *              mask   = graph_value(context.graph, flash->inputs[3]);
    const Value *              fa_out = graph_value(context.graph, flash->output);
    if (params == nullptr || query == nullptr || key == nullptr || value == nullptr || mask == nullptr ||
        fa_out == nullptr || query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 ||
        value->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || fa_out->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t head_size             = query->ne[0];
    const int64_t query_token_count     = query->ne[1];
    const int64_t query_head_count      = query->ne[2];
    const int64_t key_value_token_count = key->ne[1];
    const int64_t key_value_head_count  = key->ne[2];
    const float   expected_scale        = 1.0f / std::sqrt(static_cast<float>(head_size));
    if (head_size < 64 || head_size > 1024 || head_size % 64 != 0 || query_token_count < 1 ||
        query_token_count > 2048 || query_head_count < 1 || query_head_count > 64 || key_value_token_count < 64 ||
        key_value_token_count > 32768 || key_value_token_count % 64 != 0 || key_value_head_count < 1 ||
        query_head_count % key_value_head_count != 0 || key->ne[0] != head_size || value->ne[0] != head_size ||
        value->ne[1] != key_value_token_count || value->ne[2] != key_value_head_count || fa_out->ne[0] != head_size ||
        fa_out->ne[1] != query_head_count || fa_out->ne[2] != query_token_count || query->ne[3] != 1 ||
        key->ne[3] != 1 || value->ne[3] != 1 || fa_out->ne[3] != 1 || mask->ne[0] != key_value_token_count ||
        mask->ne[1] != query_token_count || mask->ne[2] != 1 || mask->ne[3] != 1 || query->nb[0] != sizeof(float) ||
        key->nb[0] != sizeof(ggml_fp16_t) || value->nb[0] != sizeof(ggml_fp16_t) ||
        mask->nb[0] != sizeof(ggml_fp16_t) || fa_out->nb[0] != sizeof(float) ||
        !nearly_equal(params->scale, expected_scale) || !nearly_equal(params->max_bias, 0.0f) ||
        !nearly_equal(params->logit_softcap, 0.0f) ||
        (params->prec != GGML_PREC_DEFAULT && params->prec != GGML_PREC_F32)) {
        return false;
    }

    const GraphNode * reshape = single_consumer_with_op(context.graph, flash->output, GGML_OP_RESHAPE);
    if (reshape == nullptr || reshape->inputs.size() != 1) {
        return false;
    }
    const Value *     reshaped = graph_value(context.graph, reshape->output);
    const GraphNode * mul =
        reshaped != nullptr ? single_consumer_with_op(context.graph, reshaped->id, GGML_OP_MUL) : nullptr;
    if (reshaped == nullptr || mul == nullptr || mul->inputs.size() != 2 || reshaped->type != GGML_TYPE_F32 ||
        !reshaped->contiguous || reshaped->ne[0] != head_size * query_head_count ||
        reshaped->ne[1] != query_token_count || reshaped->ne[2] != 1 || reshaped->ne[3] != 1) {
        return false;
    }

    ValueId gate_value_id;
    if (mul->inputs[0] == reshape->output) {
        gate_value_id = mul->inputs[1];
    } else if (mul->inputs[1] == reshape->output) {
        gate_value_id = mul->inputs[0];
    } else {
        return false;
    }
    const GraphNode *   sigmoid        = context.graph.index().producer(gate_value_id);
    const UnaryParams * sigmoid_params = sigmoid != nullptr ? op_params_as<UnaryParams>(sigmoid->params) : nullptr;
    if (sigmoid == nullptr || sigmoid->op != GGML_OP_UNARY || sigmoid->inputs.size() != 1 ||
        sigmoid_params == nullptr || sigmoid_params->op != GGML_UNARY_OP_SIGMOID ||
        single_consumer_with_op(context.graph, sigmoid->output, GGML_OP_MUL) != mul) {
        return false;
    }
    const GraphNode * cont = context.graph.index().producer(sigmoid->inputs[0]);
    if (cont == nullptr || cont->op != GGML_OP_CONT || cont->inputs.size() != 1 ||
        single_consumer_with_op(context.graph, cont->output, GGML_OP_UNARY) != sigmoid) {
        return false;
    }
    const GraphNode * gate_view = context.graph.index().producer(cont->inputs[0]);
    const Value *     raw_gate  = gate_view != nullptr ? graph_value(context.graph, gate_view->output) : nullptr;
    const Value *     output    = graph_value(context.graph, mul->output);
    if (gate_view == nullptr || gate_view->op != GGML_OP_VIEW || gate_view->inputs.size() != 1 || raw_gate == nullptr ||
        output == nullptr || raw_gate->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        raw_gate->ne[0] != head_size || raw_gate->ne[1] != query_head_count || raw_gate->ne[2] != query_token_count ||
        raw_gate->ne[3] != 1 || raw_gate->nb[0] != sizeof(float) || output->ne != reshaped->ne || !output->contiguous ||
        single_consumer_with_op(context.graph, gate_view->output, GGML_OP_CONT) != cont) {
        return false;
    }
    if (output->storage_root == key->storage_root || output->storage_root == value->storage_root ||
        output->storage_root == mask->storage_root || output->storage_root == raw_gate->storage_root) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenFlashAttentionDirectGateKernel);
    auto & config   = dispatch.kernel.compile_parameters;
    config.emplace("hrx2_shape_fa_d", to_config_value(head_size));
    config.emplace("hrx2_shape_fa_ntokens", to_config_value(query_token_count));
    config.emplace("hrx2_shape_fa_nheads", to_config_value(query_head_count));
    config.emplace("hrx2_shape_fa_nkv", to_config_value(key_value_token_count));
    config.emplace("hrx2_shape_fa_gqa", to_config_value(query_head_count / key_value_head_count));
    config.emplace("hrx2_shape_fa_q_stride_token", to_config_value(static_cast<int64_t>(query->nb[1] / sizeof(float))));
    config.emplace("hrx2_shape_fa_q_stride_head", to_config_value(static_cast<int64_t>(query->nb[2] / sizeof(float))));
    config.emplace("hrx2_shape_fa_k_stride_pos",
                   to_config_value(static_cast<int64_t>(key->nb[1] / sizeof(ggml_fp16_t))));
    config.emplace("hrx2_shape_fa_k_stride_head",
                   to_config_value(static_cast<int64_t>(key->nb[2] / sizeof(ggml_fp16_t))));
    config.emplace("hrx2_shape_fa_v_stride_pos",
                   to_config_value(static_cast<int64_t>(value->nb[1] / sizeof(ggml_fp16_t))));
    config.emplace("hrx2_shape_fa_v_stride_head",
                   to_config_value(static_cast<int64_t>(value->nb[2] / sizeof(ggml_fp16_t))));
    config.emplace("hrx2_shape_fa_mask_stride_token",
                   to_config_value(static_cast<int64_t>(mask->nb[1] / sizeof(ggml_fp16_t))));
    config.emplace("hrx2_shape_fa_dst_stride_head",
                   to_config_value(static_cast<int64_t>(fa_out->nb[1] / sizeof(float))));
    config.emplace("hrx2_shape_fa_dst_stride_token",
                   to_config_value(static_cast<int64_t>(fa_out->nb[2] / sizeof(float))));
    config.emplace("hrx2_shape_fa_gate_stride_head",
                   to_config_value(static_cast<int64_t>(raw_gate->nb[1] / sizeof(float))));
    config.emplace("hrx2_shape_fa_gate_stride_token",
                   to_config_value(static_cast<int64_t>(raw_gate->nb[2] / sizeof(float))));
    config.emplace("hrx2_fa_apply_gate", "1");
    config.emplace("hrx2_fa_scale", std::to_string(params->scale));
    dispatch.bindings.push_back({ query->id, 0, query->byte_count });
    dispatch.bindings.push_back({ key->id, 0, key->byte_count });
    dispatch.bindings.push_back({ value->id, 0, value->byte_count });
    dispatch.bindings.push_back({ mask->id, 0, mask->byte_count });
    dispatch.bindings.push_back({ raw_gate->id, 0, raw_gate->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    for (const GraphNode * covered : { reshape, gate_view, cont, sigmoid, mul }) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, covered,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_qwen_decode_split_flash_attention_next_q8_dispatch(const DispatchMatchContext & context,
                                                                     DispatchMatch &              dispatch_match) {
    const QwenDecodeSplitFlashAttentionMatch match =
        match_qwen_decode_split_flash_attention(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const int64_t key_value_block_count = ceil_div(match.key_value_capacity, kQwenDecodeKvTileSize);
    const size_t  partial_scalar_count  = static_cast<size_t>(match.key_value_head_count) *
                                        static_cast<size_t>(key_value_block_count) *
                                        static_cast<size_t>(kQwenDecodeRowCapacity);
    const size_t  partial_value_count  = partial_scalar_count * static_cast<size_t>(kQwenAttentionHeadSize);
    const size_t  partial_scalar_bytes = partial_scalar_count * sizeof(float);
    const size_t  partial_output_bytes = partial_value_count * sizeof(ggml_fp16_t);
    const int64_t hidden_size          = match.query_head_count * kQwenAttentionHeadSize;
    const size_t  q8_row_bytes         = q8_1_x4_byte_count(1, hidden_size);
    const size_t  q8_output_bytes      = q8_1_x4_byte_count(match.query_token_count, hidden_size);
    if (partial_scalar_bytes == 0 || partial_output_bytes == 0 || q8_row_bytes == 0 || q8_output_bytes == 0) {
        return false;
    }

    const ValueId partial_max        = match_value(context, dispatch_match, 0);
    const ValueId partial_sum        = match_value(context, dispatch_match, 1);
    const ValueId partial_output     = match_value(context, dispatch_match, 2);
    const ValueId completion_counter = match_value(context, dispatch_match, 3);
    const ValueId q8_output          = match_value(context, dispatch_match, 4);

    dispatch_match.transients.push_back(
        { partial_max, "qwen.decode.flash_attention.partial_max", partial_scalar_bytes, 256 });
    dispatch_match.transients.push_back(
        { partial_sum, "qwen.decode.flash_attention.partial_sum", partial_scalar_bytes, 256 });
    dispatch_match.transients.push_back(
        { partial_output, "qwen.decode.flash_attention.partial_output", partial_output_bytes, 256 });
    dispatch_match.transients.push_back(
        { q8_output, "qwen.decode.flash_attention.next_q8_output", q8_output_bytes, 256 });
    dispatch_match.completion_counter_requests.push_back({
        completion_counter,
        "qwen.decode.flash_attention.completion_counter",
        static_cast<uint32_t>(match.key_value_head_count),
    });

    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value({ match.output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes,
                                                          "qwen.decode.flash_attention.next_q8_output" },
                                                        metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    const size_t query_row_bytes  = static_cast<size_t>(hidden_size) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(match.key_value_token_count) * sizeof(ggml_fp16_t);
    const size_t output_row_bytes = query_row_bytes;
    for (int64_t row = 0; row < match.query_token_count; ++row) {
        Dispatch dispatch;
        dispatch.kernel = make_kernel_specialization(kQwenFlashAttentionDecodeSplitNextQ8Kernel);
        dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.query_head_count",
                                                   to_config_value(match.query_head_count));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_head_count",
                                                   to_config_value(match.key_value_head_count));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_token_capacity",
                                                   to_config_value(match.key_value_capacity));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", "1");
        dispatch.bindings.push_back(
            { match.query->id, static_cast<size_t>(row) * match.query->nb[1], query_row_bytes });
        dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
        dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
        dispatch.bindings.push_back({ match.mask->id, static_cast<size_t>(row) * match.mask->nb[1], mask_row_bytes });
        dispatch.bindings.push_back({ partial_max, 0, partial_scalar_bytes });
        dispatch.bindings.push_back({ partial_sum, 0, partial_scalar_bytes });
        dispatch.bindings.push_back({ partial_output, 0, partial_output_bytes });
        dispatch.bindings.push_back(
            { completion_counter, 0, static_cast<size_t>(match.key_value_head_count) * sizeof(int32_t) });
        dispatch.bindings.push_back(
            { match.output->id, static_cast<size_t>(row) * match.output->nb[2], output_row_bytes });
        dispatch.bindings.push_back({ q8_output, static_cast<size_t>(row) * q8_row_bytes, q8_row_bytes });
        dispatch_match.dispatches.push_back(std::move(dispatch));
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.output_layout != nullptr) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.output_layout,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    return true;
}

static bool match_qwen_flash_attention_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenFlashAttentionMatch match = match_qwen_flash_attention(context.graph, context.plan, context.root_node);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenFlashAttentionF32F16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("query_token_count", match.query_token_count);
    dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.query_head_count",
                                               to_config_value(match.query_head_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_head_count",
                                               to_config_value(match.key_value_head_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(match.query_token_count));
    dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
    dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
    dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
    dispatch.bindings.push_back({ match.mask_binding_value, 0, match.mask_binding_bytes });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.output_layout != nullptr) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.output_layout,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_qwen_flash_attention_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.flash_attention_direct_f32_f16_gate",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Qwen,
        match_qwen_flash_attention_direct_gate_dispatch,
    });
    registry.add({
        "qwen.flash_attention_decode_split_next_q8",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        200,
        DispatchSource::Qwen,
        match_qwen_decode_split_flash_attention_next_q8_dispatch,
    });
    registry.add({
        "qwen.flash_attention_f32_f16_wmma",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_flash_attention_dispatch,
    });
}

}  // namespace ggml::hrx
