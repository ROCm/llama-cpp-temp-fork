#include "dispatch-qwen-matmul.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenDenseLinearQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_f16_wmma");
static constexpr KernelCatalogRef kQwenDenseLinearQ6KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_wmma");
static constexpr KernelCatalogRef kQwenDenseLinearQ6KPackedRawF16WmmaToken1Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_packed_raw_f16_wmma_token1_scalerow_64x16");
static constexpr KernelCatalogRef kQwenDenseLinearQ4KQ8NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8");
static constexpr KernelCatalogRef kGgmlLinearQ6KQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_linear_q6k_q8_1_x4");
static constexpr KernelCatalogRef kQwenQuantizeActI4Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_quant_act_i4");
static constexpr KernelCatalogRef kQwenDenseLinearSymmetricI2LowRowKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_symi2_i4_adjacent_m16n16_wg64");
static constexpr KernelCatalogRef kGgmlTopK64F32PartitionsRegisterKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_top_k64_f32_partitions_register");
static constexpr KernelCatalogRef kGgmlTopK64F32ReduceGatherRegisterKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_top_k64_f32_reduce_gather_register");
static constexpr KernelCatalogRef kQwenDenseQ6KPackedRawSelectedRefineKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_q6k_packed_raw_selected_refine_64x16");
static constexpr KernelCatalogRef kQwenFillNegativeF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_fill_negative_f32");

static constexpr int64_t kQwenHiddenSize      = kQwen30BMoeDispatchProfile.hidden_size;
static constexpr int64_t kQwenVocabularyCount = 151936;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

static bool is_qwen_endpoint_projection(int64_t input_size, int64_t output_size) {
    return input_size == kQwenHiddenSize && output_size == kQwenVocabularyCount;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static bool multiply_size(size_t lhs, size_t rhs, size_t & result) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool q6_symmetric_i2_composite_layout_sizes(int64_t  input_size,
                                                   int64_t  output_size,
                                                   size_t & symmetric_i2_bytes,
                                                   size_t & materialized_bytes) {
    if (input_size <= 0 || input_size % 256 != 0 || output_size <= 0 || output_size % 64 != 0) {
        return false;
    }

    const size_t logical_rows    = static_cast<size_t>(output_size);
    const size_t row_group       = ((logical_rows + 255) / 256) * 32;
    const size_t physical_rows   = (logical_rows + row_group - 1) / row_group * row_group;
    const size_t block_count     = static_cast<size_t>(input_size / 256);
    size_t       packed_q6_bytes = 0;
    if (!multiply_size(physical_rows, block_count, symmetric_i2_bytes) ||
        !multiply_size(symmetric_i2_bytes, size_t{ 68 }, symmetric_i2_bytes) ||
        !multiply_size(logical_rows, block_count, packed_q6_bytes) ||
        !multiply_size(packed_q6_bytes, ggml_type_size(GGML_TYPE_Q6_K), packed_q6_bytes) ||
        symmetric_i2_bytes > std::numeric_limits<size_t>::max() - packed_q6_bytes ||
        symmetric_i2_bytes > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        symmetric_i2_bytes % 256 != 0) {
        return false;
    }
    materialized_bytes = symmetric_i2_bytes + packed_q6_bytes;
    return true;
}

static const Value * match_mtp_normalized_branch(const Graph & graph,
                                                 ValueId       branch_id,
                                                 int64_t       hidden_size,
                                                 int64_t       token_count) {
    const Value *     branch = graph_value(graph, branch_id);
    const GraphNode * mul    = graph.index().producer(branch_id);
    if (branch == nullptr || mul == nullptr || mul->op != GGML_OP_MUL || mul->inputs.size() != 2 ||
        branch->type != GGML_TYPE_F32 || !is_2d(*branch) || branch->ne[0] != hidden_size ||
        branch->ne[1] != token_count) {
        return nullptr;
    }

    const GraphNode * rms   = nullptr;
    const Value *     scale = nullptr;
    for (ValueId input_id : mul->inputs) {
        const GraphNode * producer = graph.index().producer(input_id);
        if (producer != nullptr && producer->op == GGML_OP_RMS_NORM) {
            if (rms != nullptr) {
                return nullptr;
            }
            rms = producer;
        } else {
            scale = graph_value(graph, input_id);
        }
    }
    if (rms == nullptr || rms->inputs.size() != 1 || scale == nullptr || scale->type != GGML_TYPE_F32 ||
        !is_2d(*scale) || scale->ne[0] != hidden_size || scale->ne[1] != 1) {
        return nullptr;
    }

    const Value * normalized = graph_value(graph, rms->output);
    const Value * source     = graph_value(graph, rms->inputs[0]);
    if (normalized == nullptr || source == nullptr || normalized->id != rms->output ||
        normalized->type != GGML_TYPE_F32 || source->type != GGML_TYPE_F32 || !is_2d(*normalized) || !is_2d(*source) ||
        normalized->ne[0] != hidden_size || normalized->ne[1] != token_count || source->ne[0] != hidden_size ||
        source->ne[1] != token_count) {
        return nullptr;
    }
    return source;
}

static bool value_depends_on(const Graph & graph, ValueId value, ValueId ancestor) {
    if (value.value < 0 || ancestor.value < 0) {
        return false;
    }
    std::vector<bool>    visited(graph.values().size(), false);
    std::vector<ValueId> pending = { value };
    while (!pending.empty()) {
        const ValueId current = pending.back();
        pending.pop_back();
        if (current == ancestor) {
            return true;
        }
        if (current.value < 0 || static_cast<size_t>(current.value) >= visited.size() || visited[current.value]) {
            continue;
        }
        visited[current.value]     = true;
        const GraphNode * producer = graph.index().producer(current);
        if (producer != nullptr) {
            pending.insert(pending.end(), producer->inputs.begin(), producer->inputs.end());
        }
    }
    return false;
}

static bool graph_has_mtp_preamble_ancestor(const Graph & graph, ValueId endpoint_input, int64_t hidden_size) {
    if (!graph.has_index() || hidden_size <= 0) {
        return false;
    }

    for (const GraphNode & projection : graph.nodes()) {
        if (projection.op != GGML_OP_MUL_MAT || projection.inputs.size() != 2) {
            continue;
        }
        const Value * weight      = graph_value(graph, projection.inputs[0]);
        const Value * input       = graph_value(graph, projection.inputs[1]);
        const Value * output      = graph_value(graph, projection.output);
        const int64_t token_count = input != nullptr ? input->ne[1] : 0;
        if (weight == nullptr || input == nullptr || output == nullptr || weight->type != GGML_TYPE_Q6_K ||
            input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !is_2d(*weight) || !is_2d(*input) ||
            !is_2d(*output) || weight->ne[0] != 2 * hidden_size || weight->ne[1] != hidden_size ||
            input->ne[0] != 2 * hidden_size || token_count != 1 || output->ne[0] != hidden_size ||
            output->ne[1] != token_count) {
            continue;
        }

        const GraphNode * concat = graph.index().producer(input->id);
        if (concat == nullptr || concat->op != GGML_OP_CONCAT || concat->inputs.size() != 2) {
            continue;
        }
        const Value * first_source  = match_mtp_normalized_branch(graph, concat->inputs[0], hidden_size, token_count);
        const Value * second_source = match_mtp_normalized_branch(graph, concat->inputs[1], hidden_size, token_count);
        if (first_source == nullptr || second_source == nullptr) {
            continue;
        }

        const GraphNode * first_producer  = graph.index().producer(first_source->id);
        const GraphNode * second_producer = graph.index().producer(second_source->id);
        const bool        first_is_embedding =
            first_producer != nullptr && first_producer->op == GGML_OP_GET_ROWS && first_producer->inputs.size() == 2;
        const bool second_is_embedding = second_producer != nullptr && second_producer->op == GGML_OP_GET_ROWS &&
                                         second_producer->inputs.size() == 2;
        const bool first_is_external  = first_producer == nullptr;
        const bool second_is_external = second_producer == nullptr;
        if (((first_is_embedding && second_is_external) || (second_is_embedding && first_is_external)) &&
            value_depends_on(graph, endpoint_input, projection.output)) {
            return true;
        }
    }
    return false;
}

struct QwenMatmulMatch {
    const Value *    input       = nullptr;
    const Value *    weight      = nullptr;
    const Value *    output      = nullptr;
    KernelCatalogRef kernel      = {};
    ValueId          input_value = {};
    size_t           input_bytes = 0;
    int64_t          input_size  = 0;
    int64_t          output_size = 0;
    int64_t          token_count = 0;
    bool             dense        = false;
    bool             draft_refine = false;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

struct QwenAttentionOutputNextQ8Match {
    const Value *                     input               = nullptr;
    const CommandPlanAlternateValue * input_alternate     = nullptr;
    const Value *                     weight              = nullptr;
    const Value *                     projection_output   = nullptr;
    const Value *                     residual_input      = nullptr;
    const Value *                     residual_output     = nullptr;
    const Value *                     norm_weight         = nullptr;
    const Value *                     normalized_output   = nullptr;
    const GraphNode *                 projection_get_rows = nullptr;
    const GraphNode *                 residual_get_rows   = nullptr;
    const GraphNode *                 add_node            = nullptr;
    const GraphNode *                 rms_node            = nullptr;
    const GraphNode *                 mul_node            = nullptr;
    int64_t                           input_size          = 0;
    int64_t                           output_size         = 0;
    int64_t                           token_count         = 0;

    bool matched() const {
        return input != nullptr && input_alternate != nullptr && weight != nullptr && projection_output != nullptr &&
               residual_input != nullptr && residual_output != nullptr && norm_weight != nullptr &&
               normalized_output != nullptr && add_node != nullptr && rms_node != nullptr && mul_node != nullptr;
    }
};

struct QwenAttentionOutputAccumulateMatch {
    const Value *     input             = nullptr;
    const Value *     weight            = nullptr;
    const Value *     projection_output = nullptr;
    const Value *     residual_input    = nullptr;
    const Value *     residual_output   = nullptr;
    const GraphNode * add_node          = nullptr;
    int64_t           input_size        = 0;
    int64_t           output_size       = 0;
    int64_t           token_count       = 0;

    bool matched() const {
        return input != nullptr && weight != nullptr && projection_output != nullptr && residual_input != nullptr &&
               residual_output != nullptr && add_node != nullptr;
    }
};

static size_t q8_1_x4_byte_count(int64_t token_count, int64_t input_size) {
    if (token_count <= 0 || input_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * match = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != op) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = consumer;
    }
    return match;
}

static bool value_has_no_uncovered_consumers_except(const DispatchMatchContext & context,
                                                    ValueId                      value,
                                                    const GraphNode *            expected_consumer) {
    for (const GraphNode * consumer : context.graph.index().consumers(value)) {
        if (consumer == expected_consumer) {
            continue;
        }
        size_t consumer_index = 0;
        if (!context.graph.index().node_index(consumer, consumer_index) ||
            consumer_index >= context.covered_nodes.size() || !context.covered_nodes[consumer_index]) {
            return false;
        }
    }
    return true;
}

static bool same_value_layout(const Value & lhs, const Value & rhs) {
    if (lhs.type != rhs.type || lhs.byte_count != rhs.byte_count || lhs.element_count != rhs.element_count ||
        lhs.contiguous != rhs.contiguous) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i] || lhs.nb[i] != rhs.nb[i]) {
            return false;
        }
    }
    return true;
}

static const Value * get_rows_source_with_same_layout(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return nullptr;
    }
    const Value * source = graph_value(graph, node->inputs[0]);
    const Value * output = graph_value(graph, node->output);
    if (source == nullptr || output == nullptr || !same_value_layout(*source, *output)) {
        return nullptr;
    }
    return source;
}

static QwenMatmulMatch match_qwen_q6k_q8_matmul(const Graph & graph, const GraphNode * node, const CommandPlan & plan) {
    QwenMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return {};
    }
    if (!is_2d(*weight) || !is_2d(*input) || !is_2d(*output)) {
        return {};
    }
    if (!weight->contiguous || !input->contiguous || !output->contiguous) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q6_K || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count) {
        return {};
    }
    if (!is_qwen_decode_query_length(token_count) || input_size != kQwenHiddenSize ||
        output_size != kQwenVocabularyCount || !is_supported_dense_input_size(input_size) ||
        !is_supported_dense_output_size(output_size)) {
        return {};
    }

    const size_t                      q8_byte_count = q8_1_x4_byte_count(token_count, input_size);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(graph, plan, input->id, GGML_TYPE_Q8_1, q8_byte_count);
    if (alternate == nullptr) {
        return {};
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_value = alternate->alternate_value;
    match.input_bytes = alternate->byte_count;
    match.kernel      = kGgmlLinearQ6KQ8_1X4Kernel;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    return match;
}

static QwenMatmulMatch match_qwen_decode_endpoint_q6k_matmul(const Graph & graph, const GraphNode * node) {
    QwenMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        weight->type != GGML_TYPE_Q6_K || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !is_qwen_decode_query_length(token_count) || !graph.index().consumers(output->id).empty() ||
        !is_supported_dense_input_size(input_size) || !is_supported_dense_output_size(output_size)) {
        return {};
    }

    size_t     symmetric_i2_bytes     = 0;
    size_t     composite_weight_bytes = 0;
    const bool draft_refine =
        token_count == 1 && output_size >= 65536 && output_size % 64 == 0 &&
        graph_has_mtp_preamble_ancestor(graph, input->id, input_size) &&
        q6_symmetric_i2_composite_layout_sizes(input_size, output_size, symmetric_i2_bytes, composite_weight_bytes);

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_value = input->id;
    match.input_bytes = input->byte_count;
    match.kernel       = draft_refine ? kQwenDenseLinearSymmetricI2LowRowKernel :
                         token_count == 1 && output_size >= 65536 && output_size % 64 == 0 ?
                                        kQwenDenseLinearQ6KPackedRawF16WmmaToken1Kernel :
                                        kQwenDenseLinearQ6KF16WmmaKernel;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    match.dense       = true;
    match.draft_refine = draft_refine;
    return match;
}

static QwenAttentionOutputNextQ8Match match_qwen_attention_output_next_q8(const DispatchMatchContext & context) {
    QwenAttentionOutputNextQ8Match match;
    const Graph &                  graph = context.graph;
    const GraphNode *              node  = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * weight            = graph_value(graph, node->inputs[0]);
    const Value * input             = graph_value(graph, node->inputs[1]);
    const Value * projection_output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || projection_output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*projection_output)) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 || projection_output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !input->contiguous || !projection_output->contiguous) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (token_count != 1 || input->ne[0] != input_size || projection_output->ne[0] != output_size ||
        projection_output->ne[1] != token_count || input_size != 4096 || output_size != kQwenHiddenSize) {
        return {};
    }

    const CommandPlanAlternateValue * input_alternate = find_alternate_value(
        graph, context.plan, input->id, GGML_TYPE_Q8_1, q8_1_x4_byte_count(token_count, input_size));
    if (input_alternate == nullptr) {
        return {};
    }

    const Value *     selected_projection = projection_output;
    const GraphNode * projection_get_rows = nullptr;
    const GraphNode * add_node            = find_single_consumer_with_op(graph, projection_output->id, GGML_OP_ADD);
    if (add_node == nullptr) {
        projection_get_rows = find_single_consumer_with_op(graph, projection_output->id, GGML_OP_GET_ROWS);
        if (get_rows_source_with_same_layout(graph, projection_get_rows) != projection_output) {
            return {};
        }
        selected_projection = graph_value(graph, projection_get_rows->output);
        add_node            = find_single_consumer_with_op(graph, projection_get_rows->output, GGML_OP_ADD);
    }
    if (add_node == nullptr || add_node->inputs.size() != 2) {
        return {};
    }
    const Value *     residual_input    = nullptr;
    const Value *     selected_residual = nullptr;
    const GraphNode * residual_get_rows = nullptr;
    const GraphNode * residual_consumer = add_node;
    if (add_node->inputs[0] == selected_projection->id) {
        selected_residual = graph_value(graph, add_node->inputs[1]);
    } else if (add_node->inputs[1] == selected_projection->id) {
        selected_residual = graph_value(graph, add_node->inputs[0]);
    }
    if (selected_residual == nullptr) {
        return {};
    }
    const GraphNode * selected_residual_producer = graph.index().producer(selected_residual->id);
    residual_input                               = get_rows_source_with_same_layout(graph, selected_residual_producer);
    if (residual_input != nullptr) {
        residual_get_rows = selected_residual_producer;
        residual_consumer = residual_get_rows;
    } else {
        residual_input = selected_residual;
    }
    const Value * residual_output = graph_value(graph, add_node->output);
    if (residual_input == nullptr || residual_output == nullptr || residual_input->type != GGML_TYPE_F32 ||
        residual_output->type != GGML_TYPE_F32 || !same_value_layout(*selected_projection, *selected_residual) ||
        !same_value_layout(*selected_projection, *residual_input) ||
        !same_value_layout(*selected_projection, *residual_output) ||
        !value_has_no_uncovered_consumers_except(context, residual_input->id, residual_consumer)) {
        return {};
    }

    const GraphNode * rms_node = find_single_consumer_with_op(graph, residual_output->id, GGML_OP_RMS_NORM);
    if (rms_node == nullptr || rms_node->inputs.size() != 1) {
        return {};
    }
    const Value *     rms_output = graph_value(graph, rms_node->output);
    const GraphNode * mul_node   = find_single_consumer_with_op(graph, rms_node->output, GGML_OP_MUL);
    if (rms_output == nullptr || mul_node == nullptr || mul_node->inputs.size() != 2) {
        return {};
    }

    const Value * norm_weight = nullptr;
    if (mul_node->inputs[0] == rms_node->output) {
        norm_weight = graph_value(graph, mul_node->inputs[1]);
    } else if (mul_node->inputs[1] == rms_node->output) {
        norm_weight = graph_value(graph, mul_node->inputs[0]);
    }
    const Value * normalized_output = graph_value(graph, mul_node->output);
    if (norm_weight == nullptr || normalized_output == nullptr || norm_weight->type != GGML_TYPE_F32 ||
        normalized_output->type != GGML_TYPE_F32 || !norm_weight->contiguous || !normalized_output->contiguous ||
        norm_weight->ne[0] != output_size || normalized_output->ne[0] != output_size ||
        normalized_output->ne[1] != token_count) {
        return {};
    }

    match.input               = input;
    match.input_alternate     = input_alternate;
    match.weight              = weight;
    match.projection_output   = projection_output;
    match.residual_input      = residual_input;
    match.residual_output     = residual_output;
    match.norm_weight         = norm_weight;
    match.normalized_output   = normalized_output;
    match.projection_get_rows = projection_get_rows;
    match.residual_get_rows   = residual_get_rows;
    match.add_node            = add_node;
    match.rms_node            = rms_node;
    match.mul_node            = mul_node;
    match.input_size          = input_size;
    match.output_size         = output_size;
    match.token_count         = token_count;
    return match;
}

static QwenAttentionOutputAccumulateMatch match_qwen_attention_output_accumulate(const DispatchMatchContext & context) {
    QwenAttentionOutputAccumulateMatch match;
    const Graph &                      graph = context.graph;
    const GraphNode *                  node  = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * weight            = graph_value(graph, node->inputs[0]);
    const Value * input             = graph_value(graph, node->inputs[1]);
    const Value * projection_output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || projection_output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*projection_output)) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 || projection_output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !input->contiguous || !projection_output->contiguous) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (!is_qwen_prefill_query_length(token_count) || input->ne[0] != input_size ||
        projection_output->ne[0] != output_size || projection_output->ne[1] != token_count || input_size != 4096 ||
        output_size != kQwenHiddenSize) {
        return {};
    }

    const GraphNode * add_node = find_single_consumer_with_op(graph, projection_output->id, GGML_OP_ADD);
    if (add_node == nullptr || add_node->inputs.size() != 2) {
        return {};
    }
    const Value * residual_input = nullptr;
    if (add_node->inputs[0] == projection_output->id) {
        residual_input = graph_value(graph, add_node->inputs[1]);
    } else if (add_node->inputs[1] == projection_output->id) {
        residual_input = graph_value(graph, add_node->inputs[0]);
    }
    const Value * residual_output = graph_value(graph, add_node->output);
    if (residual_input == nullptr || residual_output == nullptr || residual_input->type != GGML_TYPE_F32 ||
        residual_output->type != GGML_TYPE_F32 || !same_value_layout(*projection_output, *residual_input) ||
        !same_value_layout(*projection_output, *residual_output) ||
        !value_has_no_uncovered_consumers_except(context, residual_input->id, add_node)) {
        return {};
    }

    match.input             = input;
    match.weight            = weight;
    match.projection_output = projection_output;
    match.residual_input    = residual_input;
    match.residual_output   = residual_output;
    match.add_node          = add_node;
    match.input_size        = input_size;
    match.output_size       = output_size;
    match.token_count       = token_count;
    return match;
}

}  // namespace

static bool build_qwen_draft_refinement_dispatch(const QwenMatmulMatch &      match,
                                                 DispatchMatch &              dispatch_match,
                                                 const DispatchMatchContext & context) {
    constexpr int64_t candidate_count       = 64;
    constexpr size_t  partition_entry_count = 512;
    constexpr size_t  candidate_capacity    = 64;

    size_t symmetric_i2_bytes        = 0;
    size_t materialized_weight_bytes = 0;
    if (!q6_symmetric_i2_composite_layout_sizes(match.input_size, match.output_size, symmetric_i2_bytes,
                                                materialized_weight_bytes)) {
        dispatch_match.status.log("invalid Q6 draft refinement layout for K=%lld rows=%lld",
                                  static_cast<long long>(match.input_size), static_cast<long long>(match.output_size));
        return false;
    }

    const LlmSymmetricI4ActivationLayout activation_layout =
        llm_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kLlmSymmetricI4ActivationAlternateName) {
        alternate = nullptr;
    }

    const ValueId activation     = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    const int32_t transient_base = context.next_plan_value.value + (alternate == nullptr ? 1 : 0);
    const ValueId partial_values(transient_base);
    const ValueId partial_ids(transient_base + 1);
    const ValueId candidates(transient_base + 2);
    const ValueId approximate_candidates(transient_base + 3);
    const ValueId exact_candidates(transient_base + 4);

    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kLlmSymmetricI4ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQwenQuantizeActI4Kernel);
        quantize.kernel.compile_parameters.emplace("qwen3.qact_i4.shape_k", to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("qwen3.qact_i4.shape_cols", to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));
    }

    dispatch_match.transients.push_back(
        { partial_values, "qwen.draft.endpoint.top_k.partial_values", partition_entry_count * sizeof(float), 256 });
    dispatch_match.transients.push_back(
        { partial_ids, "qwen.draft.endpoint.top_k.partial_ids", partition_entry_count * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back(
        { candidates, "qwen.draft.endpoint.top_k.candidates", candidate_capacity * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back({ approximate_candidates, "qwen.draft.endpoint.top_k.approximate_values",
                                          candidate_capacity * sizeof(float), 256 });
    dispatch_match.transients.push_back(
        { exact_candidates, "qwen.draft.endpoint.top_k.exact_values", candidate_capacity * sizeof(float), 256 });

    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kQwenDenseLinearSymmetricI2LowRowKernel);
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_i4.lowrow.shape_k",
                                                  to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_i4.lowrow.shape_rows",
                                                  to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("qwen3.dense_q4_i4.lowrow.shape_cols",
                                                  to_config_value(match.token_count));
    contraction.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes,
                                     kQ6KSymmetricI2PackedScaleRowLayout, GGML_TYPE_Q6_K, match.input_size,
                                     match.output_size, match.weight->byte_count });
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    contraction.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    contraction.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
    dispatch_match.dispatches.push_back(std::move(contraction));

    Dispatch partition_top_k;
    partition_top_k.kernel = make_kernel_specialization(kGgmlTopK64F32PartitionsRegisterKernel);
    partition_top_k.kernel.integer_parameters.emplace("element_count", match.output_size);
    partition_top_k.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    partition_top_k.bindings.push_back({ partial_values, 0, partition_entry_count * sizeof(float) });
    partition_top_k.bindings.push_back({ partial_ids, 0, partition_entry_count * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(partition_top_k));

    Dispatch reduce_top_k;
    reduce_top_k.kernel = make_kernel_specialization(kGgmlTopK64F32ReduceGatherRegisterKernel);
    reduce_top_k.kernel.integer_parameters.emplace("element_count", match.output_size);
    reduce_top_k.bindings.push_back({ partial_values, 0, partition_entry_count * sizeof(float) });
    reduce_top_k.bindings.push_back({ partial_ids, 0, partition_entry_count * sizeof(int32_t) });
    reduce_top_k.bindings.push_back({ candidates, 0, candidate_capacity * sizeof(int32_t) });
    reduce_top_k.bindings.push_back({ approximate_candidates, 0, candidate_capacity * sizeof(float) });
    dispatch_match.dispatches.push_back(std::move(reduce_top_k));

    Dispatch fill;
    fill.kernel = make_kernel_specialization(kQwenFillNegativeF32Kernel);
    fill.kernel.integer_parameters.emplace("element_count", match.output_size);
    fill.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch_match.dispatches.push_back(std::move(fill));

    Dispatch refine;
    refine.kernel = make_kernel_specialization(kQwenDenseQ6KPackedRawSelectedRefineKernel);
    refine.kernel.integer_parameters.emplace("token_count", match.token_count);
    refine.kernel.integer_parameters.emplace("candidate_count", candidate_count);
    refine.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size", to_config_value(match.input_size));
    refine.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                             to_config_value(match.output_size));
    refine.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    refine.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.q6_refine_weight_offset",
                                             to_config_value(static_cast<int64_t>(symmetric_i2_bytes)));
    refine.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    refine.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kQ6KSymmetricI2PackedScaleRowLayout,
                                GGML_TYPE_Q6_K, match.input_size, match.output_size, match.weight->byte_count });
    refine.bindings.push_back({ candidates, 0, candidate_capacity * sizeof(int32_t) });
    refine.bindings.push_back({ exact_candidates, 0, candidate_capacity * sizeof(float) });
    refine.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch_match.dispatches.push_back(std::move(refine));

    dispatch_match.covered_nodes.push_back(context.root_index);
    return true;
}

static void build_qwen_matmul_dispatch(const QwenMatmulMatch & match,
                                       DispatchMatch &         dispatch_match,
                                       size_t                  root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    if (match.kernel.id == kGgmlLinearQ6KQ8_1X4Kernel.id) {
        dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
        dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
        dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.token_capacity",
                                                   to_config_value(match.token_count));
        dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.output_capacity",
                                                   to_config_value(match.output_size));
    } else {
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                                   to_config_value(match.token_count));
    }
    if (match.dense) {
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                                   to_config_value(match.input_size));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                                   to_config_value(match.output_size));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    }
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    if (match.kernel.id == kQwenDenseLinearQ6KPackedRawF16WmmaToken1Kernel.id) {
        const size_t block_count = static_cast<size_t>(match.input_size / ggml_blck_size(GGML_TYPE_Q6_K));
        const size_t materialized_weight_bytes =
            static_cast<size_t>(match.output_size) * block_count * ggml_type_size(GGML_TYPE_Q6_K);
        dispatch.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kQ6KPackedScaleRowLayout,
                                      GGML_TYPE_Q6_K, match.input_size, match.output_size, match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_qwen_q6k_q8_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenMatmulMatch match = match_qwen_q6k_q8_matmul(context.graph, context.root_node, context.plan);
    if (!match.matched()) {
        return false;
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_qwen_decode_endpoint_q6k_dispatch(const DispatchMatchContext & context,
                                                    DispatchMatch &              dispatch_match) {
    const QwenMatmulMatch match = match_qwen_decode_endpoint_q6k_matmul(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }
    if (match.draft_refine) {
        return build_qwen_draft_refinement_dispatch(match, dispatch_match, context);
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_qwen_attention_output_next_q8_dispatch(const DispatchMatchContext & context,
                                                         DispatchMatch &              dispatch_match) {
    const QwenAttentionOutputNextQ8Match match = match_qwen_attention_output_next_q8(context);
    if (!match.matched()) {
        return false;
    }

    const ValueId completion_counter(context.next_plan_value.value);
    const ValueId q8_output(context.next_plan_value.value + 1);
    const size_t  q8_output_bytes = q8_1_x4_byte_count(match.token_count, match.output_size);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenDenseLinearQ4KQ8NextQ8Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "1");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });
    dispatch.bindings.push_back({ match.norm_weight->id, 0, match.norm_weight->byte_count });
    dispatch.bindings.push_back({ match.normalized_output->id, 0, match.normalized_output->byte_count });
    dispatch.bindings.push_back({ completion_counter, 0, sizeof(int32_t) });
    dispatch.bindings.push_back({ q8_output, 0, q8_output_bytes });

    dispatch_match.value_aliases.push_back({ match.residual_input->id, match.residual_output->id });
    dispatch_match.completion_counter_requests.push_back({
        completion_counter,
        "qwen.decode.attention_output.completion_counter",
        1,
    });
    dispatch_match.transients.push_back(
        { q8_output, "qwen.decode.attention_output.next_q8_output", q8_output_bytes, 256 });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.normalized_output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes,
              "qwen.decode.attention_output.next_q8_output" },
            metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        (match.projection_get_rows != nullptr &&
         !append_covered_node_index_once(context.graph, context.covered_nodes, match.projection_get_rows,
                                         dispatch_match.covered_nodes)) ||
        (match.residual_get_rows != nullptr &&
         !append_covered_node_index_once(context.graph, context.covered_nodes, match.residual_get_rows,
                                         dispatch_match.covered_nodes)) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.add_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.rms_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.mul_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_qwen_attention_output_accumulate_dispatch(const DispatchMatchContext & context,
                                                            DispatchMatch &              dispatch_match) {
    const QwenAttentionOutputAccumulateMatch match = match_qwen_attention_output_accumulate(context);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenDenseLinearQ4KF16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "1");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });

    dispatch_match.value_aliases.push_back({ match.residual_input->id, match.residual_output->id });
    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.add_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_qwen_matmul_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.matmul.attention_output_q4k_q8_1_x4_next_q8",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Qwen,
        match_qwen_attention_output_next_q8_dispatch,
    });
    registry.add({
        "qwen.matmul.attention_output_q4k_f16_accumulate",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        250,
        DispatchSource::Qwen,
        match_qwen_attention_output_accumulate_dispatch,
    });
    registry.add({
        "qwen.matmul.q6k_q8_1_x4",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        200,
        DispatchSource::Qwen,
        match_qwen_q6k_q8_dispatch,
    });
    registry.add({
        "qwen.matmul.decode_endpoint_q6k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_decode_endpoint_q6k_dispatch,
    });
}

}  // namespace ggml::hrx
