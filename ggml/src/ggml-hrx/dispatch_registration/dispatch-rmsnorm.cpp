#include "dispatch-rmsnorm.h"

#include "dispatch-llm-profiles.h"
#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRmsNormF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_rmsnorm_f32");
static constexpr KernelCatalogRef kQwenRmsNormF32F16Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_rmsnorm_f32_f16");
static constexpr KernelCatalogRef kQwenRmsNormF32I4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_rmsnorm_f32_i4");
static constexpr KernelCatalogRef kQwenAddRmsNormF32I4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_add_rmsnorm_f32_i4");
static constexpr KernelCatalogRef kQwenRmsNormF32QuantizeQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4");
static constexpr KernelCatalogRef kQuantizeQ8_1X4Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
static constexpr KernelCatalogRef kGgmlLinearQ6KQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_linear_q6k_q8_1_x4");
static constexpr float   kQwenRmsNormEpsilon  = kQwen30BMoeDispatchProfile.rms_norm_epsilon;
static constexpr int64_t kQwenHiddenSize      = kQwen30BMoeDispatchProfile.hidden_size;
static constexpr int64_t kQwenVocabularyCount = 151936;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

template <size_t N> static bool pairwise_distinct_storage(const std::array<const Value *, N> & values) {
    for (size_t lhs = 0; lhs < values.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < values.size(); ++rhs) {
            if (values[lhs]->storage == values[rhs]->storage) {
                return false;
            }
        }
    }
    return true;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_qwen_rms_norm_epsilon(float eps) {
    return std::fabs(eps - kQwenRmsNormEpsilon) <= 1.0e-12f;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return is_llm_supported_query_length(kQwen30BMoeDispatchProfile, token_count);
}

static bool has_decode_q8_consumer(const Graph & graph, ValueId value) {
    if (!graph.has_index()) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && (consumer->op == GGML_OP_MUL_MAT || consumer->op == GGML_OP_MUL_MAT_ID)) {
            return true;
        }
    }
    return false;
}

static bool has_prefill_f16_dense_consumers_only(const Graph & graph,
                                                 ValueId       value,
                                                 int64_t       token_count,
                                                 int64_t       hidden_size) {
    if (!graph.has_index() || token_count < 128) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    if (consumers.size() < 2) {
        return false;
    }
    for (const GraphNode * consumer : consumers) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
            consumer->inputs[1] != value) {
            return false;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        const Value * output = graph_value(graph, consumer->output);
        if (weight == nullptr || output == nullptr ||
            (weight->type != GGML_TYPE_Q5_K && weight->type != GGML_TYPE_IQ4_XS) || output->type != GGML_TYPE_F32 ||
            !weight->contiguous || !output->contiguous || weight->ne[0] != hidden_size ||
            output->ne[0] != weight->ne[1] || output->ne[1] != token_count || output->ne[2] != 1 ||
            output->ne[3] != 1) {
            return false;
        }
    }
    return true;
}

static bool has_prefill_q5_dense_consumer(const Graph & graph,
                                          ValueId       value,
                                          int64_t       token_count,
                                          int64_t       hidden_size) {
    if (!graph.has_index() || token_count < 128 || token_count % 128 != 0) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
            consumer->inputs[1] != value) {
            continue;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        const Value * output = graph_value(graph, consumer->output);
        if (weight != nullptr && output != nullptr && weight->type == GGML_TYPE_Q5_K && output->type == GGML_TYPE_F32 &&
            weight->contiguous && output->contiguous && weight->ne[0] == hidden_size && weight->ne[1] % 64 == 0 &&
            output->ne[0] == weight->ne[1] && output->ne[1] == token_count && output->ne[2] == 1 &&
            output->ne[3] == 1) {
            return true;
        }
    }
    return false;
}

static bool has_low_row_q8_dense_consumer(const Graph & graph,
                                          ValueId       value,
                                          int64_t       token_count,
                                          int64_t       hidden_size) {
    if (!graph.has_index() || token_count != 2) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
            consumer->inputs[1] != value) {
            continue;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        const Value * output = graph_value(graph, consumer->output);
        if (weight != nullptr && output != nullptr &&
            (weight->type == GGML_TYPE_Q4_K || weight->type == GGML_TYPE_Q6_K) && output->type == GGML_TYPE_F32 &&
            weight->contiguous && output->contiguous && weight->ne[0] == hidden_size &&
            output->ne[0] == weight->ne[1] && output->ne[1] == token_count && output->ne[2] == 1 &&
            output->ne[3] == 1) {
            return true;
        }
    }
    return false;
}

static bool has_low_row_symmetric_i4_consumer(const Graph & graph,
                                              ValueId       value,
                                              int64_t       token_count,
                                              int64_t       hidden_size) {
    if (!graph.has_index() || token_count < 1 || token_count > 16 || hidden_size % 64 != 0) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
            consumer->inputs[1] != value) {
            continue;
        }
        const Value * weight = graph_value(graph, consumer->inputs[0]);
        const Value * output = graph_value(graph, consumer->output);
        if (weight == nullptr || output == nullptr || output->type != GGML_TYPE_F32 || !weight->contiguous ||
            !output->contiguous || weight->ne[0] != hidden_size || weight->ne[1] != output->ne[0] ||
            output->ne[1] != token_count || output->ne[2] != 1 || output->ne[3] != 1 || output->ne[0] % 64 != 0) {
            continue;
        }
        if (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_IQ4_XS ||
            (weight->type == GGML_TYPE_Q4_K && output->ne[0] >= 2 * hidden_size)) {
            return true;
        }
    }
    return false;
}

static size_t q8_1_x4_byte_count(int64_t token_count, int64_t hidden_size) {
    if (token_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static size_t f16_byte_count(int64_t token_count, int64_t hidden_size) {
    if (token_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_F16, hidden_size);
}

static bool is_weight_shape(const Value & weight, int64_t hidden_size) {
    if (weight.ne[0] != hidden_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (weight.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

struct RmsNormMatch {
    const GraphNode * rms_node       = nullptr;
    const GraphNode * mul_node       = nullptr;
    const Value *     input          = nullptr;
    const Value *     weight         = nullptr;
    const Value *     output         = nullptr;
    size_t            rms_node_index = 0;
    size_t            mul_node_index = 0;
    int64_t           hidden_size    = 0;
    int64_t           token_count    = 0;
    int64_t           q8_group_count = 0;

    bool matched() const {
        return rms_node != nullptr && mul_node != nullptr && input != nullptr && weight != nullptr && output != nullptr;
    }
};

static RmsNormMatch match_qwen_rmsnorm_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    RmsNormMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !is_qwen_rms_norm_epsilon(rms_params->eps)) {
        return {};
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(node->output);
    if (consumers.size() != 1) {
        return {};
    }
    const GraphNode * mul_node = consumers.front();
    size_t            mul_node_index;
    if (mul_node == nullptr || mul_node->op != GGML_OP_MUL || mul_node->inputs.size() != 2 ||
        !graph.index().node_index(mul_node, mul_node_index)) {
        return {};
    }

    const Value * weight = nullptr;
    for (ValueId input : mul_node->inputs) {
        if (input != node->output) {
            weight = graph_value(graph, input);
        }
    }
    const Value * input  = graph_value(graph, node->inputs[0]);
    const Value * rms    = graph_value(graph, node->output);
    const Value * output = graph_value(graph, mul_node->output);
    if (input == nullptr || rms == nullptr || weight == nullptr || output == nullptr) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32) {
        return {};
    }
    if (!input->contiguous || !rms->contiguous || !weight->contiguous || !output->contiguous) {
        return {};
    }
    if (!same_shape(*input, *rms) || !same_shape(*input, *output)) {
        return {};
    }
    if (!pairwise_distinct_storage(std::array<const Value *, 3>{ input, weight, output })) {
        return {};
    }

    const int64_t hidden_size = output->ne[0];
    if (!is_supported_hidden_size(hidden_size) || !is_weight_shape(*weight, hidden_size)) {
        return {};
    }
    if (hidden_size == 0 || output->element_count <= 0 || output->element_count % hidden_size != 0) {
        return {};
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (!is_supported_token_count(token_count)) {
        return {};
    }

    match.rms_node       = node;
    match.mul_node       = mul_node;
    match.input          = input;
    match.weight         = weight;
    match.output         = output;
    match.rms_node_index = node_index;
    match.mul_node_index = mul_node_index;
    match.hidden_size    = hidden_size;
    match.token_count    = token_count;
    match.q8_group_count = token_count * ((hidden_size + 127) / 128);
    return match;
}

struct AddRmsNormMatch {
    RmsNormMatch      rms;
    const GraphNode * add_node       = nullptr;
    const Value *     input          = nullptr;
    const Value *     addend         = nullptr;
    const Value *     residual       = nullptr;
    size_t            add_node_index = 0;

    bool matched() const {
        return rms.matched() && add_node != nullptr && input != nullptr && addend != nullptr && residual != nullptr;
    }
};

static AddRmsNormMatch match_qwen_add_rmsnorm_f32_i4(const Graph & graph, const GraphNode * node, size_t node_index) {
    AddRmsNormMatch match;
    if (node == nullptr || node->op != GGML_OP_ADD || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * input    = graph_value(graph, node->inputs[0]);
    const Value * addend   = graph_value(graph, node->inputs[1]);
    const Value * residual = graph_value(graph, node->output);
    if (input == nullptr || addend == nullptr || residual == nullptr || input->type != GGML_TYPE_F32 ||
        addend->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || !input->contiguous || !addend->contiguous ||
        !residual->contiguous || !same_shape(*input, *residual) || !same_shape(*addend, *residual)) {
        return {};
    }

    const GraphNode * rms_node = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(residual->id)) {
        if (consumer == nullptr || consumer->op != GGML_OP_RMS_NORM) {
            continue;
        }
        if (rms_node != nullptr) {
            return {};
        }
        rms_node = consumer;
    }
    size_t rms_node_index = 0;
    if (rms_node == nullptr || !graph.index().node_index(rms_node, rms_node_index)) {
        return {};
    }

    RmsNormMatch rms = match_qwen_rmsnorm_f32(graph, rms_node, rms_node_index);
    if (!rms.matched() || rms.input->id != residual->id ||
        !has_low_row_symmetric_i4_consumer(graph, rms.output->id, rms.token_count, rms.hidden_size)) {
        return {};
    }
    if (!pairwise_distinct_storage(std::array<const Value *, 5>{ input, addend, residual, rms.weight, rms.output })) {
        return {};
    }

    match.rms            = rms;
    match.add_node       = node;
    match.input          = input;
    match.addend         = addend;
    match.residual       = residual;
    match.add_node_index = node_index;
    return match;
}

struct QwenEndpointProjectionMatch {
    const GraphNode * projection_node       = nullptr;
    const Value *     weight                = nullptr;
    const Value *     output                = nullptr;
    size_t            projection_node_index = 0;

    bool matched() const { return projection_node != nullptr && weight != nullptr && output != nullptr; }
};

static QwenEndpointProjectionMatch match_qwen_endpoint_projection(const Graph & graph, const RmsNormMatch & match) {
    QwenEndpointProjectionMatch projection_match;
    if (!graph.has_index()) {
        return projection_match;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(match.output->id);
    if (consumers.size() != 1) {
        return {};
    }
    const GraphNode * consumer = consumers.front();
    if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2) {
        return {};
    }
    size_t consumer_index = 0;
    if (!graph.index().node_index(consumer, consumer_index)) {
        return {};
    }
    const Value * weight = graph_value(graph, consumer->inputs[0]);
    const Value * input  = graph_value(graph, consumer->inputs[1]);
    const Value * output = graph_value(graph, consumer->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return {};
    }
    if (input->id != match.output->id || weight->type != GGML_TYPE_Q6_K || output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !input->contiguous || !output->contiguous || match.hidden_size != kQwenHiddenSize ||
        !is_supported_token_count(match.token_count) || weight->ne[0] != match.hidden_size ||
        weight->ne[1] != kQwenVocabularyCount || output->ne[0] != kQwenVocabularyCount ||
        output->ne[1] != match.token_count || output->ne[2] != 1 || output->ne[3] != 1) {
        return {};
    }

    projection_match.projection_node       = consumer;
    projection_match.projection_node_index = consumer_index;
    projection_match.weight                = weight;
    projection_match.output                = output;
    return projection_match;
}

static RmsNormMatch match_qwen_endpoint_rmsnorm_from_projection(const Graph & graph, const GraphNode * projection) {
    if (projection == nullptr || projection->op != GGML_OP_MUL_MAT || projection->inputs.size() != 2 ||
        !graph.has_index()) {
        return {};
    }

    const Value * input = graph_value(graph, projection->inputs[1]);
    if (input == nullptr) {
        return {};
    }
    const GraphNode * mul_node = graph.index().producer(input->id);
    size_t            mul_node_index;
    if (mul_node == nullptr || mul_node->op != GGML_OP_MUL || mul_node->inputs.size() != 2 ||
        !graph.index().node_index(mul_node, mul_node_index)) {
        return {};
    }

    for (const ValueId mul_input : mul_node->inputs) {
        const GraphNode * rms_node = graph.index().producer(mul_input);
        size_t            rms_node_index;
        if (rms_node != nullptr && rms_node->op == GGML_OP_RMS_NORM &&
            graph.index().node_index(rms_node, rms_node_index)) {
            return match_qwen_rmsnorm_f32(graph, rms_node, rms_node_index);
        }
    }
    return {};
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

}  // namespace

static bool match_qwen_add_rmsnorm_f32_i4_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const AddRmsNormMatch add_rms = match_qwen_add_rmsnorm_f32_i4(context.graph, context.root_node, context.root_index);
    if (!add_rms.matched() || add_rms.add_node_index >= context.covered_nodes.size() ||
        add_rms.rms.rms_node_index >= context.covered_nodes.size() ||
        add_rms.rms.mul_node_index >= context.covered_nodes.size() || context.covered_nodes[add_rms.add_node_index] ||
        context.covered_nodes[add_rms.rms.rms_node_index] || context.covered_nodes[add_rms.rms.mul_node_index]) {
        return false;
    }

    const LlmSymmetricI4ActivationLayout i4_layout =
        llm_symmetric_i4_activation_layout(add_rms.rms.hidden_size, add_rms.rms.token_count);
    const ValueId i4_value = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenAddRmsNormF32I4Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", add_rms.rms.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(add_rms.rms.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(add_rms.rms.token_count));
    dispatch.bindings.push_back({ add_rms.input->id, 0, add_rms.input->byte_count });
    dispatch.bindings.push_back({ add_rms.addend->id, 0, add_rms.addend->byte_count });
    dispatch.bindings.push_back({ add_rms.residual->id, 0, add_rms.residual->byte_count });
    dispatch.bindings.push_back({ add_rms.rms.weight->id, 0, add_rms.rms.weight->byte_count });
    dispatch.bindings.push_back({ add_rms.rms.output->id, 0, add_rms.rms.output->byte_count });
    dispatch.bindings.push_back({ i4_value, 0, i4_layout.payload_bytes });
    dispatch.bindings.push_back({ i4_value, i4_layout.scales_offset, i4_layout.metadata_bytes });
    dispatch.bindings.push_back({ i4_value, i4_layout.sums_offset, i4_layout.metadata_bytes });

    Status metadata_status;
    if (!match.metadata.append_alternate_value({ add_rms.rms.output->id, i4_value, GGML_TYPE_COUNT,
                                                 i4_layout.total_bytes, kLlmSymmetricI4ActivationAlternateName },
                                               metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(add_rms.add_node_index);
    match.covered_nodes.push_back(add_rms.rms.rms_node_index);
    match.covered_nodes.push_back(add_rms.rms.mul_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back({ i4_value, kLlmSymmetricI4ActivationAlternateName, i4_layout.total_bytes, 256 });
    return true;
}

static bool match_qwen_rmsnorm_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match =
        match_qwen_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.mul_node_index >= context.covered_nodes.size() || context.covered_nodes[rms_match.rms_node_index] ||
        context.covered_nodes[rms_match.mul_node_index]) {
        return false;
    }

    const bool publish_i4                  = has_low_row_symmetric_i4_consumer(context.graph, rms_match.output->id,
                                                                               rms_match.token_count, rms_match.hidden_size);
    const bool has_prefill_dense_consumers = has_prefill_f16_dense_consumers_only(
        context.graph, rms_match.output->id, rms_match.token_count, rms_match.hidden_size);
    const bool publish_q8 =
        !publish_i4 && (has_prefill_q5_dense_consumer(context.graph, rms_match.output->id, rms_match.token_count,
                                                      rms_match.hidden_size) ||
                        has_low_row_q8_dense_consumer(context.graph, rms_match.output->id, rms_match.token_count,
                                                      rms_match.hidden_size));
    const bool    publish_f16 = has_prefill_dense_consumers && !publish_q8 && !publish_i4;
    const size_t  f16_bytes   = publish_f16 ? f16_byte_count(rms_match.token_count, rms_match.hidden_size) : 0;
    const ValueId f16_value   = publish_f16 ? context.next_plan_value : ValueId{};
    const size_t  q8_bytes    = publish_q8 ? q8_1_x4_byte_count(rms_match.token_count, rms_match.hidden_size) : 0;
    const ValueId q8_value    = publish_q8 ? ValueId(context.next_plan_value.value + (publish_f16 ? 1 : 0)) : ValueId{};
    const LlmSymmetricI4ActivationLayout i4_layout =
        llm_symmetric_i4_activation_layout(rms_match.hidden_size, rms_match.token_count);
    const ValueId i4_value = publish_i4 ? context.next_plan_value : ValueId{};

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(publish_i4  ? kQwenRmsNormF32I4Kernel :
                                                 publish_f16 ? kQwenRmsNormF32F16Kernel :
                                                               kQwenRmsNormF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(rms_match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                               to_config_value(rms_match.q8_group_count));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.weight->id, 0, rms_match.weight->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });
    if (publish_f16) {
        dispatch.bindings.push_back({ f16_value, 0, f16_bytes });
        Status metadata_status;
        if (!match.metadata.append_alternate_value(
                { rms_match.output->id, f16_value, GGML_TYPE_F16, f16_bytes, "qwen.prefill.f16_hidden" },
                metadata_status)) {
            match.status.append(metadata_status);
            return false;
        }
        match.transients.push_back({ f16_value, "qwen.prefill.f16_hidden", f16_bytes, 256 });
    }
    if (publish_q8) {
        Status metadata_status;
        if (!match.metadata.append_alternate_value(
                { rms_match.output->id, q8_value, GGML_TYPE_Q8_1, q8_bytes, "qwen.prefill.q8_hidden" },
                metadata_status)) {
            match.status.append(metadata_status);
            return false;
        }
        match.transients.push_back({ q8_value, "qwen.prefill.q8_hidden", q8_bytes, 256 });
    }
    if (publish_i4) {
        dispatch.bindings.push_back({ i4_value, 0, i4_layout.payload_bytes });
        dispatch.bindings.push_back({ i4_value, i4_layout.scales_offset, i4_layout.metadata_bytes });
        dispatch.bindings.push_back({ i4_value, i4_layout.sums_offset, i4_layout.metadata_bytes });
        Status metadata_status;
        if (!match.metadata.append_alternate_value({ rms_match.output->id, i4_value, GGML_TYPE_COUNT,
                                                     i4_layout.total_bytes, kLlmSymmetricI4ActivationAlternateName },
                                                   metadata_status)) {
            match.status.append(metadata_status);
            return false;
        }
        match.transients.push_back({ i4_value, kLlmSymmetricI4ActivationAlternateName, i4_layout.total_bytes, 256 });
    }

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.mul_node_index);
    match.dispatches.push_back(std::move(dispatch));
    if (publish_q8) {
        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeQ8_1X4Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
        quantize.kernel.integer_parameters.emplace("input_size", rms_match.hidden_size);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                   to_config_value(rms_match.q8_group_count));
        quantize.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });
        quantize.bindings.push_back({ q8_value, 0, q8_bytes });
        match.dispatches.push_back(std::move(quantize));
    }
    return true;
}

static bool match_qwen_rmsnorm_f32_quantize_q8_1_x4_dispatch(const DispatchMatchContext & context,
                                                             DispatchMatch &              match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match =
        context.root_node->op == GGML_OP_MUL_MAT ?
            match_qwen_endpoint_rmsnorm_from_projection(context.graph, context.root_node) :
            match_qwen_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    const QwenEndpointProjectionMatch projection_match =
        rms_match.matched() ? match_qwen_endpoint_projection(context.graph, rms_match) : QwenEndpointProjectionMatch{};
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.mul_node_index >= context.covered_nodes.size() || context.covered_nodes[rms_match.rms_node_index] ||
        context.covered_nodes[rms_match.mul_node_index] || !projection_match.matched() ||
        projection_match.projection_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[projection_match.projection_node_index] ||
        (context.root_node->op == GGML_OP_MUL_MAT && projection_match.projection_node != context.root_node)) {
        return false;
    }

    const size_t q8_byte_count = q8_1_x4_byte_count(rms_match.token_count, rms_match.hidden_size);
    if (q8_byte_count == 0) {
        return false;
    }

    const ValueId q8_value = context.next_plan_value;

    Dispatch rms_dispatch;
    rms_dispatch.kernel = make_kernel_specialization(kQwenRmsNormF32QuantizeQ8_1X4Kernel);
    rms_dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    rms_dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size",
                                                   to_config_value(rms_match.hidden_size));
    rms_dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    rms_dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                                   to_config_value(rms_match.token_count));
    rms_dispatch.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                   to_config_value(rms_match.q8_group_count));
    rms_dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    rms_dispatch.bindings.push_back({ rms_match.weight->id, 0, rms_match.weight->byte_count });
    rms_dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });
    rms_dispatch.bindings.push_back({ q8_value, 0, q8_byte_count });

    Dispatch projection_dispatch;
    projection_dispatch.kernel = make_kernel_specialization(kGgmlLinearQ6KQ8_1X4Kernel);
    projection_dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    projection_dispatch.kernel.integer_parameters.emplace("input_size", rms_match.hidden_size);
    projection_dispatch.kernel.integer_parameters.emplace("output_size", kQwenVocabularyCount);
    projection_dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.token_capacity",
                                                          to_config_value(rms_match.token_count));
    projection_dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.output_capacity",
                                                          to_config_value(kQwenVocabularyCount));
    projection_dispatch.bindings.push_back({ q8_value, 0, q8_byte_count });
    projection_dispatch.bindings.push_back({ projection_match.weight->id, 0, projection_match.weight->byte_count });
    projection_dispatch.bindings.push_back({ projection_match.output->id, 0, projection_match.output->byte_count });

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.mul_node_index);
    match.covered_nodes.push_back(projection_match.projection_node_index);
    match.dispatches.push_back(std::move(rms_dispatch));
    match.dispatches.push_back(std::move(projection_dispatch));
    match.transients.push_back({ q8_value, "qwen.rmsnorm.q8_1_x4", q8_byte_count, 256 });
    return match.status.success();
}

static bool match_qwen_decode_rmsnorm_f32_quantize_q8_1_x4_dispatch(const DispatchMatchContext & context,
                                                                    DispatchMatch &              match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match =
        match_qwen_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.token_count != 1 || rms_match.hidden_size != kQwenHiddenSize ||
        rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.mul_node_index >= context.covered_nodes.size() || context.covered_nodes[rms_match.rms_node_index] ||
        context.covered_nodes[rms_match.mul_node_index]) {
        return false;
    }
    if (!has_decode_q8_consumer(context.graph, rms_match.output->id)) {
        return false;
    }

    const size_t q8_byte_count = q8_1_x4_byte_count(rms_match.token_count, rms_match.hidden_size);
    if (q8_byte_count == 0) {
        return false;
    }

    const ValueId q8_value = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRmsNormF32QuantizeQ8_1X4Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(rms_match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                               to_config_value(rms_match.q8_group_count));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.weight->id, 0, rms_match.weight->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });
    dispatch.bindings.push_back({ q8_value, 0, q8_byte_count });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { rms_match.output->id, q8_value, GGML_TYPE_Q8_1, q8_byte_count, "qwen.decode.q8_hidden" },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.mul_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back({ q8_value, "qwen.decode.q8_hidden", q8_byte_count, 256 });
    return match.status.success();
}

void register_qwen_rmsnorm_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.add_rmsnorm_f32_i4",
        GGML_OP_ADD,
        DispatchMatchKind::Fused,
        1050,
        DispatchSource::Qwen,
        match_qwen_add_rmsnorm_f32_i4_dispatch,
    });
    registry.add({
        "qwen.endpoint_rmsnorm_q6k_q8_1_x4",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        1200,
        DispatchSource::Qwen,
        match_qwen_rmsnorm_f32_quantize_q8_1_x4_dispatch,
    });
    registry.add({
        "qwen.decode_rmsnorm_f32_quantize_q8_1_x4",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1150,
        DispatchSource::Qwen,
        match_qwen_decode_rmsnorm_f32_quantize_q8_1_x4_dispatch,
    });
    registry.add({
        "qwen.rmsnorm_f32_quantize_q8_1_x4",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1100,
        DispatchSource::Qwen,
        match_qwen_rmsnorm_f32_quantize_q8_1_x4_dispatch,
    });
    registry.add({
        "qwen.rmsnorm_f32.mul_weight",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Qwen,
        match_qwen_rmsnorm_f32_dispatch,
    });
}

}  // namespace ggml::hrx
