#include "dispatch-add.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kAddF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_add_f32");
static constexpr KernelCatalogRef kScaleBiasF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_scale_bias_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool distinct_storage(const Value & lhs, const Value & rhs) {
    return lhs.storage != rhs.storage;
}

static std::string float_config_value(float value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
}

static bool scale_storage_is_safe(const Value & input, const Value & output) {
    if (input.storage_root != output.storage_root) {
        return true;
    }
    if (input.storage_offset == output.storage_offset && input.byte_count == output.byte_count) {
        return true;
    }
    if (input.storage_offset > std::numeric_limits<size_t>::max() - input.byte_count ||
        output.storage_offset > std::numeric_limits<size_t>::max() - output.byte_count) {
        return false;
    }
    return input.storage_offset + input.byte_count <= output.storage_offset ||
           output.storage_offset + output.byte_count <= input.storage_offset;
}

static bool supports_add_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_ADD || node->inputs.size() != 2) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * a      = graph_value(graph, node->inputs[0]);
    const Value * b      = graph_value(graph, node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }
    return output->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           same_shape(*output, *a) && same_shape(*output, *b) && output->contiguous && a->contiguous && b->contiguous &&
           distinct_storage(*output, *a) && distinct_storage(*output, *b) && output->element_count > 0 &&
           static_cast<uint64_t>(output->element_count) <= std::numeric_limits<uint32_t>::max();
}

static bool match_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!supports_add_f32_dispatch(context.graph, context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    const Value * a      = graph_value(context.graph, context.root_node->inputs[0]);
    const Value * b      = graph_value(context.graph, context.root_node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ a->id, 0, a->byte_count });
    dispatch.bindings.push_back({ b->id, 0, b->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_scale_bias_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SCALE || node->inputs.size() != 1) {
        return false;
    }
    const ScaleParams * params = op_params_as<ScaleParams>(node->params);
    const Value *       input  = graph_value(context.graph, node->inputs[0]);
    const Value *       output = graph_value(context.graph, node->output);
    if (params == nullptr || !std::isfinite(params->scale) || !std::isfinite(params->bias) || input == nullptr ||
        output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !input->contiguous ||
        !output->contiguous || !same_shape(*input, *output) || !scale_storage_is_safe(*input, *output) ||
        output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleBiasF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale.scale", float_config_value(params->scale));
    dispatch.kernel.compile_parameters.emplace("ggml.scale.bias", float_config_value(params->bias));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_add_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.scale_bias_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_scale_bias_f32_dispatch,
    });
    registry.add({
        "common.add_f32",
        GGML_OP_ADD,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_add_f32_dispatch,
    });
}

}  // namespace ggml::hrx
