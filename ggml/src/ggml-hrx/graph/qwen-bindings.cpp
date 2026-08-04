#include "qwen-bindings.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>

namespace ggml::hrx {
namespace {

static constexpr OperationId kFirstLayerOperation = 1;
static constexpr OperationId kRegularLayerOperationCount = 63;
static constexpr OperationId kTerminalLayerOperation = 2962;
// These are properties of the selected split-attention implementation, not
// model dimensions. Model and request geometry is recovered below.
static constexpr size_t kSplitAttentionKvTileSize = 64;
static constexpr size_t kSplitAttentionQueryCapacity = 16;
static constexpr size_t kExpertPartitionRouteTileSize = 32;

// Family-generic facts consumed by kernel recipes. The Qwen adapter below is
// only responsible for locating these facts in today's cgraph topology.
struct RoutedTransformerFacts {
    size_t layer_count = 0;
    size_t token_count = 0;
    size_t context_count = 0;
    size_t vocabulary_count = 0;
    size_t hidden_size = 0;
    size_t query_size = 0;
    size_t key_value_size = 0;
    size_t head_size = 0;
    size_t query_head_count = 0;
    size_t key_value_head_count = 0;
    size_t expert_count = 0;
    size_t route_count = 0;
    size_t route_stride = 0;
    size_t expert_intermediate_size = 0;
    std::string rms_epsilon;
};

class QwenBindingImplementation {
public:

static OperationId layer_start(size_t layer) {
    return layer < 47 ? kFirstLayerOperation + static_cast<OperationId>(layer * kRegularLayerOperationCount)
                      : kTerminalLayerOperation;
}

struct FactQuery {
    const Graph & graph;
    std::vector<std::string> & errors;

    const Operation * operation(OperationId id, const std::string & role) {
        if (id >= graph.operations.size()) {
            errors.push_back("cannot query " + role + ": operation " + std::to_string(id) + " is absent");
            return nullptr;
        }
        return &graph.operations[id];
    }

    const Value * value(ValueId id, const std::string & role) {
        if (id >= graph.values.size()) {
            errors.push_back("cannot query " + role + ": value " + std::to_string(id) + " is absent");
            return nullptr;
        }
        return &graph.values[id];
    }

    ValueId output(OperationId id, const std::string & role) {
        const Operation * op = operation(id, role);
        if (!op || !value(op->output, role + " output")) return kInvalidId;
        return op->output;
    }

    ValueId input(OperationId id, size_t index, const std::string & role) {
        const Operation * op = operation(id, role);
        if (!op) return kInvalidId;
        if (index >= op->inputs.size()) {
            errors.push_back("cannot query " + role + ": input " + std::to_string(index) + " is absent");
            return kInvalidId;
        }
        return value(op->inputs[index], role + " input") ? op->inputs[index] : kInvalidId;
    }

    size_t dimension(ValueId id, size_t axis, const std::string & role) {
        const Value * tensor = value(id, role);
        if (!tensor) return 0;
        if (axis >= tensor->access.shape.size() || tensor->access.shape[axis] <= 0) {
            errors.push_back("cannot query " + role + ": dimension " + std::to_string(axis) + " is absent or non-positive");
            return 0;
        }
        return static_cast<size_t>(tensor->access.shape[axis]);
    }

    size_t stride_elements(ValueId id, size_t axis, const std::string & role) {
        const Value * tensor = value(id, role);
        if (!tensor) return 0;
        if (axis >= tensor->access.strides.size()) {
            errors.push_back("cannot query " + role + ": stride " + std::to_string(axis) + " is absent");
            return 0;
        }
        const size_t type_size = ggml_type_size(tensor->type);
        const size_t byte_stride = tensor->access.strides[axis];
        if (type_size == 0 || byte_stride == 0 || byte_stride % type_size != 0) {
            errors.push_back("cannot query " + role + ": byte stride is not an integral element stride");
            return 0;
        }
        return byte_stride / type_size;
    }
};

static RoutedTransformerFacts recover_transformer_facts(const Graph & graph, const Schedule & schedule,
                                                         std::vector<std::string> & errors) {
    RoutedTransformerFacts facts;
    FactQuery query{graph, errors};
    facts.layer_count = schedule.invocations.size() >= 2 ? schedule.invocations.size() - 2 : 0;
    const OperationId start = layer_start(0);
    const ValueId embedding = query.output(0, "embedding");
    const ValueId embedding_weight = query.input(0, 0, "embedding");
    const ValueId query_projection = query.output(start + 2, "layer 0 query projection");
    const ValueId value_projection = query.output(start + 7, "layer 0 value projection");
    const ValueId query_heads = query.output(start + 19, "layer 0 query head view");
    const ValueId key_heads = query.output(start + 21, "layer 0 key head view");
    const ValueId attention_mask = query.input(start + 24, 3, "layer 0 attention mask");
    const ValueId router = query.output(start + 30, "layer 0 router");
    const ValueId route_ids = query.output(start + 34, "layer 0 route IDs");
    const ValueId expert_intermediate = query.output(start + 44, "layer 0 expert intermediate");
    facts.token_count = query.dimension(embedding, 1, "embedding output token count");
    facts.hidden_size = query.dimension(embedding, 0, "embedding output hidden size");
    facts.vocabulary_count = query.dimension(embedding_weight, 1, "embedding vocabulary");
    facts.query_size = query.dimension(query_projection, 0, "query projection width");
    facts.key_value_size = query.dimension(value_projection, 0, "value projection width");
    facts.head_size = query.dimension(query_heads, 0, "attention head width");
    facts.query_head_count = query.dimension(query_heads, 2, "query head count");
    facts.context_count = query.dimension(attention_mask, 0, "attention context extent");
    facts.key_value_head_count = query.dimension(key_heads, 2, "key/value head count");
    facts.expert_count = query.dimension(router, 0, "router expert count");
    facts.route_count = query.dimension(route_ids, 0, "logical route count");
    facts.route_stride = query.stride_elements(route_ids, 1, "physical route row stride");
    facts.expert_intermediate_size = query.dimension(expert_intermediate, 0, "expert intermediate width");
    const Operation * rmsnorm = query.operation(start, "layer 0 attention RMSNorm");
    if (rmsnorm) {
        float epsilon = 0.0f;
        static_assert(sizeof(epsilon) <= GGML_MAX_OP_PARAMS);
        std::memcpy(&epsilon, rmsnorm->raw_params.data(), sizeof(epsilon));
        if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
            errors.push_back("cannot query RMS epsilon: value is not finite and positive");
        } else {
            std::ostringstream out;
            out << std::setprecision(9) << epsilon;
            facts.rms_epsilon = out.str();
        }
    }
    if (!errors.empty()) return facts;

    auto require = [&](bool condition, const std::string & message) {
        if (!condition) errors.push_back("incoherent Qwen model facts: " + message);
    };
    require(facts.layer_count != 0 && facts.token_count != 0 && facts.context_count != 0, "empty program geometry");
    require(facts.hidden_size != 0 && facts.vocabulary_count != 0, "empty embedding geometry");
    require(facts.query_size == facts.query_head_count * facts.head_size, "query size/head geometry mismatch");
    require(facts.key_value_size == facts.key_value_head_count * facts.head_size, "key/value size/head geometry mismatch");
    require(facts.route_count != 0 && facts.route_stride >= facts.route_count, "invalid logical/physical route geometry");
    require(facts.expert_count != 0 && facts.expert_intermediate_size != 0, "empty expert geometry");
    require(query.dimension(query.output(start + 9, "layer 0 key projection"), 0, "key projection width") ==
                facts.key_value_size,
            "K/V projection size mismatch");
    require(query.dimension(query.output(3029, "endpoint logits"), 0, "endpoint vocabulary") == facts.vocabulary_count,
            "endpoint vocabulary mismatch");
    for (size_t layer = 0; layer < facts.layer_count; ++layer) {
        const OperationId layer_op = layer_start(layer);
        const int ff_shift = layer + 1 == facts.layer_count ? 2 : 0;
        const std::string prefix = "layer " + std::to_string(layer) + " ";
        require(query.dimension(query.output(layer_op + 2, prefix + "query projection"), 0,
                                prefix + "query projection width") == facts.query_size,
                "layer " + std::to_string(layer) + " query projection mismatch");
        const ValueId query_weight = query.input(layer_op + 2, 0, prefix + "query projection");
        require(query.dimension(query_weight, 0, prefix + "query weight input width") == facts.hidden_size &&
                query.dimension(query_weight, 1, prefix + "query weight output width") == facts.query_size,
                "layer " + std::to_string(layer) + " query weight geometry mismatch");
        require(query.dimension(query.output(layer_op + 7, prefix + "value projection"), 0,
                                prefix + "value projection width") == facts.key_value_size &&
                query.dimension(query.output(layer_op + 9, prefix + "key projection"), 0,
                                prefix + "key projection width") == facts.key_value_size,
                "layer " + std::to_string(layer) + " key/value projection mismatch");
        for (const auto & projection : { std::pair<OperationId, const char *>{layer_op + 7, "value"},
                                         std::pair<OperationId, const char *>{layer_op + 9, "key"} }) {
            const ValueId weight = query.input(projection.first, 0, prefix + projection.second + " projection");
            require(query.dimension(weight, 0, prefix + projection.second + " weight input width") == facts.hidden_size &&
                    query.dimension(weight, 1, prefix + projection.second + " weight output width") == facts.key_value_size,
                    "layer " + std::to_string(layer) + " " + projection.second + " weight geometry mismatch");
        }
        require(query.dimension(query.output(layer_op + 30 + ff_shift, prefix + "router"), 0,
                                prefix + "router expert count") == facts.expert_count,
                "layer " + std::to_string(layer) + " expert count mismatch");
        const ValueId router_weight = query.input(layer_op + 30 + ff_shift, 0, prefix + "router");
        require(query.dimension(router_weight, 0, prefix + "router weight input width") == facts.hidden_size &&
                query.dimension(router_weight, 1, prefix + "router weight expert count") == facts.expert_count,
                "layer " + std::to_string(layer) + " router weight geometry mismatch");
        const ValueId layer_routes = query.output(layer_op + 34 + ff_shift, prefix + "route IDs");
        require(query.dimension(layer_routes, 0, prefix + "logical route count") == facts.route_count &&
                query.stride_elements(layer_routes, 1, prefix + "physical route row stride") == facts.route_stride,
                "layer " + std::to_string(layer) + " route layout mismatch");
        require(query.dimension(query.output(layer_op + 44 + ff_shift, prefix + "expert intermediate"), 0,
                                prefix + "expert intermediate width") == facts.expert_intermediate_size,
                "layer " + std::to_string(layer) + " expert intermediate mismatch");
        for (const auto & expert : { std::pair<OperationId, const char *>{layer_op + 42 + ff_shift, "gate"},
                                     std::pair<OperationId, const char *>{layer_op + 43 + ff_shift, "up"} }) {
            const ValueId weight = query.input(expert.first, 0, prefix + expert.second + " expert projection");
            require(query.dimension(weight, 0, prefix + expert.second + " expert input width") == facts.hidden_size &&
                    query.dimension(weight, 1, prefix + expert.second + " expert output width") ==
                        facts.expert_intermediate_size &&
                    query.dimension(weight, 2, prefix + expert.second + " expert count") == facts.expert_count,
                    "layer " + std::to_string(layer) + " " + expert.second + " expert weight geometry mismatch");
        }
        const OperationId down = layer_op + (layer + 1 == facts.layer_count ? 47 : 45);
        const ValueId down_weight = query.input(down, 0, prefix + "down expert projection");
        require(query.dimension(down_weight, 0, prefix + "down expert input width") == facts.expert_intermediate_size &&
                query.dimension(down_weight, 1, prefix + "down expert output width") == facts.hidden_size &&
                query.dimension(down_weight, 2, prefix + "down expert count") == facts.expert_count,
                "layer " + std::to_string(layer) + " down expert weight geometry mismatch");
    }
    return facts;
}

static TensorBinding binding(const char * role, ValueId value, size_t offset = 0, size_t length = 0) {
    return { role, value, offset, length };
}

static ValueId append_scratch(Graph & graph, const std::string & name, size_t byte_length) {
    Storage storage;
    storage.id = static_cast<StorageId>(graph.storages.size());
    storage.root = static_cast<ValueId>(graph.values.size());
    storage.size = byte_length;
    graph.storages.push_back(storage);

    Value value;
    value.id = storage.root;
    value.type = GGML_TYPE_I8;
    value.op = GGML_OP_NONE;
    value.access.storage = storage.id;
    value.access.shape = { static_cast<int64_t>(byte_length), 1, 1, 1 };
    value.access.strides = { 1, byte_length, byte_length, byte_length };
    value.name = "hrx.synthetic." + name;
    graph.values.push_back(value);
    return value.id;
}

static size_t q8_row_bytes(size_t element_count) {
    return ggml_row_size(GGML_TYPE_Q8_1, element_count);
}

static ValueId op_input(const Graph & graph, OperationId operation, size_t index) {
    return graph.operations.at(operation).inputs.at(index);
}

static ValueId op_output(const Graph & graph, OperationId operation) {
    return graph.operations.at(operation).output;
}

static void runtime_scalar(Dispatch & dispatch, const char * name, int64_t value) {
    dispatch.kernel.integer_parameters[name] = value;
}

static void compile_config(Dispatch & dispatch, const char * name, const std::string & value) {
    dispatch.kernel.compile_parameters[name] = value;
}

static void rmsnorm_config(Dispatch & dispatch, const RoutedTransformerFacts & facts) {
    compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
    compile_config(dispatch, "qwen3_moe.model.rms_epsilon", facts.rms_epsilon);
}

struct Scratch {
    ValueId control = kInvalidId;
    ValueId inverse_frequencies = kInvalidId;
    ValueId q8_hidden = kInvalidId;
    ValueId q8_attention = kInvalidId;
    ValueId q8_swiglu = kInvalidId;
    ValueId expert_table = kInvalidId;
    ValueId partition_table = kInvalidId;
    ValueId partial_max = kInvalidId;
    ValueId partial_sum = kInvalidId;
    ValueId partial_output = kInvalidId;
    ValueId completion_counter = kInvalidId;
};

static void bind_preamble(const Graph & graph, Invocation & invocation, const Scratch & scratch,
                          const RoutedTransformerFacts & facts, std::vector<std::string> & errors) {
    const OperationId start = layer_start(0);
    for (Dispatch & dispatch : invocation.dispatches) {
        const std::string & variant = dispatch.kernel.variant;
        if (variant == "qwen_token_embedding_q4k_bringup_workaround") {
            dispatch.bindings = {
                binding("token_ids", op_input(graph, 0, 1)),
                binding("weight", op_input(graph, 0, 0)),
                binding("output", op_output(graph, 0)),
            };
        } else if (variant == "qwen_attention_metadata_bringup_workaround") {
            dispatch.bindings = {
                binding("control", scratch.control),
                binding("positions", op_input(graph, start + 6, 1)),
                binding("key_cache_indices", op_input(graph, start + 15, 1)),
                binding("value_cache_indices", op_input(graph, start + 17, 1)),
                binding("attention_mask", op_input(graph, start + 24, 3)),
            };
            runtime_scalar(dispatch, "context_capacity", facts.context_count);
        } else if (variant == "qwen3_moe_attention_rmsnorm_quantize_q8_1_x4") {
            dispatch.bindings = {
                binding("input", op_output(graph, 0)),
                binding("weight", op_input(graph, 2, 1)),
                binding("output", scratch.q8_hidden),
            };
            rmsnorm_config(dispatch, facts);
        } else {
            errors.push_back("unexpected Qwen preamble dispatch " + variant);
        }
        runtime_scalar(dispatch, "token_count", facts.token_count);
    }
}

static void bind_attention_common(const Graph & graph, Dispatch & dispatch, OperationId start,
                                  const Scratch & scratch, const RoutedTransformerFacts & facts) {
    const std::string & variant = dispatch.kernel.variant;
    if (variant == "qwen3_moe_attention_postprocess_f32_f16") {
        dispatch.bindings = {
            binding("positions", op_input(graph, start + 6, 1)),
            binding("key_cache_indices", op_input(graph, start + 15, 1)),
            binding("value_cache_indices", op_input(graph, start + 17, 1)),
            binding("query_input", op_output(graph, start + 2)),
            binding("key_input", op_output(graph, start + 9)),
            binding("value_input", op_output(graph, start + 7)),
            binding("query_norm_weight", op_input(graph, start + 5, 1)),
            binding("key_norm_weight", op_input(graph, start + 12, 1)),
            binding("inverse_frequencies", scratch.inverse_frequencies),
            binding("query_output", op_output(graph, start + 6)),
            binding("key_cache", op_input(graph, start + 15, 2)),
            binding("value_cache", op_input(graph, start + 17, 2)),
        };
        runtime_scalar(dispatch, "cache_row_count", facts.context_count);
        compile_config(dispatch, "qwen3_moe.attention.head_size", std::to_string(facts.head_size));
        compile_config(dispatch, "qwen3_moe.attention.key_value_size", std::to_string(facts.key_value_size));
        compile_config(dispatch, "qwen3_moe.attention.query_size", std::to_string(facts.query_size));
        compile_config(dispatch, "qwen3_moe.model.rms_epsilon", facts.rms_epsilon);
    } else if (variant == "qwen3_moe_flash_attention_f32_f16_wmma") {
        dispatch.bindings = {
            binding("query", op_output(graph, start + 19)),
            binding("key", op_output(graph, start + 21)),
            binding("value", op_output(graph, start + 24 - 1)),
            binding("mask", op_input(graph, start + 24, 3)),
            binding("output", op_output(graph, start + 24)),
        };
        runtime_scalar(dispatch, "query_token_count", facts.token_count);
        runtime_scalar(dispatch, "key_value_token_count", facts.context_count);
        compile_config(dispatch, "qwen3_moe.attention.key_value_head_count", std::to_string(facts.key_value_head_count));
        compile_config(dispatch, "qwen3_moe.attention.query_head_count", std::to_string(facts.query_head_count));
    } else if (variant == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma") {
        dispatch.bindings = {
            binding("query", op_output(graph, start + 19)),
            binding("key", op_output(graph, start + 21)),
            binding("value", op_output(graph, start + 23)),
            binding("mask", op_input(graph, start + 24, 3)),
            binding("partial_max", scratch.partial_max),
            binding("partial_sum", scratch.partial_sum),
            binding("partial_output", scratch.partial_output),
            binding("completion_counter", scratch.completion_counter),
            binding("output", op_output(graph, start + 24)),
        };
        runtime_scalar(dispatch, "key_value_token_count", facts.context_count);
        compile_config(dispatch, "qwen3_moe.attention.key_value_head_count", std::to_string(facts.key_value_head_count));
        compile_config(dispatch, "qwen3_moe.attention.query_head_count", std::to_string(facts.query_head_count));
    }
}

static void bind_prefill_layer(const Graph & graph, Invocation & invocation, const Scratch & scratch,
                               size_t layer, const RoutedTransformerFacts & facts,
                               ValueId hidden_state, std::vector<std::string> & errors) {
    const OperationId start = layer_start(layer);
    const bool terminal = layer + 1 == facts.layer_count;
    const size_t row_offset = terminal ? (facts.token_count - 1) * facts.hidden_size * sizeof(float) : 0;
    const size_t row_length = terminal ? facts.hidden_size * sizeof(float) : 0;
    const int ff_shift = terminal ? 2 : 0;
    const OperationId norm_mul = start + 29 + ff_shift;
    const OperationId router = start + 30 + ff_shift;
    const OperationId route_ids = start + 34 + ff_shift;
    const OperationId route_weights = start + 40 + ff_shift;
    const OperationId gate = start + 42 + ff_shift;
    const OperationId up = start + 43 + ff_shift;
    const OperationId swiglu = start + 44 + ff_shift;
    const OperationId down = start + (terminal ? 47 : 45);
    size_t dense_index = 0;
    size_t rmsnorm_index = 0;
    for (Dispatch & dispatch : invocation.dispatches) {
        const std::string & variant = dispatch.kernel.variant;
        if (variant == "qwen3_moe_rmsnorm_f32") {
            const size_t occurrence = rmsnorm_index++;
            if (layer == 0 && occurrence == 0) {
                dispatch.bindings = { binding("input", hidden_state), binding("weight", op_input(graph, start + 1, 1)),
                                      binding("output", op_output(graph, start + 1)) };
            } else if (!terminal && occurrence == (layer == 0 ? 2u : 1u)) {
                const OperationId next = layer_start(layer + 1);
                dispatch.bindings = { binding("input", hidden_state), binding("weight", op_input(graph, next + 1, 1)),
                                      binding("output", op_output(graph, next + 1)) };
            } else {
                dispatch.bindings = { binding("input", hidden_state, row_offset, row_length),
                                      binding("weight", op_input(graph, norm_mul, 1)),
                                      binding("output", op_output(graph, norm_mul)) };
            }
            rmsnorm_config(dispatch, facts);
        } else if (variant == "qwen3_moe_dense_linear_q4k_f16_wmma" ||
                   variant == "qwen3_moe_dense_linear_q6k_f16_wmma") {
            if (dense_index < 3) {
                static constexpr int projection_offsets[] = { 2, 7, 9 };
                const OperationId projection = start + projection_offsets[dense_index++];
                dispatch.bindings = { binding("input", op_input(graph, projection, 1)),
                                      binding("weight", op_input(graph, projection, 0)),
                                      binding("output", op_output(graph, projection)) };
                compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_size",
                               std::to_string(dense_index == 1 ? facts.query_size : facts.key_value_size));
            } else {
                ++dense_index;
                dispatch.bindings = { binding("input", op_output(graph, start + 25)),
                                      binding("weight", op_input(graph, start + 26, 0)),
                                      binding("output", hidden_state) };
                compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.query_size));
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "1");
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_size", std::to_string(facts.hidden_size));
            }
        } else if (variant == "qwen3_moe_attention_postprocess_f32_f16" ||
                   variant == "qwen3_moe_flash_attention_f32_f16_wmma") {
            bind_attention_common(graph, dispatch, start, scratch, facts);
        } else if (variant == "qwen3_moe_router_projection_f32_four_row_wave32") {
            dispatch.bindings = { binding("input", op_output(graph, norm_mul)), binding("weight", op_input(graph, router, 0)),
                                  binding("output", op_output(graph, router)) };
            compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
            compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
        } else if (variant == "qwen3_moe_router_top8_f32") {
            dispatch.bindings = { binding("logits", op_output(graph, router)), binding("route_ids", op_output(graph, route_ids)),
                                  binding("route_weights", op_output(graph, route_weights)) };
            runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
            compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
            compile_config(dispatch, "qwen3_moe.router.route_count", std::to_string(facts.route_count));
        } else if (variant == "qwen3_moe_build_expert_table") {
            dispatch.bindings = { binding("route_ids", op_output(graph, route_ids)), binding("expert_table", scratch.expert_table) };
            runtime_scalar(dispatch, "route_count", facts.route_count);
            runtime_scalar(dispatch, "route_stride", facts.route_stride);
            runtime_scalar(dispatch, "expert_count", facts.expert_count);
        } else if (variant == "qwen3_moe_build_expert_partition_table") {
            dispatch.bindings = { binding("expert_table", scratch.expert_table), binding("partition_table", scratch.partition_table) };
            runtime_scalar(dispatch, "route_count", facts.route_count);
            runtime_scalar(dispatch, "expert_count", facts.expert_count);
        } else if (variant == "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma") {
            dispatch.bindings = { binding("input", op_output(graph, norm_mul)), binding("expert_table", scratch.expert_table),
                                  binding("partition_table", scratch.partition_table), binding("gate_weight", op_input(graph, gate, 0)),
                                  binding("up_weight", op_input(graph, up, 0)), binding("output", op_output(graph, swiglu)) };
            compile_config(dispatch, "qwen3_moe.routed_gate_up.input_size", std::to_string(facts.hidden_size));
            compile_config(dispatch, "qwen3_moe.routed_gate_up.route_count", std::to_string(facts.route_count));
            compile_config(dispatch, "qwen3_moe.routed_gate_up.expert_count", std::to_string(facts.expert_count));
            compile_config(dispatch, "qwen3_moe.routed_gate_up.output_size",
                           std::to_string(facts.expert_intermediate_size));
        } else if (variant == "qwen3_moe_routed_down_q4k_f16_wmma_grouped" ||
                   variant == "qwen3_moe_routed_down_q6k_f16_wmma_grouped") {
            dispatch.bindings = { binding("input", op_output(graph, swiglu)), binding("expert_table", scratch.expert_table),
                                  binding("weight", op_input(graph, down, 0)), binding("output", op_output(graph, down)) };
            compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
            compile_config(dispatch, "qwen3_moe.routed_down.input_size", std::to_string(facts.expert_intermediate_size));
            compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
            compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
        } else if (variant == "qwen3_moe_routed_down_weighted_reduce_f16_f32") {
            dispatch.bindings = { binding("route_weights", op_output(graph, route_weights)),
                                  binding("routed_output", op_output(graph, down)),
                                  binding("output", hidden_state, row_offset, row_length) };
            compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
            compile_config(dispatch, "qwen3_moe.routed_down.input_size", std::to_string(facts.expert_intermediate_size));
            compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
            compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
        } else {
            errors.push_back("unexpected prefill layer dispatch " + variant);
        }
        runtime_scalar(dispatch, "token_count", dispatch.kernel.integer_parameters["token_count"]);
    }
}

static void bind_decode_layer(const Graph & graph, Invocation & invocation, const Scratch & scratch,
                              size_t layer, const RoutedTransformerFacts & facts, ValueId hidden_state,
                              std::vector<std::string> & errors) {
    const OperationId start = layer_start(layer);
    const bool terminal = layer + 1 == facts.layer_count;
    const int ff_shift = terminal ? 2 : 0;
    const OperationId norm_mul = start + 29 + ff_shift;
    const OperationId router = start + 30 + ff_shift;
    const OperationId route_ids = start + 34 + ff_shift;
    const OperationId route_weights = start + 40 + ff_shift;
    const OperationId gate = start + 42 + ff_shift;
    const OperationId up = start + 43 + ff_shift;
    const OperationId swiglu = start + 44 + ff_shift;
    const OperationId down = start + (terminal ? 47 : 45);
    size_t dual_norm_index = 0;
    for (Dispatch & dispatch : invocation.dispatches) {
        const std::string & variant = dispatch.kernel.variant;
        if (variant == "qwen3_moe_attention_qkv_quantized") {
            dispatch.bindings = { binding("q8_input", scratch.q8_hidden), binding("query_weight", op_input(graph, start + 2, 0)),
                                  binding("key_weight", op_input(graph, start + 9, 0)), binding("value_weight", op_input(graph, start + 7, 0)),
                                  binding("query_output", op_output(graph, start + 2)), binding("key_output", op_output(graph, start + 9)),
                                  binding("value_output", op_output(graph, start + 7)) };
            compile_config(dispatch, "qwen3_moe.attention.key_value_size", std::to_string(facts.key_value_size));
            compile_config(dispatch, "qwen3_moe.attention.query_size", std::to_string(facts.query_size));
            compile_config(dispatch, "qwen3_moe.attention.value_uses_q6",
                           dispatch.kernel.integer_parameters["value_weight_type"] == GGML_TYPE_Q6_K ? "1" : "0");
            compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.hidden_size));
            compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
            compile_config(dispatch, "qwen3_moe.dense_quantized.output_size", std::to_string(facts.key_value_size));
            rmsnorm_config(dispatch, facts);
        } else if (variant == "qwen3_moe_attention_postprocess_f32_f16" ||
                   variant == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma") {
            bind_attention_common(graph, dispatch, start, scratch, facts);
        } else if (variant == "ggml_quantize_q8_1_x4_f32") {
            if (dispatch.kernel.integer_parameters["token_count"] == 1) {
                dispatch.bindings = { binding("input", op_output(graph, start + 25)), binding("output", scratch.q8_attention) };
                runtime_scalar(dispatch, "input_size", facts.query_size);
            } else {
                dispatch.bindings = { binding("input", op_output(graph, swiglu)), binding("output", scratch.q8_swiglu) };
                runtime_scalar(dispatch, "input_size", facts.expert_intermediate_size);
            }
        } else if (variant == "qwen3_moe_dense_linear_q4k_q8_1_x4") {
            dispatch.bindings = { binding("q8_input", scratch.q8_attention), binding("weight", op_input(graph, start + 26, 0)),
                                  binding("output", hidden_state) };
            compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.query_size));
            compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "1");
            compile_config(dispatch, "qwen3_moe.dense_quantized.output_size", std::to_string(facts.hidden_size));
        } else if (variant == "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4") {
            const OperationId selected_mul = terminal && dual_norm_index++ == 1 ? 3028 : norm_mul;
            dispatch.bindings = { binding("input", hidden_state), binding("weight", op_input(graph, selected_mul, 1)),
                                  binding("normalized_output", op_output(graph, selected_mul)), binding("q8_output", scratch.q8_hidden) };
            rmsnorm_config(dispatch, facts);
        } else if (variant == "qwen3_moe_attention_rmsnorm_quantize_q8_1_x4") {
            const OperationId next = layer_start(layer + 1);
            dispatch.bindings = { binding("input", hidden_state), binding("weight", op_input(graph, next + 1, 1)),
                                  binding("output", scratch.q8_hidden) };
            rmsnorm_config(dispatch, facts);
        } else if (variant == "qwen3_moe_router_projection_f32_one_row_wave64") {
            dispatch.bindings = { binding("input", op_output(graph, norm_mul)), binding("weight", op_input(graph, router, 0)),
                                  binding("output", op_output(graph, router)) };
            compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
            compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
        } else if (variant == "qwen3_moe_router_top8_f32") {
            dispatch.bindings = { binding("logits", op_output(graph, router)), binding("route_ids", op_output(graph, route_ids)),
                                  binding("route_weights", op_output(graph, route_weights)) };
            runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
            compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
            compile_config(dispatch, "qwen3_moe.router.route_count", std::to_string(facts.route_count));
        } else if (variant == "qwen3_moe_routed_gate_up_swiglu_q4k_q8") {
            dispatch.bindings = { binding("q8_input", scratch.q8_hidden), binding("route_ids", op_output(graph, route_ids)),
                                  binding("gate_weight", op_input(graph, gate, 0)), binding("up_weight", op_input(graph, up, 0)),
                                  binding("output", op_output(graph, swiglu)) };
            runtime_scalar(dispatch, "route_count", facts.route_count);
            runtime_scalar(dispatch, "route_stride", facts.route_stride);
            runtime_scalar(dispatch, "expert_count", facts.expert_count);
            runtime_scalar(dispatch, "output_size", facts.expert_intermediate_size);
            compile_config(dispatch, "qwen3_moe.routed_gate_up.input_size", std::to_string(facts.hidden_size));
        } else if (variant == "qwen3_moe_routed_down_q4k_q8_1_x4" ||
                   variant == "qwen3_moe_routed_down_q6k_q8_1_x4") {
            dispatch.bindings = { binding("q8_input", scratch.q8_swiglu), binding("route_ids", op_output(graph, route_ids)),
                                  binding("route_weights", op_output(graph, route_weights)), binding("weight", op_input(graph, down, 0)),
                                  binding("output", hidden_state) };
            runtime_scalar(dispatch, "input_size", facts.expert_intermediate_size);
            runtime_scalar(dispatch, "route_count", facts.route_count);
            runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
            runtime_scalar(dispatch, "expert_count", facts.expert_count);
            runtime_scalar(dispatch, "output_size", facts.hidden_size);
        } else {
            errors.push_back("unexpected decode layer dispatch " + variant);
        }
        runtime_scalar(dispatch, "token_count", dispatch.kernel.integer_parameters["token_count"]);
    }
}

static void bind_endpoint(const Graph & graph, Invocation & invocation, const Scratch & scratch,
                          const RoutedTransformerFacts & facts, ValueId hidden_state, std::vector<std::string> & errors) {
    for (Dispatch & dispatch : invocation.dispatches) {
        const std::string & variant = dispatch.kernel.variant;
        if (variant == "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4") {
            const size_t offset = facts.token_count == 1 ? 0 : (facts.token_count - 1) * facts.hidden_size * sizeof(float);
            dispatch.bindings = { binding("input", hidden_state, offset, facts.hidden_size * sizeof(float)),
                                  binding("weight", op_input(graph, 3028, 1)), binding("normalized_output", op_output(graph, 3028)),
                                  binding("q8_output", scratch.q8_hidden) };
            runtime_scalar(dispatch, "token_count", 1);
            rmsnorm_config(dispatch, facts);
        } else if (variant == "ggml_linear_q6k_q8_1_x4") {
            dispatch.bindings = { binding("q8_input", scratch.q8_hidden), binding("weight", op_input(graph, 3029, 0)),
                                  binding("output", op_output(graph, 3029)) };
            runtime_scalar(dispatch, "token_count", 1);
            runtime_scalar(dispatch, "input_size", facts.hidden_size);
            runtime_scalar(dispatch, "output_size", facts.vocabulary_count);
        } else {
            errors.push_back("unexpected Qwen endpoint dispatch " + variant);
        }
    }
}

};

} // namespace

VerificationResult QwenProgramProof::materialize_dispatch_bindings(Graph & graph, Schedule & schedule) {
    VerificationResult result;
    if (schedule.invocations.size() != 50 || graph.operations.size() != 3030) {
        result.errors.push_back("cannot materialize bindings for a noncanonical Qwen program");
        return result;
    }
    RoutedTransformerFacts facts = QwenBindingImplementation::recover_transformer_facts(graph, schedule, result.errors);
    if (!result.errors.empty()) return result;
    const bool prefill = schedule.workload.rfind("prefill-", 0) == 0;
    const bool decode = schedule.workload.rfind("decode-", 0) == 0;
    if (!prefill && !decode) {
        result.errors.push_back("cannot materialize bindings for unsupported workload " + schedule.workload);
        return result;
    }
    const std::string expected_workload = prefill ? "prefill-" + std::to_string(facts.token_count)
                                                  : "decode-" + std::to_string(facts.context_count);
    if (schedule.workload != expected_workload || (decode && facts.token_count != 1)) {
        result.errors.push_back("schedule workload does not agree with recovered graph geometry");
        return result;
    }
    const size_t kv_blocks = (facts.context_count + kSplitAttentionKvTileSize - 1) / kSplitAttentionKvTileSize;
    QwenBindingImplementation::Scratch scratch;
    scratch.control = QwenBindingImplementation::append_scratch(graph, "request_control", sizeof(int32_t));
    scratch.inverse_frequencies = QwenBindingImplementation::append_scratch(graph, "inverse_frequencies", (facts.head_size / 2) * sizeof(float));
    scratch.q8_hidden = QwenBindingImplementation::append_scratch(graph, "q8_hidden", facts.token_count * QwenBindingImplementation::q8_row_bytes(facts.hidden_size));
    if (decode) {
        scratch.q8_attention = QwenBindingImplementation::append_scratch(graph, "q8_attention", QwenBindingImplementation::q8_row_bytes(facts.query_size));
        scratch.q8_swiglu = QwenBindingImplementation::append_scratch(graph, "q8_swiglu",
                                           facts.token_count * facts.route_count * QwenBindingImplementation::q8_row_bytes(facts.expert_intermediate_size));
        scratch.partial_max = QwenBindingImplementation::append_scratch(graph, "attention_partial_max",
                                             facts.key_value_head_count * kv_blocks * kSplitAttentionQueryCapacity * sizeof(float));
        scratch.partial_sum = QwenBindingImplementation::append_scratch(graph, "attention_partial_sum",
                                             facts.key_value_head_count * kv_blocks * kSplitAttentionQueryCapacity * sizeof(float));
        scratch.partial_output = QwenBindingImplementation::append_scratch(graph, "attention_partial_output",
                                                facts.key_value_head_count * kv_blocks * kSplitAttentionQueryCapacity *
                                                    facts.head_size * sizeof(uint16_t));
        scratch.completion_counter = QwenBindingImplementation::append_scratch(graph, "attention_completion_counter",
                                                    facts.key_value_head_count * sizeof(int32_t));
    } else {
        scratch.expert_table = QwenBindingImplementation::append_scratch(graph, "expert_table",
                                              (facts.expert_count + facts.expert_count * facts.token_count) * sizeof(int32_t));
        const size_t route_tiles = (facts.token_count * facts.route_count + kExpertPartitionRouteTileSize - 1) /
                                   kExpertPartitionRouteTileSize;
        scratch.partition_table = QwenBindingImplementation::append_scratch(graph, "partition_table",
                                                 (1 + facts.expert_count + route_tiles) * sizeof(int32_t));
    }

    const ValueId hidden_state = QwenBindingImplementation::op_output(graph, 0);
    QwenBindingImplementation::bind_preamble(graph, schedule.invocations[0], scratch, facts, result.errors);
    for (size_t layer = 0; layer < facts.layer_count; ++layer) {
        if (prefill) QwenBindingImplementation::bind_prefill_layer(graph, schedule.invocations[layer + 1], scratch, layer, facts,
                                       hidden_state, result.errors);
        else QwenBindingImplementation::bind_decode_layer(graph, schedule.invocations[layer + 1], scratch, layer, facts,
                               hidden_state, result.errors);
    }
    QwenBindingImplementation::bind_endpoint(graph, schedule.invocations.back(), scratch, facts, hidden_state, result.errors);
    return result;
}

} // namespace ggml::hrx
