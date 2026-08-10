#include "transitional-graph-index.h"
#include "transitional-llm-schedule.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace ggml::hrx {
namespace {

static constexpr size_t kSplitAttentionKvTileSize     = 64;
static constexpr size_t kSplitAttentionQueryCapacity  = 16;
static constexpr size_t kExpertPartitionRouteTileSize = 32;
static constexpr size_t kDecodeCompletionAlignment    = 16;
static constexpr size_t kGateUpCompletionTileSize     = 128;

struct Facts {
    size_t      layer_count              = 0;
    size_t      token_count              = 0;
    size_t      output_count             = 0;
    size_t      context_count            = 0;
    size_t      vocabulary_count         = 0;
    size_t      hidden_size              = 0;
    size_t      query_size               = 0;
    size_t      key_value_size           = 0;
    size_t      head_size                = 0;
    size_t      query_head_count         = 0;
    size_t      key_value_head_count     = 0;
    size_t      expert_count             = 0;
    size_t      route_count              = 0;
    size_t      route_stride             = 0;
    size_t      expert_intermediate_size = 0;
    std::string rms_epsilon;
};

struct Scratch {
    ValueId control                    = kInvalidId;
    ValueId inverse_frequencies        = kInvalidId;
    ValueId q8_hidden                  = kInvalidId;
    ValueId q8_attention               = kInvalidId;
    ValueId q8_swiglu                  = kInvalidId;
    ValueId expert_table               = kInvalidId;
    ValueId partition_table            = kInvalidId;
    ValueId partial_max                = kInvalidId;
    ValueId partial_sum                = kInvalidId;
    ValueId partial_output             = kInvalidId;
    ValueId prefill_completion_counter = kInvalidId;

    // Aliased counters used by sequential fused decode stages.
    struct {
        // Storage containing every counter initialized by one command.
        ValueId storage               = kInvalidId;
        // Byte length shared by Q/K/V publication and gate/up packing.
        size_t  grouped_byte_length   = 0;
        // Byte offset of the split-attention counter span.
        size_t  attention_byte_offset = 0;
        // Byte length of the split-attention counter span.
        size_t  attention_byte_length = 0;
        // Byte offset of the single counter shared by fused publication stages.
        size_t  shared_byte_offset    = 0;
        // Total initialized byte length, including alignment padding.
        size_t  total_byte_length     = 0;
    } completion;
};

class RoutedTransformerBindingImplementation {
  public:
    static ValueId op_input(const Graph & graph, OperationId operation_id, size_t index) {
        return graph.operations.at(operation_id).inputs.at(index);
    }

    static ValueId op_output(const Graph & graph, OperationId operation_id) {
        return graph.operations.at(operation_id).output;
    }

    static OperationId producer(const Graph & graph, ValueId value_id) {
        return value_id < graph.values.size() ? graph.values[value_id].producer : kInvalidId;
    }

    static OperationId primary_ancestor(const Graph & graph, ValueId value_id, enum ggml_op kind) {
        while (value_id < graph.values.size()) {
            const OperationId operation_id = producer(graph, value_id);
            if (operation_id == kInvalidId) {
                break;
            }
            const Operation & op = graph.operations[operation_id];
            if (op.op == kind) {
                return operation_id;
            }
            if (op.inputs.empty()) {
                break;
            }
            value_id = op.inputs[0];
        }
        return kInvalidId;
    }

    static TensorBinding binding(const char * role, ValueId value_id, size_t offset = 0, size_t length = 0) {
        return { role, value_id, offset, length };
    }

    static void runtime_scalar(Dispatch & dispatch, const char * name, int64_t scalar) {
        dispatch.kernel.integer_parameters[name] = scalar;
    }

    static void compile_config(Dispatch & dispatch, const char * name, const std::string & config) {
        dispatch.kernel.compile_parameters[name] = config;
    }

    static void rmsnorm_config(Dispatch & dispatch, const Facts & facts) {
        compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
        compile_config(dispatch, "qwen3_moe.model.rms_epsilon", facts.rms_epsilon);
    }

    static void routed_gate_up_config(Dispatch & dispatch, const Facts & facts) {
        compile_config(dispatch, "qwen3_moe.routed_gate_up.input_size", std::to_string(facts.hidden_size));
        compile_config(dispatch, "qwen3_moe.routed_gate_up.route_count", std::to_string(facts.route_count));
        compile_config(dispatch, "qwen3_moe.routed_gate_up.expert_count", std::to_string(facts.expert_count));
        compile_config(dispatch, "qwen3_moe.routed_gate_up.output_size",
                       std::to_string(facts.expert_intermediate_size));
    }

    static void routed_down_config(Dispatch & dispatch, const Facts & facts) {
        compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
        compile_config(dispatch, "qwen3_moe.routed_down.input_size", std::to_string(facts.expert_intermediate_size));
        compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
        compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
    }

    static ValueId append_scratch(Graph & graph, const std::string & name, size_t byte_length) {
        Storage storage;
        storage.id   = static_cast<StorageId>(graph.storages.size());
        storage.root = static_cast<ValueId>(graph.values.size());
        storage.size = byte_length;
        graph.storages.push_back(storage);
        Value scratch;
        scratch.id             = storage.root;
        scratch.type           = GGML_TYPE_I8;
        scratch.op             = GGML_OP_NONE;
        scratch.access.storage = storage.id;
        scratch.access.shape   = { static_cast<int64_t>(byte_length), 1, 1, 1 };
        scratch.access.strides = { 1, byte_length, byte_length, byte_length };
        scratch.name           = "hrx.synthetic." + name;
        graph.values.push_back(scratch);
        return scratch.id;
    }

    static size_t q8_row_bytes(size_t element_count) { return ggml_row_size(GGML_TYPE_Q8_1, element_count); }

    static size_t align_up(size_t value, size_t alignment) { return (value + alignment - 1) / alignment * alignment; }

    static Facts recover_facts(const Graph &                  graph,
                               const RoutedTransformerModel & model,
                               std::vector<std::string> &     errors) {
        Facts facts;
        facts.layer_count           = model.blocks.size();
        facts.token_count           = model.query_token_count;
        facts.output_count          = model.output_token_count;
        facts.context_count         = model.key_value_token_count;
        facts.hidden_size           = model.hidden_size;
        facts.query_size            = model.query_size;
        facts.key_value_size        = model.key_value_size;
        facts.expert_count          = model.expert_count;
        facts.route_count           = model.route_count;
        const OperationId embedding = model.operations_by_role.program_embedding;
        if (embedding == kInvalidId || graph.operations[embedding].inputs.empty()) {
            errors.push_back("routed-transformer binding facts have no embedding role");
            return facts;
        }
        facts.vocabulary_count               = graph.values[op_input(graph, embedding, 0)].access.shape[1];
        const RoutedTransformerBlock & first = model.blocks.front();
        const Operation &              flash = graph.operations[first.operations_by_role.attention_flash];
        if (flash.inputs.size() < 3) {
            errors.push_back("routed-transformer binding facts have an invalid flash-attention role");
            return facts;
        }
        const Value & query           = graph.values[flash.inputs[0]];
        const Value & key             = graph.values[flash.inputs[1]];
        facts.head_size               = query.access.shape[0];
        facts.query_head_count        = query.access.shape[2];
        facts.key_value_head_count    = key.access.shape[2];
        const Value & route_ids       = graph.values[first.values_by_role.router_route_ids];
        const size_t  route_type_size = ggml_type_size(route_ids.type);
        if (route_type_size == 0 || route_ids.access.strides[1] % route_type_size != 0) {
            errors.push_back("route ID stride is not an integral element stride");
            return facts;
        }
        facts.route_stride             = route_ids.access.strides[1] / route_type_size;
        facts.expert_intermediate_size = graph.values[first.values_by_role.experts_activation].access.shape[0];
        const Operation & norm         = graph.operations[first.operations_by_role.attention_norm];
        float             epsilon      = 0.0f;
        std::memcpy(&epsilon, norm.raw_params.data(), sizeof(epsilon));
        if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
            errors.push_back("RMS epsilon is not finite and positive");
            return facts;
        }
        std::ostringstream epsilon_text;
        epsilon_text << std::setprecision(9) << epsilon;
        facts.rms_epsilon = epsilon_text.str();

        auto require = [&](bool condition, const std::string & message) {
            if (!condition) {
                errors.push_back("incoherent routed-transformer facts: " + message);
            }
        };
        require(facts.query_size == facts.query_head_count * facts.head_size, "query head geometry mismatch");
        require(facts.key_value_size == facts.key_value_head_count * facts.head_size,
                "key/value head geometry mismatch");
        require(facts.route_count != 0 && facts.route_stride >= facts.route_count, "route layout mismatch");
        require(facts.output_count != 0 && facts.output_count <= facts.token_count,
                "output token count exceeds query token count");
        for (const RoutedTransformerBlock & block : model.blocks) {
            const std::string prefix           = "block " + std::to_string(block.ordinal) + ' ';
            const OperationId query_projection = block.operations_by_role.attention_query_projection;
            const OperationId key_projection   = block.operations_by_role.attention_key_projection;
            const OperationId value_projection = block.operations_by_role.attention_value_projection;
            require(graph.values[op_output(graph, query_projection)].access.shape[0] ==
                        static_cast<int64_t>(facts.query_size),
                    prefix + "query projection mismatch");
            require(graph.values[op_output(graph, key_projection)].access.shape[0] ==
                            static_cast<int64_t>(facts.key_value_size) &&
                        graph.values[op_output(graph, value_projection)].access.shape[0] ==
                            static_cast<int64_t>(facts.key_value_size),
                    prefix + "key/value projection mismatch");
            require(graph.values[block.values_by_role.router_route_ids].access.shape[0] ==
                        static_cast<int64_t>(facts.route_count),
                    prefix + "route count mismatch");
            const Value & block_route_ids       = graph.values[block.values_by_role.router_route_ids];
            const size_t  block_route_type_size = ggml_type_size(block_route_ids.type);
            require(block_route_type_size != 0 && block_route_ids.access.strides[1] % block_route_type_size == 0 &&
                        block_route_ids.access.strides[1] / block_route_type_size == facts.route_stride,
                    prefix + "route layout mismatch");
            require(graph.values[block.values_by_role.experts_activation].access.shape[0] ==
                        static_cast<int64_t>(facts.expert_intermediate_size),
                    prefix + "expert width mismatch");
        }
        return facts;
    }

    static Scratch allocate_scratch(Graph & graph, const Facts & facts, bool decode) {
        Scratch scratch;
        scratch.inverse_frequencies =
            append_scratch(graph, "inverse_frequencies", (facts.head_size / 2) * sizeof(float));
        scratch.q8_hidden = append_scratch(graph, "q8_hidden", facts.token_count * q8_row_bytes(facts.hidden_size));
        if (decode) {
            const size_t kv_blocks = (facts.context_count + kSplitAttentionKvTileSize - 1) / kSplitAttentionKvTileSize;
            scratch.q8_attention   = append_scratch(graph, "q8_attention", q8_row_bytes(facts.query_size));
            scratch.q8_swiglu =
                append_scratch(graph, "q8_swiglu",
                               facts.token_count * facts.route_count * q8_row_bytes(facts.expert_intermediate_size));
            scratch.partial_max =
                append_scratch(graph, "attention_partial_max",
                               facts.key_value_head_count * kv_blocks * kSplitAttentionQueryCapacity * sizeof(float));
            scratch.partial_sum =
                append_scratch(graph, "attention_partial_sum",
                               facts.key_value_head_count * kv_blocks * kSplitAttentionQueryCapacity * sizeof(float));
            scratch.partial_output =
                append_scratch(graph, "attention_partial_output",
                               facts.key_value_head_count * kv_blocks * kSplitAttentionQueryCapacity * facts.head_size *
                                   sizeof(uint16_t));
            const size_t qkv_counter_count = facts.query_head_count + 2 * facts.key_value_head_count;
            const size_t gate_up_counter_count =
                facts.route_count *
                ((facts.expert_intermediate_size + kGateUpCompletionTileSize - 1) / kGateUpCompletionTileSize);
            const size_t grouped_counter_count = std::max(qkv_counter_count, gate_up_counter_count);
            scratch.completion.grouped_byte_length =
                align_up(grouped_counter_count * sizeof(int32_t), kDecodeCompletionAlignment);
            scratch.completion.attention_byte_offset = scratch.completion.grouped_byte_length;
            scratch.completion.attention_byte_length =
                align_up(facts.key_value_head_count * sizeof(int32_t), kDecodeCompletionAlignment);
            scratch.completion.shared_byte_offset =
                scratch.completion.attention_byte_offset + scratch.completion.attention_byte_length;
            scratch.completion.total_byte_length =
                align_up(scratch.completion.shared_byte_offset + sizeof(int32_t), kDecodeCompletionAlignment);
            scratch.completion.storage =
                append_scratch(graph, "decode_completion_counters", scratch.completion.total_byte_length);
        } else {
            scratch.control      = append_scratch(graph, "request_control", sizeof(int32_t));
            scratch.expert_table = append_scratch(
                graph, "expert_table", (facts.expert_count + facts.expert_count * facts.token_count) * sizeof(int32_t));
            const size_t route_tiles = (facts.token_count * facts.route_count + kExpertPartitionRouteTileSize - 1) /
                                       kExpertPartitionRouteTileSize;
            scratch.partition_table =
                append_scratch(graph, "partition_table", (1 + facts.expert_count + route_tiles) * sizeof(int32_t));
            if (facts.token_count == 512) {
                scratch.prefill_completion_counter =
                    append_scratch(graph, "expert_table_completion_counter", sizeof(int32_t));
            }
        }
        return scratch;
    }

    static void bind_preamble(const Graph &                  graph,
                              const RoutedTransformerModel & model,
                              Invocation &                   invocation,
                              const Scratch &                scratch,
                              const Facts &                  facts,
                              std::vector<std::string> &     errors) {
        const OperationId              embedding    = model.operations_by_role.program_embedding;
        const RoutedTransformerBlock & first        = model.blocks.front();
        const OperationId              query_rope   = first.operations_by_role.attention_query_rope;
        const OperationId              key_writer   = first.operations_by_role.attention_key_cache_writer;
        const OperationId              value_writer = first.operations_by_role.attention_value_cache_writer;
        const OperationId              flash        = first.operations_by_role.attention_flash;
        const OperationId              prepared     = first.operations_by_role.attention_prepared;
        for (Dispatch & dispatch : invocation.dispatches) {
            const std::string & variant = dispatch.kernel.variant;
            if (variant == "qwen_token_embedding_q4k_bringup_workaround") {
                dispatch.bindings = { binding("token_ids", op_input(graph, embedding, 1)),
                                      binding("weight", op_input(graph, embedding, 0)),
                                      binding("output", op_output(graph, embedding)) };
            } else if (variant == "qwen_attention_context_base_capture") {
                dispatch.bindings = { binding("positions", op_input(graph, query_rope, 1)),
                                      binding("control", scratch.control) };
            } else if (variant == "qwen_attention_decode_state_initialize") {
                dispatch.bindings = {
                    binding("positions", op_input(graph, query_rope, 1)),
                    binding("key_cache_indices", op_input(graph, key_writer, 1)),
                    binding("value_cache_indices", op_input(graph, value_writer, 1)),
                    binding("attention_mask", op_input(graph, flash, 3)),
                    binding("completion_counters", scratch.completion.storage, 0, scratch.completion.total_byte_length),
                };
                runtime_scalar(dispatch, "context_capacity", facts.context_count);
                runtime_scalar(dispatch, "completion_counter_count",
                               scratch.completion.total_byte_length / sizeof(int32_t));
            } else if (variant == "qwen_attention_metadata_bringup_workaround") {
                dispatch.bindings = { binding("control", scratch.control),
                                      binding("positions", op_input(graph, query_rope, 1)),
                                      binding("key_cache_indices", op_input(graph, key_writer, 1)),
                                      binding("value_cache_indices", op_input(graph, value_writer, 1)),
                                      binding("attention_mask", op_input(graph, flash, 3)) };
                runtime_scalar(dispatch, "context_capacity", facts.context_count);
            } else if (variant == "qwen3_moe_attention_rmsnorm_quantize_q8_1_x4") {
                dispatch.bindings = { binding("input", op_output(graph, embedding)),
                                      binding("weight", op_input(graph, prepared, 1)),
                                      binding("output", scratch.q8_hidden) };
                rmsnorm_config(dispatch, facts);
            } else {
                errors.push_back("unexpected routed-transformer preamble dispatch " + variant);
            }
            if (variant != "qwen_attention_context_base_capture" &&
                variant != "qwen_attention_decode_state_initialize") {
                runtime_scalar(dispatch, "token_count", facts.token_count);
            }
            if (variant.rfind("qwen3_moe_", 0) == 0) {
                compile_config(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(facts.token_count));
            }
        }
    }

    static void bind_attention_common(const Graph &                  graph,
                                      const RoutedTransformerBlock & block,
                                      Dispatch &                     dispatch,
                                      const Scratch &                scratch,
                                      const Facts &                  facts) {
        const OperationId query_projection = block.operations_by_role.attention_query_projection;
        const OperationId key_projection   = block.operations_by_role.attention_key_projection;
        const OperationId value_projection = block.operations_by_role.attention_value_projection;
        const OperationId query_rope       = block.operations_by_role.attention_query_rope;
        const OperationId key_writer       = block.operations_by_role.attention_key_cache_writer;
        const OperationId value_writer     = block.operations_by_role.attention_value_cache_writer;
        const OperationId flash_id         = block.operations_by_role.attention_flash;
        const Operation & flash            = graph.operations[flash_id];
        if (dispatch.kernel.variant == "qwen3_moe_attention_postprocess_f32_f16") {
            const OperationId query_scale = producer(graph, op_input(graph, query_rope, 0));
            const OperationId key_rope    = primary_ancestor(graph, op_input(graph, key_writer, 0), GGML_OP_ROPE);
            const OperationId key_scale =
                key_rope == kInvalidId ? kInvalidId : producer(graph, op_input(graph, key_rope, 0));
            dispatch.bindings = {
                binding("positions", op_input(graph, query_rope, 1)),
                binding("key_cache_indices", op_input(graph, key_writer, 1)),
                binding("value_cache_indices", op_input(graph, value_writer, 1)),
                binding("query_input", op_output(graph, query_projection)),
                binding("key_input", op_output(graph, key_projection)),
                binding("value_input", op_output(graph, value_projection)),
                binding("query_norm_weight", op_input(graph, query_scale, 1)),
                binding("key_norm_weight", op_input(graph, key_scale, 1)),
                binding("inverse_frequencies", scratch.inverse_frequencies),
                binding("query_output", op_output(graph, query_rope)),
                binding("key_cache", op_input(graph, key_writer, 2)),
                binding("value_cache", op_input(graph, value_writer, 2)),
            };
            runtime_scalar(dispatch, "cache_row_count", facts.context_count);
            compile_config(dispatch, "qwen3_moe.attention.head_size", std::to_string(facts.head_size));
            compile_config(dispatch, "qwen3_moe.attention.key_value_size", std::to_string(facts.key_value_size));
            compile_config(dispatch, "qwen3_moe.attention.query_size", std::to_string(facts.query_size));
            compile_config(dispatch, "qwen3_moe.model.rms_epsilon", facts.rms_epsilon);
        } else if (dispatch.kernel.variant == "qwen3_moe_flash_attention_f32_f16_wmma") {
            dispatch.bindings = { binding("query", flash.inputs[0]), binding("key", flash.inputs[1]),
                                  binding("value", flash.inputs[2]), binding("mask", flash.inputs[3]),
                                  binding("output", flash.output) };
            runtime_scalar(dispatch, "query_token_count", facts.token_count);
            runtime_scalar(dispatch, "key_value_token_count", facts.context_count);
            compile_config(dispatch, "qwen3_moe.attention.key_value_head_count",
                           std::to_string(facts.key_value_head_count));
            compile_config(dispatch, "qwen3_moe.attention.query_head_count", std::to_string(facts.query_head_count));
        } else if (dispatch.kernel.variant == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma") {
            dispatch.bindings = { binding("query", flash.inputs[0]),
                                  binding("key", flash.inputs[1]),
                                  binding("value", flash.inputs[2]),
                                  binding("mask", flash.inputs[3]),
                                  binding("partial_max", scratch.partial_max),
                                  binding("partial_sum", scratch.partial_sum),
                                  binding("partial_output", scratch.partial_output),
                                  binding("completion_counter", scratch.completion.storage,
                                          scratch.completion.attention_byte_offset,
                                          scratch.completion.attention_byte_length),
                                  binding("output", flash.output) };
            runtime_scalar(dispatch, "key_value_token_count", facts.context_count);
            compile_config(dispatch, "qwen3_moe.attention.key_value_head_count",
                           std::to_string(facts.key_value_head_count));
            compile_config(dispatch, "qwen3_moe.attention.query_head_count", std::to_string(facts.query_head_count));
        }
    }

    static void bind_prefill_block(const Graph &                  graph,
                                   const RoutedTransformerModel & model,
                                   const RoutedTransformerBlock & block,
                                   Invocation &                   invocation,
                                   const Scratch &                scratch,
                                   const Facts &                  facts,
                                   ValueId                        hidden_state,
                                   std::vector<std::string> &     errors) {
        const bool        terminal            = block.ordinal + 1 == model.blocks.size();
        const OperationId attention_selection = block.operations_by_role.attention_output_selection;
        const OperationId hidden_selection    = block.operations_by_role.hidden_state_selection;
        const bool        publishes_selection = terminal && attention_selection != kInvalidId;
        const ValueId     active_hidden_state =
            publishes_selection ? op_output(graph, block.operations_by_role.attention_residual) : hidden_state;
        const OperationId        prepared         = block.operations_by_role.attention_prepared;
        const OperationId        attention_result = block.operations_by_role.attention_result_reshape;
        const OperationId        attention_output = block.operations_by_role.attention_output_projection;
        const OperationId        ff_prepared      = block.operations_by_role.feed_forward_prepared;
        const OperationId        router           = block.operations_by_role.router_projection;
        const OperationId        route_ids        = block.operations_by_role.router_route_ids;
        const OperationId        route_weights    = block.operations_by_role.router_route_weights;
        const OperationId        gate             = block.operations_by_role.experts_gate_projection;
        const OperationId        up               = block.operations_by_role.experts_up_projection;
        const OperationId        swiglu           = block.operations_by_role.experts_gate_up;
        const OperationId        down             = block.operations_by_role.experts_routed_down;
        std::vector<OperationId> projections      = {
            block.operations_by_role.attention_query_projection,
            block.operations_by_role.attention_value_projection,
            block.operations_by_role.attention_key_projection,
        };
        size_t projection_index = 0;
        for (Dispatch & dispatch : invocation.dispatches) {
            const std::string & variant = dispatch.kernel.variant;
            if (variant == "qwen3_moe_rmsnorm_f32") {
                if (invocation.recipe == "attention.prepare") {
                    dispatch.bindings = { binding("input", hidden_state),
                                          binding("weight", op_input(graph, prepared, 1)),
                                          binding("output", op_output(graph, prepared)) };
                } else {
                    dispatch.bindings = { binding("input", active_hidden_state),
                                          binding("weight", op_input(graph, ff_prepared, 1)),
                                          binding("output", op_output(graph, ff_prepared)) };
                }
                rmsnorm_config(dispatch, facts);
            } else if (variant == "qwen3_moe_dense_linear_q4k_f16_wmma" ||
                       variant == "qwen3_moe_dense_linear_q6k_f16_wmma") {
                if (invocation.recipe == "attention.qkv_publication") {
                    const OperationId projection = projections[projection_index++];
                    dispatch.bindings            = { binding("input", op_input(graph, projection, 1)),
                                                     binding("weight", op_input(graph, projection, 0)),
                                                     binding("output", op_output(graph, projection)) };
                    compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.hidden_size));
                    compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
                    compile_config(
                        dispatch, "qwen3_moe.dense_quantized.output_size",
                        std::to_string(projection == projections.front() ? facts.query_size : facts.key_value_size));
                } else {
                    ++projection_index;
                    dispatch.bindings = { binding("input", op_output(graph, attention_result)),
                                          binding("weight", op_input(graph, attention_output, 0)),
                                          binding("output", publishes_selection ? op_output(graph, attention_output) :
                                                                                  hidden_state) };
                    compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.query_size));
                    compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation",
                                   publishes_selection ? "0" : "1");
                    compile_config(dispatch, "qwen3_moe.dense_quantized.output_size",
                                   std::to_string(facts.hidden_size));
                }
            } else if (variant == "ggml_gather_add_f32") {
                if (!publishes_selection || hidden_selection == kInvalidId ||
                    op_input(graph, attention_selection, 1) != op_input(graph, hidden_selection, 1)) {
                    errors.push_back("gather/add dispatch has no coherent terminal output selection");
                    continue;
                }
                dispatch.bindings = { binding("attention", op_output(graph, attention_output)),
                                      binding("residual", hidden_state),
                                      binding("output_ids", op_input(graph, attention_selection, 1)),
                                      binding("output", active_hidden_state) };
                runtime_scalar(dispatch, "source_token_count", facts.token_count);
                runtime_scalar(dispatch, "output_token_count", facts.output_count);
                runtime_scalar(dispatch, "hidden_size", facts.hidden_size);
            } else if (variant == "qwen3_moe_attention_postprocess_f32_f16" ||
                       variant == "qwen3_moe_flash_attention_f32_f16_wmma") {
                bind_attention_common(graph, block, dispatch, scratch, facts);
            } else if (variant == "qwen3_moe_router_projection_f32_four_row_wave32") {
                dispatch.bindings = { binding("input", op_output(graph, ff_prepared)),
                                      binding("weight", op_input(graph, router, 0)),
                                      binding("output", op_output(graph, router)) };
                compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
            } else if (variant == "qwen3_moe_router_top8_f32") {
                dispatch.bindings = { binding("logits", op_output(graph, router)),
                                      binding("route_ids", op_output(graph, route_ids)),
                                      binding("route_weights", op_output(graph, route_weights)) };
                runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
                compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
                compile_config(dispatch, "qwen3_moe.router.route_count", std::to_string(facts.route_count));
            } else if (variant == "qwen3_moe_build_expert_table") {
                dispatch.bindings = { binding("route_ids", op_output(graph, route_ids)),
                                      binding("expert_table", scratch.expert_table) };
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                compile_config(dispatch, "qwen3_moe.routed_gate_up.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.expert_count", std::to_string(facts.expert_count));
            } else if (variant == "qwen3_moe_build_expert_partition_table") {
                dispatch.bindings = { binding("expert_table", scratch.expert_table),
                                      binding("partition_table", scratch.partition_table) };
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                routed_gate_up_config(dispatch, facts);
            } else if (variant == "qwen3_moe_build_expert_table_partition_prefill_512") {
                dispatch.bindings = { binding("route_ids", op_output(graph, route_ids)),
                                      binding("expert_table", scratch.expert_table),
                                      binding("partition_table", scratch.partition_table),
                                      binding("completion_counter", scratch.prefill_completion_counter) };
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
            } else if (variant == "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma") {
                dispatch.bindings = { binding("input", op_output(graph, ff_prepared)),
                                      binding("expert_table", scratch.expert_table),
                                      binding("partition_table", scratch.partition_table),
                                      binding("gate_weight", op_input(graph, gate, 0)),
                                      binding("up_weight", op_input(graph, up, 0)),
                                      binding("output", op_output(graph, swiglu)) };
                compile_config(dispatch, "qwen3_moe.routed_gate_up.input_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.expert_count", std::to_string(facts.expert_count));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.output_size",
                               std::to_string(facts.expert_intermediate_size));
            } else if (variant == "qwen3_moe_routed_down_q4k_f16_wmma_grouped" ||
                       variant == "qwen3_moe_routed_down_q6k_f16_wmma_grouped") {
                dispatch.bindings = { binding("input", op_output(graph, swiglu)),
                                      binding("expert_table", scratch.expert_table),
                                      binding("weight", op_input(graph, down, 0)),
                                      binding("output", op_output(graph, down)) };
                compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
                compile_config(dispatch, "qwen3_moe.routed_down.input_size",
                               std::to_string(facts.expert_intermediate_size));
                compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
            } else if (variant == "qwen3_moe_routed_down_weighted_reduce_f16_f32") {
                dispatch.bindings = { binding("route_weights", op_output(graph, route_weights)),
                                      binding("routed_output", op_output(graph, down)),
                                      binding("output", active_hidden_state) };
                routed_down_config(dispatch, facts);
            } else if (variant == "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32") {
                if (terminal) {
                    errors.push_back("terminal block cannot publish next attention RMSNorm");
                    continue;
                }
                const RoutedTransformerBlock & next_block    = model.blocks[block.ordinal + 1];
                const OperationId              next_prepared = next_block.operations_by_role.attention_prepared;
                dispatch.bindings = { binding("route_weights", op_output(graph, route_weights)),
                                      binding("routed_output", op_output(graph, down)),
                                      binding("hidden_state", active_hidden_state),
                                      binding("next_norm_weight", op_input(graph, next_prepared, 1)),
                                      binding("next_projection_input", op_output(graph, next_prepared)) };
                compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.model.rms_epsilon", facts.rms_epsilon);
                compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
            } else {
                errors.push_back("unexpected routed-transformer prefill dispatch " + variant);
            }
            runtime_scalar(dispatch, "token_count", dispatch.kernel.integer_parameters["token_count"]);
            if (variant.rfind("qwen3_moe_", 0) == 0) {
                compile_config(dispatch, "qwen3_moe.workload.token_capacity",
                               std::to_string(dispatch.kernel.integer_parameters["token_count"]));
            }
        }
    }

    static void bind_decode_block(const Graph &                  graph,
                                  const RoutedTransformerModel & model,
                                  const RoutedTransformerBlock & block,
                                  Invocation &                   invocation,
                                  const Scratch &                scratch,
                                  const Facts &                  facts,
                                  ValueId                        hidden_state,
                                  std::vector<std::string> &     errors) {
        const bool        terminal         = block.ordinal + 1 == model.blocks.size();
        const OperationId query            = block.operations_by_role.attention_query_projection;
        const OperationId key              = block.operations_by_role.attention_key_projection;
        const OperationId value_projection = block.operations_by_role.attention_value_projection;
        const OperationId query_rope       = block.operations_by_role.attention_query_rope;
        const OperationId key_writer       = block.operations_by_role.attention_key_cache_writer;
        const OperationId value_writer     = block.operations_by_role.attention_value_cache_writer;
        const OperationId flash_id         = block.operations_by_role.attention_flash;
        const Operation & flash            = graph.operations[flash_id];
        const OperationId attention_result = block.operations_by_role.attention_result_reshape;
        const OperationId attention_output = block.operations_by_role.attention_output_projection;
        const OperationId ff_prepared      = block.operations_by_role.feed_forward_prepared;
        const OperationId router           = block.operations_by_role.router_projection;
        const OperationId route_ids        = block.operations_by_role.router_route_ids;
        const OperationId route_weights    = block.operations_by_role.router_route_weights;
        const OperationId gate             = block.operations_by_role.experts_gate_projection;
        const OperationId up               = block.operations_by_role.experts_up_projection;
        const OperationId swiglu           = block.operations_by_role.experts_gate_up;
        const OperationId down             = block.operations_by_role.experts_routed_down;
        for (Dispatch & dispatch : invocation.dispatches) {
            const std::string & variant = dispatch.kernel.variant;
            if (variant == "qwen3_moe_attention_qkv_quantized") {
                dispatch.bindings = { binding("q8_input", scratch.q8_hidden),
                                      binding("query_weight", op_input(graph, query, 0)),
                                      binding("key_weight", op_input(graph, key, 0)),
                                      binding("value_weight", op_input(graph, value_projection, 0)),
                                      binding("query_output", op_output(graph, query)),
                                      binding("key_output", op_output(graph, key)),
                                      binding("value_output", op_output(graph, value_projection)) };
                compile_config(dispatch, "qwen3_moe.attention.key_value_size", std::to_string(facts.key_value_size));
                compile_config(dispatch, "qwen3_moe.attention.query_size", std::to_string(facts.query_size));
                compile_config(dispatch, "qwen3_moe.attention.value_uses_q6",
                               dispatch.kernel.integer_parameters["value_weight_type"] == GGML_TYPE_Q6_K ? "1" : "0");
                compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_size", std::to_string(facts.key_value_size));
                rmsnorm_config(dispatch, facts);
            } else if (variant == "qwen3_moe_attention_qkv_postprocess_fused_decode") {
                const OperationId query_scale = producer(graph, op_input(graph, query_rope, 0));
                const OperationId key_rope    = primary_ancestor(graph, op_input(graph, key_writer, 0), GGML_OP_ROPE);
                const OperationId key_scale   = producer(graph, op_input(graph, key_rope, 0));
                dispatch.bindings             = {
                    binding("q8_input", scratch.q8_hidden),
                    binding("query_weight", op_input(graph, query, 0)),
                    binding("key_weight", op_input(graph, key, 0)),
                    binding("value_weight", op_input(graph, value_projection, 0)),
                    binding("positions", op_input(graph, query_rope, 1)),
                    binding("key_cache_indices", op_input(graph, key_writer, 1)),
                    binding("value_cache_indices", op_input(graph, value_writer, 1)),
                    binding("query_output_raw", op_output(graph, query)),
                    binding("key_output_raw", op_output(graph, key)),
                    binding("value_output_raw", op_output(graph, value_projection)),
                    binding("query_norm_weight", op_input(graph, query_scale, 1)),
                    binding("key_norm_weight", op_input(graph, key_scale, 1)),
                    binding("inverse_frequencies", scratch.inverse_frequencies),
                    binding("query_output", op_output(graph, query_rope)),
                    binding("key_cache", op_input(graph, key_writer, 2)),
                    binding("value_cache", op_input(graph, value_writer, 2)),
                    binding("completion_counters", scratch.completion.storage, 0, scratch.completion.total_byte_length),
                };
                runtime_scalar(dispatch, "cache_row_count", facts.context_count);
                compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.attention.query_size", std::to_string(facts.query_size));
                compile_config(dispatch, "qwen3_moe.attention.key_value_size", std::to_string(facts.key_value_size));
                compile_config(dispatch, "qwen3_moe.attention.value_uses_q6",
                               dispatch.kernel.integer_parameters["value_weight_type"] == GGML_TYPE_Q6_K ? "1" : "0");
                compile_config(dispatch, "qwen3_moe.attention.head_size", std::to_string(facts.head_size));
                compile_config(dispatch, "qwen3_moe.model.rms_epsilon", facts.rms_epsilon);
            } else if (variant == "qwen3_moe_attention_postprocess_f32_f16" ||
                       variant == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma") {
                bind_attention_common(graph, block, dispatch, scratch, facts);
            } else if (variant == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8") {
                dispatch.bindings = {
                    binding("query", flash.inputs[0]),
                    binding("key", flash.inputs[1]),
                    binding("value", flash.inputs[2]),
                    binding("mask", flash.inputs[3]),
                    binding("partial_max", scratch.partial_max),
                    binding("partial_sum", scratch.partial_sum),
                    binding("partial_output", scratch.partial_output),
                    binding("completion_counter", scratch.completion.storage, scratch.completion.attention_byte_offset,
                            scratch.completion.attention_byte_length),
                    binding("output", flash.output),
                    binding("next_q8_output", scratch.q8_attention),
                };
                runtime_scalar(dispatch, "key_value_token_count", facts.context_count);
                compile_config(dispatch, "qwen3_moe.attention.key_value_head_count",
                               std::to_string(facts.key_value_head_count));
                compile_config(dispatch, "qwen3_moe.attention.query_head_count",
                               std::to_string(facts.query_head_count));
                compile_config(dispatch, "qwen3_moe.attention.key_value_token_capacity",
                               std::to_string(facts.context_count));
            } else if (variant == "ggml_quantize_q8_1_x4_f32") {
                if (dispatch.kernel.integer_parameters["token_count"] == 1) {
                    dispatch.bindings = { binding("input", op_output(graph, attention_result)),
                                          binding("output", scratch.q8_attention) };
                    runtime_scalar(dispatch, "input_size", facts.query_size);
                } else {
                    dispatch.bindings = { binding("input", op_output(graph, swiglu)),
                                          binding("output", scratch.q8_swiglu) };
                    runtime_scalar(dispatch, "input_size", facts.expert_intermediate_size);
                }
                compile_config(dispatch, "ggml.quantize_q8_1_x4.group_capacity",
                               std::to_string(dispatch.kernel.integer_parameters["token_count"] *
                                              dispatch.kernel.integer_parameters["input_size"] / 128));
            } else if (variant == "qwen3_moe_dense_linear_q4k_q8_1_x4") {
                dispatch.bindings = { binding("q8_input", scratch.q8_attention),
                                      binding("weight", op_input(graph, attention_output, 0)),
                                      binding("output", hidden_state) };
                compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.query_size));
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "1");
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_size", std::to_string(facts.hidden_size));
            } else if (variant == "qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8") {
                dispatch.bindings = {
                    binding("q8_input", scratch.q8_attention),
                    binding("weight", op_input(graph, attention_output, 0)),
                    binding("output", hidden_state),
                    binding("norm_weight", op_input(graph, ff_prepared, 1)),
                    binding("normalized_output", op_output(graph, ff_prepared)),
                    binding("completion_counter", scratch.completion.storage, scratch.completion.shared_byte_offset,
                            sizeof(int32_t)),
                    binding("next_q8_output", scratch.q8_hidden),
                };
                compile_config(dispatch, "qwen3_moe.dense_quantized.input_size", std::to_string(facts.query_size));
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "1");
                compile_config(dispatch, "qwen3_moe.dense_quantized.output_size", std::to_string(facts.hidden_size));
                rmsnorm_config(dispatch, facts);
            } else if (variant == "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4") {
                const OperationId selected = invocation.recipe == "experts.down_publication" && terminal ?
                                                 model.operations_by_role.endpoint_prepared :
                                                 ff_prepared;
                dispatch.bindings = { binding("input", hidden_state), binding("weight", op_input(graph, selected, 1)),
                                      binding("normalized_output", op_output(graph, selected)),
                                      binding("q8_output", scratch.q8_hidden) };
                rmsnorm_config(dispatch, facts);
            } else if (variant == "qwen3_moe_attention_rmsnorm_quantize_q8_1_x4") {
                const OperationId next = model.blocks[block.ordinal + 1].operations_by_role.attention_prepared;
                dispatch.bindings      = { binding("input", hidden_state), binding("weight", op_input(graph, next, 1)),
                                           binding("output", scratch.q8_hidden) };
                rmsnorm_config(dispatch, facts);
            } else if (variant == "qwen3_moe_router_projection_f32_one_row_wave64") {
                dispatch.bindings = { binding("input", op_output(graph, ff_prepared)),
                                      binding("weight", op_input(graph, router, 0)),
                                      binding("output", op_output(graph, router)) };
                compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
            } else if (variant == "qwen3_moe_router_top8_f32") {
                dispatch.bindings = { binding("logits", op_output(graph, router)),
                                      binding("route_ids", op_output(graph, route_ids)),
                                      binding("route_weights", op_output(graph, route_weights)) };
                runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
                compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
                compile_config(dispatch, "qwen3_moe.router.route_count", std::to_string(facts.route_count));
            } else if (variant == "qwen3_moe_router_projection_top8_fused_decode_f32") {
                dispatch.bindings = {
                    binding("input", op_output(graph, ff_prepared)),
                    binding("weight", op_input(graph, router, 0)),
                    binding("logits", op_output(graph, router)),
                    binding("completion_counter", scratch.completion.storage, scratch.completion.shared_byte_offset,
                            sizeof(int32_t)),
                    binding("route_ids", op_output(graph, route_ids)),
                    binding("route_weights", op_output(graph, route_weights)),
                };
                runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
                compile_config(dispatch, "qwen3_moe.model.hidden_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.router.expert_count", std::to_string(facts.expert_count));
                compile_config(dispatch, "qwen3_moe.router.route_count", std::to_string(facts.route_count));
            } else if (variant == "qwen3_moe_routed_gate_up_swiglu_q4k_q8") {
                dispatch.bindings = { binding("q8_input", scratch.q8_hidden),
                                      binding("route_ids", op_output(graph, route_ids)),
                                      binding("gate_weight", op_input(graph, gate, 0)),
                                      binding("up_weight", op_input(graph, up, 0)),
                                      binding("output", op_output(graph, swiglu)) };
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                runtime_scalar(dispatch, "output_size", facts.expert_intermediate_size);
                compile_config(dispatch, "qwen3_moe.routed_gate_up.input_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.output_size",
                               std::to_string(facts.expert_intermediate_size));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.expert_count", std::to_string(facts.expert_count));
            } else if (variant == "qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8") {
                dispatch.bindings = {
                    binding("q8_input", scratch.q8_hidden),
                    binding("route_ids", op_output(graph, route_ids)),
                    binding("gate_weight", op_input(graph, gate, 0)),
                    binding("up_weight", op_input(graph, up, 0)),
                    binding("output", op_output(graph, swiglu)),
                    binding("completion_counters", scratch.completion.storage, 0,
                            scratch.completion.grouped_byte_length),
                    binding("next_q8_output", scratch.q8_swiglu),
                };
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                runtime_scalar(dispatch, "output_size", facts.expert_intermediate_size);
                compile_config(dispatch, "qwen3_moe.routed_gate_up.input_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.output_size",
                               std::to_string(facts.expert_intermediate_size));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_gate_up.expert_count", std::to_string(facts.expert_count));
            } else if (variant == "qwen3_moe_routed_down_q4k_q8_1_x4" ||
                       variant == "qwen3_moe_routed_down_q6k_q8_1_x4") {
                dispatch.bindings = { binding("q8_input", scratch.q8_swiglu),
                                      binding("route_ids", op_output(graph, route_ids)),
                                      binding("route_weights", op_output(graph, route_weights)),
                                      binding("weight", op_input(graph, down, 0)), binding("output", hidden_state) };
                runtime_scalar(dispatch, "input_size", facts.expert_intermediate_size);
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                runtime_scalar(dispatch, "output_size", facts.hidden_size);
                compile_config(dispatch, "qwen3_moe.routed_down.input_size",
                               std::to_string(facts.expert_intermediate_size));
                compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
            } else if (variant == "qwen3_moe_routed_down_q4k_q8_1_x4_next_q8" ||
                       variant == "qwen3_moe_routed_down_q6k_f32_wave64_next_q8") {
                const bool        q6   = variant == "qwen3_moe_routed_down_q6k_f32_wave64_next_q8";
                const OperationId next = model.blocks[block.ordinal + 1].operations_by_role.attention_prepared;
                dispatch.bindings      = {
                    binding(q6 ? "input" : "q8_input", q6 ? op_output(graph, swiglu) : scratch.q8_swiglu),
                    binding("route_ids", op_output(graph, route_ids)),
                    binding("route_weights", op_output(graph, route_weights)),
                    binding("weight", op_input(graph, down, 0)),
                    binding("output", hidden_state),
                    binding("norm_weight", op_input(graph, next, 1)),
                    binding("completion_counter", scratch.completion.storage, scratch.completion.shared_byte_offset,
                                 sizeof(int32_t)),
                    binding("next_q8_output", scratch.q8_hidden),
                };
                runtime_scalar(dispatch, "input_size", facts.expert_intermediate_size);
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                runtime_scalar(dispatch, "output_size", facts.hidden_size);
                compile_config(dispatch, "qwen3_moe.routed_down.input_size",
                               std::to_string(facts.expert_intermediate_size));
                compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
                rmsnorm_config(dispatch, facts);
            } else if (variant == "qwen3_moe_routed_down_q6k_f32_wave64") {
                dispatch.bindings = {
                    binding("input", op_output(graph, swiglu)),
                    binding("route_ids", op_output(graph, route_ids)),
                    binding("route_weights", op_output(graph, route_weights)),
                    binding("weight", op_input(graph, down, 0)),
                    binding("output", hidden_state),
                };
                runtime_scalar(dispatch, "input_size", facts.expert_intermediate_size);
                runtime_scalar(dispatch, "route_count", facts.route_count);
                runtime_scalar(dispatch, "route_id_stride", facts.route_stride);
                runtime_scalar(dispatch, "expert_count", facts.expert_count);
                runtime_scalar(dispatch, "output_size", facts.hidden_size);
                compile_config(dispatch, "qwen3_moe.routed_down.input_size",
                               std::to_string(facts.expert_intermediate_size));
                compile_config(dispatch, "qwen3_moe.routed_down.output_size", std::to_string(facts.hidden_size));
                compile_config(dispatch, "qwen3_moe.routed_down.route_count", std::to_string(facts.route_count));
                compile_config(dispatch, "qwen3_moe.routed_down.expert_count", std::to_string(facts.expert_count));
            } else {
                errors.push_back("unexpected routed-transformer decode dispatch " + variant);
            }
            runtime_scalar(dispatch, "token_count", dispatch.kernel.integer_parameters["token_count"]);
            if (variant.rfind("qwen3_moe_", 0) == 0) {
                compile_config(dispatch, "qwen3_moe.workload.token_capacity",
                               std::to_string(dispatch.kernel.integer_parameters["token_count"]));
            }
        }
    }

    static void bind_endpoint(const Graph &                  graph,
                              const RoutedTransformerModel & model,
                              Invocation &                   invocation,
                              const Scratch &                scratch,
                              const Facts &                  facts,
                              ValueId                        hidden_state,
                              std::vector<std::string> &     errors) {
        const OperationId              prepared   = model.operations_by_role.endpoint_prepared;
        const OperationId              projection = model.operations_by_role.endpoint_projection;
        const RoutedTransformerBlock & terminal   = model.blocks.back();
        const ValueId                  endpoint_hidden_state =
            model.query_token_count != 1 && terminal.operations_by_role.attention_output_selection != kInvalidId ?
                                 op_output(graph, terminal.operations_by_role.attention_residual) :
                                 hidden_state;
        for (Dispatch & dispatch : invocation.dispatches) {
            if (dispatch.kernel.variant == "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4") {
                dispatch.bindings = { binding("input", endpoint_hidden_state),
                                      binding("weight", op_input(graph, prepared, 1)),
                                      binding("normalized_output", op_output(graph, prepared)),
                                      binding("q8_output", scratch.q8_hidden) };
                runtime_scalar(dispatch, "token_count", facts.output_count);
                rmsnorm_config(dispatch, facts);
            } else if (dispatch.kernel.variant == "ggml_linear_q6k_q8_1_x4") {
                dispatch.bindings = { binding("q8_input", scratch.q8_hidden),
                                      binding("weight", op_input(graph, projection, 0)),
                                      binding("output", op_output(graph, projection)) };
                runtime_scalar(dispatch, "token_count", facts.output_count);
                runtime_scalar(dispatch, "input_size", facts.hidden_size);
                runtime_scalar(dispatch, "output_size", facts.vocabulary_count);
                compile_config(dispatch, "ggml.linear_q6k_q8_1_x4.token_capacity", std::to_string(facts.output_count));
                compile_config(dispatch, "ggml.linear_q6k_q8_1_x4.output_capacity",
                               std::to_string(facts.vocabulary_count));
            } else {
                errors.push_back("unexpected routed-transformer endpoint dispatch " + dispatch.kernel.variant);
            }
            if (dispatch.kernel.variant.rfind("qwen3_moe_", 0) == 0) {
                compile_config(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(facts.output_count));
            }
        }
    }
};

}  // namespace

VerificationResult RoutedTransformerProgramProof::materialize_dispatch_bindings(Graph &                        graph,
                                                                                Schedule &                     schedule,
                                                                                const RoutedTransformerModel & model) {
    VerificationResult result;
    const GraphIndex   index(graph);
    if (!model.valid()) {
        result.errors = model.errors;
        return result;
    }
    const bool prefill = schedule.workload.rfind("prefill-", 0) == 0;
    const bool decode  = schedule.workload.rfind("decode-", 0) == 0;
    if (!prefill && !decode) {
        result.errors.push_back("unsupported routed-transformer workload " + schedule.workload);
        return result;
    }
    Facts facts = RoutedTransformerBindingImplementation::recover_facts(graph, model, result.errors);
    if (!result.errors.empty()) {
        return result;
    }
    const std::string expected =
        prefill ? "prefill-" + std::to_string(facts.token_count) : "decode-" + std::to_string(facts.context_count);
    if (schedule.workload != expected || (decode && facts.token_count != 1)) {
        result.errors.push_back("routed-transformer workload disagrees with graph facts");
        return result;
    }
    const Scratch scratch      = RoutedTransformerBindingImplementation::allocate_scratch(graph, facts, decode);
    const ValueId hidden_state = model.values_by_role.program_hidden_state;
    for (Invocation & invocation : schedule.invocations) {
        if (invocation.recipe == "program.preamble") {
            RoutedTransformerBindingImplementation::bind_preamble(graph, model, invocation, scratch, facts,
                                                                  result.errors);
        } else if (invocation.recipe == "program.endpoint") {
            RoutedTransformerBindingImplementation::bind_endpoint(graph, model, invocation, scratch, facts,
                                                                  hidden_state, result.errors);
        } else if (invocation.recipe.rfind("atom.", 0) == 0) {
            if (invocation.dispatches.size() != 1) {
                result.errors.push_back("atom recipe " + invocation.recipe + " does not contain one dispatch");
                continue;
            }
            Dispatch & dispatch = invocation.dispatches.front();
            dispatch.bindings   = invocation.inputs;
            dispatch.bindings.insert(dispatch.bindings.end(), invocation.outputs.begin(), invocation.outputs.end());
        } else if (invocation.layer < 0 || static_cast<size_t>(invocation.layer) >= model.blocks.size()) {
            result.errors.push_back("routed-transformer recipe " + invocation.recipe + " has no owning block");
        } else {
            const RoutedTransformerBlock & block = model.blocks[invocation.layer];
            if (prefill) {
                RoutedTransformerBindingImplementation::bind_prefill_block(graph, model, block, invocation, scratch,
                                                                           facts, hidden_state, result.errors);
            } else {
                RoutedTransformerBindingImplementation::bind_decode_block(graph, model, block, invocation, scratch,
                                                                          facts, hidden_state, result.errors);
            }
        }
    }
    return result;
}

}  // namespace ggml::hrx
