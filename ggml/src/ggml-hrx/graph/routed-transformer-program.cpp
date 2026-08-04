#include "routed-transformer-program.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace ggml::hrx {
namespace {

static constexpr size_t kAttentionKvTileSize = 64;

class RoutedTransformerProgramImplementation {
public:

static enum ggml_type weight_type(const Graph & graph, OperationId operation) {
    if (operation >= graph.operations.size() || graph.operations[operation].inputs.empty()) return GGML_TYPE_COUNT;
    const ValueId weight = graph.operations[operation].inputs[0];
    return weight < graph.values.size() ? graph.values[weight].type : GGML_TYPE_COUNT;
}

static std::string type_suffix(enum ggml_type type) {
    return type == GGML_TYPE_COUNT ? "invalid" : ggml_type_name(type);
}

static KernelSpecialization kernel(const std::string & variant, int layer, size_t token_count,
                                   KernelSpecialization::ExecutionKind kind = KernelSpecialization::ExecutionKind::Native) {
    KernelSpecialization result;
    result.family = "qwen3_moe";
    result.variant = variant;
    result.execution_kind = kind;
    result.integer_parameters["token_count"] = token_count;
    if (layer >= 0) result.integer_parameters["layer"] = layer;
    return result;
}

static KernelSpecialization storage_kernel(const std::string & q4_variant, const std::string & q6_variant,
                                           const std::string & semantic_role, enum ggml_type type, int layer,
                                           size_t token_count, std::vector<std::string> & gaps) {
    if (type == GGML_TYPE_Q4_K && !q4_variant.empty()) {
        KernelSpecialization result = kernel(q4_variant, layer, token_count);
        result.integer_parameters["weight_type"] = GGML_TYPE_Q4_K;
        return result;
    }
    if (type == GGML_TYPE_Q6_K && !q6_variant.empty()) {
        KernelSpecialization result = kernel(q6_variant, layer, token_count);
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

static void add_dispatch(Invocation & invocation, KernelSpecialization specialization, size_t & ordinal) {
    Dispatch dispatch;
    dispatch.kernel = std::move(specialization);
    if (ordinal != 0) dispatch.dependencies.push_back(static_cast<uint32_t>(ordinal - 1));
    invocation.dispatches.push_back(std::move(dispatch));
    ++ordinal;
}

static void append_prefill(const Graph & graph, const RoutedTransformerModel & model,
                           const RoutedTransformerBlock & block, Invocation & invocation,
                           size_t & ordinal, std::vector<std::string> & gaps) {
    const int layer = static_cast<int>(block.ordinal);
    const bool terminal = block.ordinal + 1 == model.blocks.size();
    const size_t token_count = model.query_token_count;
    const size_t active_token_count = terminal ? 1 : token_count;
    if (block.ordinal == 0) add_dispatch(invocation, kernel("qwen3_moe_rmsnorm_f32", layer, token_count), ordinal);
    for (const char * role : { "attention.query_projection", "attention.value_projection", "attention.key_projection" }) {
        const OperationId projection = block.bindings.operations.at(role);
        add_dispatch(invocation, storage_kernel("qwen3_moe_dense_linear_q4k_f16_wmma",
            "qwen3_moe_dense_linear_q6k_f16_wmma", "dense_attention_projection",
            weight_type(graph, projection), layer, token_count, gaps), ordinal);
    }
    add_dispatch(invocation, kernel("qwen3_moe_attention_postprocess_f32_f16", layer, token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_flash_attention_f32_f16_wmma", layer, token_count), ordinal);
    const OperationId attention_output = block.bindings.operations.at("attention.output_projection");
    add_dispatch(invocation, storage_kernel("qwen3_moe_dense_linear_q4k_f16_wmma",
        "qwen3_moe_dense_linear_q6k_f16_wmma", "dense_attention_output",
        weight_type(graph, attention_output), layer, token_count, gaps), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_rmsnorm_f32", layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_router_projection_f32_four_row_wave32", layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_router_top8_f32", layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_build_expert_table", layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_build_expert_partition_table", layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma", layer, active_token_count), ordinal);
    const OperationId down = block.bindings.operations.at("experts.routed_down");
    add_dispatch(invocation, storage_kernel("qwen3_moe_routed_down_q4k_f16_wmma_grouped",
        "qwen3_moe_routed_down_q6k_f16_wmma_grouped", "routed_down",
        weight_type(graph, down), layer, active_token_count, gaps), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_routed_down_weighted_reduce_f16_f32",
                                    layer, active_token_count), ordinal);
    if (!terminal) add_dispatch(invocation, kernel("qwen3_moe_rmsnorm_f32", layer, token_count), ordinal);
}

static void append_decode(const Graph & graph, const RoutedTransformerModel & model,
                          const RoutedTransformerBlock & block, Invocation & invocation,
                          size_t & ordinal, std::vector<std::string> & gaps) {
    const int layer = static_cast<int>(block.ordinal);
    const size_t token_count = model.query_token_count;
    const OperationId query = block.bindings.operations.at("attention.query_projection");
    const OperationId key = block.bindings.operations.at("attention.key_projection");
    const OperationId value = block.bindings.operations.at("attention.value_projection");
    const enum ggml_type query_type = weight_type(graph, query);
    const enum ggml_type key_type = weight_type(graph, key);
    const enum ggml_type value_type = weight_type(graph, value);
    if (query_type == GGML_TYPE_Q4_K && key_type == GGML_TYPE_Q4_K &&
        (value_type == GGML_TYPE_Q4_K || value_type == GGML_TYPE_Q6_K)) {
        KernelSpecialization qkv = kernel("qwen3_moe_attention_qkv_quantized", layer, token_count);
        qkv.integer_parameters["query_weight_type"] = query_type;
        qkv.integer_parameters["key_weight_type"] = key_type;
        qkv.integer_parameters["value_weight_type"] = value_type;
        add_dispatch(invocation, std::move(qkv), ordinal);
    } else {
        const std::string variant = "qwen3_moe_attention_qkv_postprocess_fused_decode_missing_" +
            type_suffix(query_type) + '_' + type_suffix(key_type) + '_' + type_suffix(value_type);
        gaps.push_back("block " + std::to_string(layer) + ": " + variant);
        add_dispatch(invocation, kernel(variant, layer, token_count,
                                       KernelSpecialization::ExecutionKind::NativeGap), ordinal);
    }
    add_dispatch(invocation, kernel("qwen3_moe_attention_postprocess_f32_f16", layer, token_count), ordinal);
    KernelSpecialization flash = kernel("qwen3_moe_flash_attention_decode_split_f32_f16_wmma", layer, token_count);
    flash.integer_parameters["key_value_token_count"] = model.key_value_token_count;
    flash.integer_parameters["key_value_tile_size"] = kAttentionKvTileSize;
    flash.integer_parameters["key_value_block_count"] =
        (model.key_value_token_count + kAttentionKvTileSize - 1) / kAttentionKvTileSize;
    add_dispatch(invocation, std::move(flash), ordinal);
    add_dispatch(invocation, kernel("ggml_quantize_q8_1_x4_f32", layer, token_count), ordinal);
    const OperationId attention_output = block.bindings.operations.at("attention.output_projection");
    add_dispatch(invocation, storage_kernel("qwen3_moe_dense_linear_q4k_q8_1_x4", "",
        "dense_attention_output_q8", weight_type(graph, attention_output), layer, token_count, gaps), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4", layer, token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_router_projection_f32_one_row_wave64", layer, token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_router_top8_f32", layer, token_count), ordinal);
    add_dispatch(invocation, kernel("qwen3_moe_routed_gate_up_swiglu_q4k_q8", layer, token_count), ordinal);
    add_dispatch(invocation, kernel("ggml_quantize_q8_1_x4_f32", layer, token_count * model.route_count), ordinal);
    const OperationId down = block.bindings.operations.at("experts.routed_down");
    add_dispatch(invocation, storage_kernel("qwen3_moe_routed_down_q4k_q8_1_x4",
        "qwen3_moe_routed_down_q6k_q8_1_x4", "routed_down_q8",
        weight_type(graph, down), layer, token_count, gaps), ordinal);
    add_dispatch(invocation, kernel(block.ordinal + 1 == model.blocks.size()
        ? "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4"
        : "qwen3_moe_attention_rmsnorm_quantize_q8_1_x4", layer, token_count), ordinal);
}

};

} // namespace

RoutedTransformerProgramProof RoutedTransformerProgramProof::recover(const Graph & graph) {
    RoutedTransformerProgramProof proof;
    const GraphIndex index(graph);
    if (!index.valid()) {
        proof.errors = index.errors();
        return proof;
    }
    auto model = std::make_shared<const RoutedTransformerModel>(RoutedTransformerModel::analyze(index));
    if (!model->valid()) {
        proof.errors = model->errors;
        return proof;
    }
    proof.structurally_recognized = true;
    if (model->output_token_count != 1) {
        proof.errors.push_back("current routed-transformer recipes require one demanded output token; graph demands " +
                               std::to_string(model->output_token_count));
        return proof;
    }
    SearchOptions search_options;
    search_options.require_complete_coverage = true;
    search_options.record_trace = true;
    proof.search = SearchResult::search(index, RoutedTransformerProvider::make_planner({}, model), search_options);
    if (!proof.search.valid()) {
        proof.errors = proof.search.errors;
        return proof;
    }

    Schedule & schedule = proof.schedule;
    schedule.graph_fingerprint = graph.fingerprint;
    const bool prefill = model->query_token_count != 1;
    schedule.workload = prefill ? "prefill-" + std::to_string(model->query_token_count)
                                : "decode-" + std::to_string(model->key_value_token_count);
    schedule.oracle_revision = "routed-transformer-search-v1";
    for (ValueId root : graph.roots) {
        const Value & value = graph.values[root];
        const std::string materialization = value.access.shape[0] > model->hidden_size
            ? "full_f32_logits" : "normalized_f32_hidden_state";
        schedule.roots.push_back({ root, RootDisposition::Materialized, materialization });
    }

    size_t dispatch_ordinal = 0;
    Invocation preamble;
    preamble.stage = "program.preamble";
    preamble.kernel = RoutedTransformerProgramImplementation::kernel("owned_program_preamble", -1, model->query_token_count);
    preamble.covered_operations = model->preamble_operations;
    RoutedTransformerProgramImplementation::calculate_boundaries(index, preamble);
    KernelSpecialization embedding = RoutedTransformerProgramImplementation::kernel("qwen_token_embedding_q4k_bringup_workaround", -1, model->query_token_count);
    embedding.integer_parameters["vocabulary_count"] = graph.values[graph.operations[model->preamble_operations.front()].inputs[0]].access.shape[1];
    embedding.integer_parameters["hidden_size"] = model->hidden_size;
    RoutedTransformerProgramImplementation::add_dispatch(preamble, std::move(embedding), dispatch_ordinal);
    KernelSpecialization metadata = RoutedTransformerProgramImplementation::kernel("qwen_attention_metadata_bringup_workaround", -1, model->query_token_count);
    metadata.integer_parameters["context_capacity"] = model->key_value_token_count;
    RoutedTransformerProgramImplementation::add_dispatch(preamble, std::move(metadata), dispatch_ordinal);
    if (!prefill) RoutedTransformerProgramImplementation::add_dispatch(preamble, RoutedTransformerProgramImplementation::kernel("qwen3_moe_attention_rmsnorm_quantize_q8_1_x4",
                                               -1, model->query_token_count), dispatch_ordinal);
    schedule.invocations.push_back(std::move(preamble));

    for (const RoutedTransformerBlock & block : model->blocks) {
        Invocation invocation;
        invocation.stage = block.ordinal + 1 == model->blocks.size() ? "program.terminal_block" : "program.block";
        invocation.layer = static_cast<int32_t>(block.ordinal);
        invocation.kernel = RoutedTransformerProgramImplementation::kernel(prefill ? "owned_prefill_block" : "owned_decode_block",
                                   invocation.layer, model->query_token_count);
        invocation.covered_operations = block.operations;
        RoutedTransformerProgramImplementation::calculate_boundaries(index, invocation);
        if (prefill) RoutedTransformerProgramImplementation::append_prefill(graph, *model, block, invocation, dispatch_ordinal, proof.native_gaps);
        else RoutedTransformerProgramImplementation::append_decode(graph, *model, block, invocation, dispatch_ordinal, proof.native_gaps);
        schedule.invocations.push_back(std::move(invocation));
    }

    Invocation endpoint;
    endpoint.stage = "program.endpoint";
    endpoint.kernel = RoutedTransformerProgramImplementation::kernel("owned_program_endpoint", -1, 1);
    endpoint.covered_operations = model->endpoint_operations;
    RoutedTransformerProgramImplementation::calculate_boundaries(index, endpoint);
    if (prefill) RoutedTransformerProgramImplementation::add_dispatch(endpoint, RoutedTransformerProgramImplementation::kernel("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4", -1, 1), dispatch_ordinal);
    RoutedTransformerProgramImplementation::add_dispatch(endpoint, RoutedTransformerProgramImplementation::kernel("ggml_linear_q6k_q8_1_x4", -1, 1), dispatch_ordinal);
    schedule.invocations.push_back(std::move(endpoint));
    // The count is a derived schedule contract, not a model identity. Recipe
    // additions are expected to change it while preserving full graph coverage.
    schedule.expected_dispatch_count = Schedule::dispatch_count(schedule);
    return proof;
}

} // namespace ggml::hrx
