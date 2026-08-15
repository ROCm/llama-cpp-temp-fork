#include "command-program.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph-index.h"
#include "graph-ir.h"
#include "kernel-corpus.h"
#include "reactive-plan.h"
#include "routed-transformer.h"
#include "schedule.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

namespace {

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (0)

struct Fixture {
    ggml_context * context = nullptr;
    ggml_cgraph *  graph   = nullptr;

    Fixture() {
        ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
        context                 = ggml_init(params);
        REQUIRE(context != nullptr);
        graph = ggml_new_graph_custom(context, 64, false);
    }

    ~Fixture() { ggml_free(context); }
};

static ggml::hrx::Graph make_arithmetic_graph(const char * prefix) {
    Fixture       fixture;
    ggml_tensor * x = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * y = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * z = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_input(z);
    ggml_tensor * add = ggml_add(fixture.context, x, y);
    ggml_tensor * mul = ggml_mul(fixture.context, add, z);
    ggml_set_name(x, (std::string(prefix) + "-x").c_str());
    ggml_set_name(add, (std::string(prefix) + "-add").c_str());
    ggml_set_name(mul, (std::string(prefix) + "-mul").c_str());
    ggml_set_output(mul);
    ggml_build_forward_expand(fixture.graph, mul);
    return ggml::hrx::Graph::import(fixture.graph);
}

static ggml_tensor * build_arithmetic_graph(Fixture & fixture, const char * prefix) {
    ggml_tensor * x = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * y = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * z = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_input(z);
    ggml_tensor * add = ggml_add(fixture.context, x, y);
    ggml_tensor * mul = ggml_mul(fixture.context, add, z);
    ggml_set_name(x, (std::string(prefix) + "-x").c_str());
    ggml_set_name(add, (std::string(prefix) + "-add").c_str());
    ggml_set_name(mul, (std::string(prefix) + "-mul").c_str());
    ggml_set_output(mul);
    ggml_build_forward_expand(fixture.graph, mul);
    return mul;
}

static void build_fork_join_graph(Fixture & fixture, const char * prefix) {
    ggml_tensor * x = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * y = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * z = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * w = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_set_input(z);
    ggml_set_input(w);
    ggml_tensor * left   = ggml_add(fixture.context, x, y);
    ggml_tensor * right  = ggml_mul(fixture.context, z, w);
    ggml_tensor * result = ggml_add(fixture.context, left, right);
    ggml_set_name(left, (std::string(prefix) + "-left").c_str());
    ggml_set_name(right, (std::string(prefix) + "-right").c_str());
    ggml_set_name(result, (std::string(prefix) + "-result").c_str());
    ggml_set_output(result);
    ggml_build_forward_expand(fixture.graph, result);
}

static void test_deterministic_import_and_schedule() {
    const ggml::hrx::Graph first  = make_arithmetic_graph("first");
    const ggml::hrx::Graph second = make_arithmetic_graph("second");
    REQUIRE(first.valid());
    REQUIRE(first.operations.size() == 2);
    REQUIRE(first.fingerprint == second.fingerprint);
    const std::string graph_json = ggml::hrx::Graph::serialize_json(first);
    REQUIRE(graph_json == ggml::hrx::Graph::serialize_json(first));
    REQUIRE(graph_json.find("\"params\":") != std::string::npos);
    REQUIRE(graph_json.find("\"roots\":[") != std::string::npos);
    const ggml::hrx::Graph round_trip_graph = ggml::hrx::Graph::deserialize_json(graph_json);
    REQUIRE(round_trip_graph.valid());
    REQUIRE(ggml::hrx::Graph::serialize_json(round_trip_graph) == graph_json);

    ggml::hrx::Schedule schedule;
    schedule.graph_fingerprint = first.fingerprint;
    ggml::hrx::Invocation invocation;
    invocation.kernel = {
        "test", "fused_add_mul", { { "width", 4 } }, ggml::hrx::KernelSpecialization::ExecutionKind::Native, {}
    };
    invocation.covered_operations = { 0, 1 };
    for (ggml::hrx::ValueId value = 0; value < first.values.size(); ++value) {
        if (first.values[value].producer == ggml::hrx::kInvalidId) {
            invocation.inputs.push_back({ "input" + std::to_string(value), value, 0, 0, {} });
        }
    }
    invocation.outputs.push_back({ "result", first.operations.back().output, 0, 0, {} });
    invocation.dispatches.push_back({
        invocation.kernel,
        {
                 invocation.inputs[0],
                 invocation.inputs[1],
                 invocation.inputs[2],
                 invocation.outputs[0],
                 },
        {}
    });
    schedule.invocations.push_back(invocation);
    REQUIRE(ggml::hrx::verify_schedule(first, schedule).valid());
    const std::string         schedule_json = ggml::hrx::serialize_schedule_json(schedule);
    std::vector<std::string>  schedule_errors;
    const ggml::hrx::Schedule round_trip_schedule =
        ggml::hrx::deserialize_schedule_json(schedule_json, schedule_errors);
    REQUIRE(schedule_errors.empty());
    REQUIRE(ggml::hrx::verify_schedule(first, round_trip_schedule).valid());
    REQUIRE(ggml::hrx::serialize_schedule_json(round_trip_schedule) == schedule_json);
    ggml::hrx::Schedule wrong_dispatch_oracle     = round_trip_schedule;
    wrong_dispatch_oracle.expected_dispatch_count = ggml::hrx::schedule_dispatch_count(wrong_dispatch_oracle) + 1;
    REQUIRE(!ggml::hrx::verify_schedule(first, wrong_dispatch_oracle).valid());
    schedule.invocations[0].covered_operations.pop_back();
    REQUIRE(!ggml::hrx::verify_schedule(first, schedule).valid());
}

static void test_strided_and_empty_view_spans() {
    {
        Fixture       fixture;
        ggml_tensor * input = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 8192, 512);
        ggml_set_input(input);
        ggml_tensor * transposed = ggml_transpose(fixture.context, input);
        ggml_set_output(transposed);
        ggml_build_forward_expand(fixture.graph, transposed);

        const ggml::hrx::Graph graph = ggml::hrx::Graph::import(fixture.graph);
        REQUIRE(graph.valid());
    }
    {
        Fixture       fixture;
        ggml_tensor * input = ggml_new_tensor_1d(fixture.context, GGML_TYPE_F32, 24576);
        ggml_set_input(input);
        ggml_tensor * empty = ggml_view_2d(fixture.context, input, 24576, 0, input->nb[1], ggml_nbytes(input));
        ggml_set_output(empty);
        ggml_build_forward_expand(fixture.graph, empty);

        const ggml::hrx::Graph graph = ggml::hrx::Graph::import(fixture.graph);
        REQUIRE(graph.valid());
    }
}

static void test_set_rows_effects_and_views() {
    Fixture       fixture;
    ggml_tensor * destination = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * rows        = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 2);
    ggml_tensor * indices     = ggml_new_tensor_1d(fixture.context, GGML_TYPE_I32, 2);
    ggml_set_input(destination);
    ggml_set_input(rows);
    ggml_set_input(indices);
    ggml_tensor * result = ggml_set_rows(fixture.context, destination, rows, indices);
    ggml_set_output(result);
    ggml_build_forward_expand(fixture.graph, result);

    const ggml::hrx::Graph graph = ggml::hrx::Graph::import(fixture.graph);
    REQUIRE(graph.valid());
    REQUIRE(graph.operations.size() == 1);
    const ggml::hrx::Operation & operation = graph.operations[0];
    REQUIRE(operation.op == GGML_OP_SET_ROWS);
    REQUIRE(operation.inputs.size() == 3);
    const ggml::hrx::Value & output = graph.values[operation.output];
    REQUIRE(output.view_source != ggml::hrx::kInvalidId);
    REQUIRE(output.access.storage == graph.values[operation.inputs[2]].access.storage);
    REQUIRE(output.access.version == 1);
    const auto write =
        std::find_if(operation.effects.begin(), operation.effects.end(),
                     [](const ggml::hrx::Effect & effect) { return effect.kind == ggml::hrx::EffectKind::Write; });
    REQUIRE(write != operation.effects.end());
    REQUIRE(write->storage == output.access.storage);
    REQUIRE(write->before_version == 0);
    REQUIRE(write->after_version == 1);
    REQUIRE(write->size == graph.storages[write->storage].size);
    REQUIRE(!write->exact);
    REQUIRE(graph.storages[write->storage].mutable_state);

    const ggml::hrx::ProgramPlan mutation_plan = ggml::hrx::build_reactive_plan(graph, "test-target");
    REQUIRE(mutation_plan.valid());
    REQUIRE(mutation_plan.resources.resources[write->storage].imported);
    REQUIRE(mutation_plan.resources.resources[write->storage].exported);
    const auto planned_mutation =
        std::find_if(mutation_plan.resources.uses.begin(), mutation_plan.resources.uses.end(),
                     [&](const ggml::hrx::ResourceUse & use) { return use.storage == write->storage; });
    REQUIRE(planned_mutation != mutation_plan.resources.uses.end());
    REQUIRE(planned_mutation->access == ggml::hrx::ResourceAccess::ReadWrite);
    ggml::hrx::ResourceProgram missing_mutation_export         = mutation_plan.resources;
    missing_mutation_export.resources[write->storage].exported = false;
    REQUIRE(!ggml::hrx::verify_resource_program(graph, mutation_plan.schedule, missing_mutation_export).valid());
    ggml::hrx::ResourceProgram bad_mutation_version = mutation_plan.resources;
    const auto                 mutation_use =
        std::find_if(bad_mutation_version.uses.begin(), bad_mutation_version.uses.end(),
                     [&](const ggml::hrx::ResourceUse & use) { return use.storage == write->storage; });
    REQUIRE(mutation_use != bad_mutation_version.uses.end());
    mutation_use->after_version = graph.storages[write->storage].final_version + 1;
    REQUIRE(!ggml::hrx::verify_resource_program(graph, mutation_plan.schedule, bad_mutation_version).valid());

    ggml_tensor * read_view        = ggml_view_tensor(fixture.context, destination);
    ggml_tensor * increment        = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * read_after_write = ggml_add(fixture.context, read_view, increment);
    ggml_set_output(read_after_write);
    ggml_build_forward_expand(fixture.graph, read_after_write);
    const ggml::hrx::Graph effect_graph = ggml::hrx::Graph::import(fixture.graph);
    REQUIRE(effect_graph.valid());

    const ggml::hrx::OperationId add_id = static_cast<ggml::hrx::OperationId>(effect_graph.operations.size() - 1);
    const auto                   writer =
        std::find_if(effect_graph.operations.begin(), effect_graph.operations.end(),
                     [](const ggml::hrx::Operation & candidate) { return candidate.op == GGML_OP_SET_ROWS; });
    REQUIRE(writer != effect_graph.operations.end());
    const ggml::hrx::OperationId set_rows_id = writer->id;
    const ggml::hrx::GraphIndex  effect_index(effect_graph);
    const auto &                 predecessors = effect_index.predecessors(add_id);
    REQUIRE(std::find(predecessors.begin(), predecessors.end(), set_rows_id) != predecessors.end());
    const ggml::hrx::OperationId view_id = graph.operations.size();
    ggml::hrx::Schedule          reversed;
    reversed.graph_fingerprint = effect_graph.fingerprint;
    ggml::hrx::Invocation consumer;
    consumer.kernel = { "test", "consumer", {}, ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} };
    consumer.covered_operations = { view_id, add_id };
    consumer.inputs             = {
        { "destination", effect_graph.operations[view_id].inputs[0], 0, 0, {} },
        { "increment",   effect_graph.operations[add_id].inputs[1],  0, 0, {} }
    };
    consumer.outputs = {
        { "sum", effect_graph.operations[add_id].output, 0, 0, {} }
    };
    ggml::hrx::Invocation writer_invocation;
    writer_invocation.kernel = { "test", "writer", {}, ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} };
    writer_invocation.covered_operations = { set_rows_id };
    writer_invocation.inputs             = {
        { "rows",        operation.inputs[0], 0, 0, {} },
        { "indices",     operation.inputs[1], 0, 0, {} },
        { "destination", operation.inputs[2], 0, 0, {} }
    };
    writer_invocation.outputs = {
        { "updated", operation.output, 0, 0, {} }
    };
    reversed.invocations = { std::move(consumer), std::move(writer_invocation) };
    REQUIRE(!ggml::hrx::verify_schedule(effect_graph, reversed).valid());
}

static void test_reactive_cache_and_bindings() {
    ggml::hrx::ReactivePlanCache    cache;
    const ggml::hrx::ExecutionFrame null_frame = cache.prepare(nullptr, "test-target");
    REQUIRE(!null_frame.valid());
    REQUIRE(!null_frame.errors.empty());
    REQUIRE(null_frame.errors.front().find("null ggml_cgraph") != std::string::npos);
    REQUIRE(cache.stats().failures == 1);

    Fixture first;
    build_arithmetic_graph(first, "first-runtime");
    first.graph->uid                            = 101;
    const ggml::hrx::ExecutionFrame first_frame = cache.prepare(first.graph, "test-target");
    REQUIRE(first_frame.valid());
    REQUIRE(first_frame.plan->schedule.invocations.size() == 2);
    REQUIRE(!first_frame.plan->warnings.empty());
    REQUIRE(ggml::hrx::schedule_execution_kind_count(first_frame.plan->schedule,
                                                     ggml::hrx::KernelSpecialization::ExecutionKind::NativeEager) == 2);
    REQUIRE(cache.stats().builds == 1);
    REQUIRE(cache.stats().hits == 0);

    const ggml::hrx::ExecutionFrame second_frame = cache.prepare(first.graph, "test-target");
    REQUIRE(second_frame.valid());
    REQUIRE(second_frame.plan == first_frame.plan);
    REQUIRE(cache.stats().builds == 1);
    REQUIRE(cache.stats().hits == 1);
    REQUIRE(first_frame.values.size() == second_frame.values.size());
    REQUIRE(first_frame.values.front() == second_frame.values.front());

    Fixture structural_twin;
    build_arithmetic_graph(structural_twin, "first-runtime");
    structural_twin.graph->uid = 105;
    const ggml::hrx::ExecutionFrame structural_twin_frame =
        cache.prepare(structural_twin.graph, "test-target");
    REQUIRE(structural_twin_frame.valid());
    REQUIRE(structural_twin_frame.plan == first_frame.plan);
    REQUIRE(structural_twin_frame.values.size() == first_frame.values.size());
    REQUIRE(structural_twin_frame.values.front() != first_frame.values.front());
    REQUIRE(cache.stats().builds == 1);
    REQUIRE(cache.stats().hits == 2);

    Fixture same_semantics;
    build_arithmetic_graph(same_semantics, "renamed-runtime");
    same_semantics.graph->uid                            = 102;
    const ggml::hrx::ExecutionFrame same_semantics_frame = cache.prepare(same_semantics.graph, "test-target");
    REQUIRE(same_semantics_frame.valid());
    REQUIRE(same_semantics_frame.plan == first_frame.plan);
    REQUIRE(cache.stats().builds == 1);
    REQUIRE(cache.stats().hits == 3);

    const ggml::hrx::ExecutionFrame wrong_target = cache.prepare(first.graph, "different-target");
    REQUIRE(!wrong_target.valid());
    REQUIRE(!wrong_target.errors.empty());
    REQUIRE(wrong_target.errors.front().find("target") != std::string::npos);
    REQUIRE(cache.stats().failures == 2);

    Fixture missing_uid;
    build_arithmetic_graph(missing_uid, "missing-uid");
    REQUIRE(missing_uid.graph->uid == 0);
    const ggml::hrx::ExecutionFrame missing_uid_frame = cache.prepare(missing_uid.graph, "test-target");
    REQUIRE(missing_uid_frame.valid());
    REQUIRE(cache.stats().builds == 2);
    REQUIRE(cache.stats().hits == 3);
    REQUIRE(cache.stats().failures == 2);
    const ggml::hrx::ExecutionFrame repeated_missing_uid_frame = cache.prepare(missing_uid.graph, "test-target");
    REQUIRE(repeated_missing_uid_frame.valid());
    REQUIRE(repeated_missing_uid_frame.plan != missing_uid_frame.plan);
    REQUIRE(cache.stats().builds == 3);
    REQUIRE(cache.stats().hits == 3);
    REQUIRE(cache.stats().failures == 2);

    Fixture       changed;
    ggml_tensor * x = ggml_new_tensor_2d(changed.context, GGML_TYPE_F32, 4, 9);
    ggml_tensor * y = ggml_new_tensor_2d(changed.context, GGML_TYPE_F32, 4, 9);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_tensor * add = ggml_add(changed.context, x, y);
    ggml_set_output(add);
    ggml_build_forward_expand(changed.graph, add);
    changed.graph->uid = 103;
    REQUIRE(cache.prepare(changed.graph, "test-target").valid());
    REQUIRE(cache.stats().builds == 4);

    ggml::hrx::ReactivePlanCache concurrent_cache;
    first.graph->uid                      = 104;
    std::atomic<int>         valid_frames = 0;
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([&] {
            if (concurrent_cache.prepare(first.graph, "concurrent-target").valid()) {
                ++valid_frames;
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    REQUIRE(valid_frames == 4);
    REQUIRE(concurrent_cache.stats().builds == 1);
    REQUIRE(concurrent_cache.stats().hits == 3);
}

static void test_eager_capabilities_and_resource_verification() {
    REQUIRE(ggml::hrx::eager_capability_declared(GGML_OP_NONE));
    for (enum ggml_op op : {
             GGML_OP_ADD,
             GGML_OP_ARGSORT,
             GGML_OP_CLAMP,
             GGML_OP_CONCAT,
             GGML_OP_CONT,
             GGML_OP_CPY,
             GGML_OP_DIV,
             GGML_OP_FLASH_ATTN_EXT,
             GGML_OP_GATED_DELTA_NET,
             GGML_OP_GET_ROWS,
             GGML_OP_GLU,
             GGML_OP_L2_NORM,
             GGML_OP_MUL,
             GGML_OP_MUL_MAT,
             GGML_OP_MUL_MAT_ID,
             GGML_OP_PERMUTE,
             GGML_OP_RESHAPE,
             GGML_OP_RMS_NORM,
             GGML_OP_ROPE,
             GGML_OP_SCALE,
             GGML_OP_SET_ROWS,
             GGML_OP_SOFT_MAX,
             GGML_OP_SSM_CONV,
             GGML_OP_SUM_ROWS,
             GGML_OP_TRANSPOSE,
             GGML_OP_UNARY,
             GGML_OP_VIEW,
         }) {
        REQUIRE(ggml::hrx::eager_capability_declared(op));
    }
    REQUIRE(!ggml::hrx::eager_capability_declared(GGML_OP_CONV_2D));

    Fixture fixture;
    build_arithmetic_graph(fixture, "resource");
    const ggml::hrx::ImportedGraph imported = ggml::hrx::ImportedGraph::import(fixture.graph);
    REQUIRE(imported.graph.valid());
    const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(imported.graph, "test-target");
    REQUIRE(plan.valid());
    REQUIRE(ggml::hrx::verify_resource_program(plan.graph, plan.schedule, plan.resources).valid());
    const ggml::hrx::StorageId          add_storage = plan.graph.values[plan.graph.operations[0].output].access.storage;
    const ggml::hrx::ResourceContract & add_resource = plan.resources.resources[add_storage];
    REQUIRE(add_resource.elidable);
    REQUIRE(add_resource.first_invocation != UINT32_MAX);
    const auto add_write =
        std::find_if(plan.resources.uses.begin(), plan.resources.uses.end(), [&](const ggml::hrx::ResourceUse & use) {
            return use.storage == add_storage && (use.access == ggml::hrx::ResourceAccess::Write ||
                                                  use.access == ggml::hrx::ResourceAccess::ReadWrite);
        });
    REQUIRE(add_write != plan.resources.uses.end());

    ggml::hrx::ResourceProgram missing_import = plan.resources;
    const auto                 external = std::find_if(missing_import.resources.begin(), missing_import.resources.end(),
                                                       [](const ggml::hrx::ResourceContract & resource) { return resource.imported; });
    REQUIRE(external != missing_import.resources.end());
    external->imported = false;
    REQUIRE(!ggml::hrx::verify_resource_program(plan.graph, plan.schedule, missing_import).valid());

    ggml::hrx::ResourceProgram bad_alias = plan.resources;
    REQUIRE(!bad_alias.resources.empty());
    bad_alias.resources[0].aliases.push_back(plan.graph.values.size());
    REQUIRE(!ggml::hrx::verify_resource_program(plan.graph, plan.schedule, bad_alias).valid());

    ggml::hrx::Graph unknown = imported.graph;
    unknown.operations[0].op = GGML_OP_CONV_2D;
    unknown.fingerprint      = "unknown-op";
    REQUIRE(!ggml::hrx::build_reactive_plan(unknown, "test-target").valid());
}

static ggml::hrx::kernel_corpus make_test_corpus(const ggml::hrx::ProgramPlan & plan) {
    static std::vector<ggml::hrx::kernel_definition>                      kernels;
    static std::vector<std::string>                                       sources;
    static std::vector<std::vector<ggml::hrx::kernel_binding_definition>> bindings;
    static std::vector<ggml::hrx::kernel_source>                          source_records;
    static std::vector<ggml::hrx::kernel_source_ref>                      primary_sources;
    const size_t dispatch_count = ggml::hrx::schedule_dispatch_count(plan.schedule);
    kernels.clear();
    sources.clear();
    bindings.clear();
    source_records.clear();
    primary_sources.clear();
    kernels.reserve(dispatch_count);
    sources.reserve(dispatch_count);
    bindings.reserve(dispatch_count);
    source_records.reserve(dispatch_count);
    primary_sources.reserve(dispatch_count);

    ggml::hrx::kernel_corpus corpus;
    corpus.upstream_revision = "test-revision";
    corpus.plan_case_count   = 1;
    for (const ggml::hrx::Invocation & invocation : plan.schedule.invocations) {
        for (const ggml::hrx::Dispatch & dispatch : invocation.dispatches) {
            if (std::find_if(kernels.begin(), kernels.end(), [&](const ggml::hrx::kernel_definition & item) {
                    return item.family != nullptr && item.name != nullptr && dispatch.kernel.family == item.family &&
                           dispatch.kernel.variant == item.name;
                }) != kernels.end()) {
                continue;
            }
            sources.push_back("test/" + dispatch.kernel.variant + ".loom");
            source_records.emplace_back();
            source_records.back().source.data   = sources.back().c_str();
            source_records.back().source.length = sources.back().size();
            source_records.back().source.format = ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT;
            primary_sources.emplace_back();
            primary_sources.back().path     = sources.back().c_str();
            primary_sources.back().contents = &source_records.back();
            bindings.emplace_back();
            bindings.back().reserve(dispatch.bindings.size());
            for (const ggml::hrx::TensorBinding & binding : dispatch.bindings) {
                bindings.back().push_back({ binding.role.c_str(), ggml::hrx::ResourceAccess::ReadWrite });
            }
            ggml::hrx::kernel_definition kernel;
            kernel.family                         = dispatch.kernel.family.c_str();
            kernel.name                           = dispatch.kernel.variant.c_str();
            kernel.id                             = ggml::hrx::kernel_catalog_id(kernel.family, kernel.name);
            kernel.source                         = sources.back().c_str();
            kernel.symbol                         = dispatch.kernel.variant.c_str();
            kernel.backend                        = "amdgpu";
            kernel.bindings                       = { bindings.back().data(), bindings.back().size() };
            kernel.compile_recipe.mode            = "direct";
            kernel.compile_recipe.primary_sources = { &primary_sources.back(), 1 };
            kernels.push_back(kernel);
        }
    }
    corpus.kernels = { kernels.data(), kernels.size() };
    return corpus;
}

static void test_command_program_and_diagnostics() {
    Fixture fixture;
    build_arithmetic_graph(fixture, "command-program");
    const ggml::hrx::ImportedGraph imported = ggml::hrx::ImportedGraph::import(fixture.graph);
    const ggml::hrx::ProgramPlan   plan     = ggml::hrx::build_reactive_plan(imported.graph, "gfx1151");
    REQUIRE(plan.valid());
    const ggml::hrx::kernel_corpus corpus = make_test_corpus(plan);
    REQUIRE(ggml::hrx::verify_kernel_corpus(corpus).valid());
    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(plan, corpus);
    for (const std::string & error : commands.errors) {
        std::fprintf(stderr, "command program: %s\n", error.c_str());
    }
    REQUIRE(commands.valid());
    REQUIRE(ggml::hrx::verify_command_program(plan, corpus, commands).valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(!commands.transients.allocations.empty());
    REQUIRE(commands.transients.arena_size >= commands.transients.allocations.front().size);

    const std::string command_text = ggml::hrx::format_command_program(commands);
    REQUIRE(command_text == ggml::hrx::format_command_program(commands));
    REQUIRE(command_text.find("command-program ggml-hrx-command-program-v1") != std::string::npos);
    REQUIRE(command_text.find("transients arena=") != std::string::npos);
    REQUIRE(ggml::hrx::serialize_command_program_json(commands) == ggml::hrx::serialize_command_program_json(commands));
    REQUIRE(ggml::hrx::format_kernel_corpus(corpus).find("revision=test-revision") != std::string::npos);
    REQUIRE(ggml::hrx::format_resource_program(plan.resources).find("flags=") != std::string::npos);
    REQUIRE(ggml::hrx::command_program_dot(commands).find("c0 -> c1") != std::string::npos);

    ggml::hrx::kernel_corpus missing_kernel = corpus;
    --missing_kernel.kernels.count;
    REQUIRE(!ggml::hrx::build_command_program(plan, missing_kernel).valid());
    ggml::hrx::CommandProgram bad_range = commands;
    bad_range.commands.front().bindings.front().length =
        plan.graph.storages[bad_range.commands.front().bindings.front().storage].size + 1;
    REQUIRE(!ggml::hrx::verify_command_program(plan, corpus, bad_range).valid());
    ggml::hrx::CommandProgram bad_dependency    = commands;
    bad_dependency.commands.back().dependencies = { bad_dependency.commands.back().ordinal };
    REQUIRE(!ggml::hrx::verify_command_program(plan, corpus, bad_dependency).valid());
    ggml::hrx::CommandProgram bad_abi = commands;
    bad_abi.commands.front().bindings.pop_back();
    REQUIRE(!ggml::hrx::verify_command_program(plan, corpus, bad_abi).valid());

    ggml::hrx::BindingSnapshot snapshot;
    snapshot.device_identity = "gfx1151:0";
    uint64_t identity        = 1;
    for (const ggml::hrx::ResourceContract & resource : plan.resources.resources) {
        if (resource.elidable) {
            continue;
        }
        snapshot.bindings.push_back({ resource.storage, identity++, 1, resource.size, 0, resource.size });
    }
    REQUIRE(ggml::hrx::verify_binding_snapshot(plan, snapshot).valid());
    ggml::hrx::ProgramPlan empty_boundary_plan = plan;
    const ggml::hrx::StorageId empty_boundary_storage = snapshot.bindings.front().storage;
    empty_boundary_plan.graph.storages[empty_boundary_storage].size = 0;
    empty_boundary_plan.resources.resources[empty_boundary_storage].size = 0;
    ggml::hrx::BindingSnapshot empty_boundary_snapshot = snapshot;
    empty_boundary_snapshot.bindings.erase(empty_boundary_snapshot.bindings.begin());
    REQUIRE(ggml::hrx::verify_binding_snapshot(empty_boundary_plan, empty_boundary_snapshot).valid());
    const ggml::hrx::AllocationFingerprint first = ggml::hrx::fingerprint_bindings(snapshot);
    REQUIRE(!first.value.empty());
    REQUIRE(ggml::hrx::format_binding_snapshot(snapshot).find("buffer=<runtime>") != std::string::npos);
    REQUIRE(ggml::hrx::format_binding_snapshot(snapshot).find("0x") == std::string::npos);
    REQUIRE(ggml::hrx::format_binding_snapshot(snapshot, true).find("buffer=0x") != std::string::npos);
    REQUIRE(ggml::hrx::serialize_binding_snapshot_json(snapshot).find("buffer_identity") == std::string::npos);
    snapshot.bindings.front().generation++;
    REQUIRE(ggml::hrx::fingerprint_bindings(snapshot) != first);
    snapshot.bindings.pop_back();
    REQUIRE(!ggml::hrx::verify_binding_snapshot(plan, snapshot).valid());

    Fixture fork_join_fixture;
    build_fork_join_graph(fork_join_fixture, "unordered-alias");
    const ggml::hrx::ImportedGraph fork_join_graph = ggml::hrx::ImportedGraph::import(fork_join_fixture.graph);
    ggml::hrx::ProgramPlan         fork_join_plan  = ggml::hrx::build_reactive_plan(fork_join_graph.graph, "gfx1151");
    REQUIRE(fork_join_plan.valid());
    REQUIRE(fork_join_plan.schedule.invocations.size() == 3);
    fork_join_plan.schedule.invocations[0].dispatches[0].dependencies.clear();
    fork_join_plan.schedule.invocations[1].dispatches[0].dependencies.clear();
    fork_join_plan.schedule.invocations[2].dispatches[0].dependencies = { 0, 1 };
    REQUIRE(ggml::hrx::verify_schedule(fork_join_plan.graph, fork_join_plan.schedule).valid());
    const ggml::hrx::kernel_corpus  fork_join_corpus = make_test_corpus(fork_join_plan);
    const ggml::hrx::CommandProgram fork_join_commands =
        ggml::hrx::build_command_program(fork_join_plan, fork_join_corpus);
    REQUIRE(fork_join_commands.valid());
    REQUIRE(fork_join_commands.commands[0].dependencies.empty());
    REQUIRE(fork_join_commands.commands[1].dependencies.empty());
    REQUIRE(fork_join_commands.commands[2].dependencies == std::vector<uint32_t>({ 0, 1 }));
    REQUIRE(fork_join_commands.transients.allocations.size() == 2);
    const ggml::hrx::TransientAllocation & left  = fork_join_commands.transients.allocations[0];
    const ggml::hrx::TransientAllocation & right = fork_join_commands.transients.allocations[1];
    REQUIRE(left.arena_offset + left.size <= right.arena_offset ||
            right.arena_offset + right.size <= left.arena_offset);
    REQUIRE(ggml::hrx::verify_command_program(fork_join_plan, fork_join_corpus, fork_join_commands).valid());

    ggml::hrx::CommandProgram unordered_alias               = fork_join_commands;
    unordered_alias.transients.allocations[0].first_command = 0;
    unordered_alias.transients.allocations[0].last_command  = 0;
    unordered_alias.transients.allocations[1].first_command = 1;
    unordered_alias.transients.allocations[1].last_command  = 1;
    unordered_alias.transients.allocations[1].arena_offset  = unordered_alias.transients.allocations[0].arena_offset;
    const ggml::hrx::VerificationResult unordered_alias_verification =
        ggml::hrx::verify_command_program(fork_join_plan, fork_join_corpus, unordered_alias);
    REQUIRE(!unordered_alias_verification.valid());
    REQUIRE(std::find(unordered_alias_verification.errors.begin(), unordered_alias_verification.errors.end(),
                      "aliased transient lifetimes overlap in the command dependency graph") !=
            unordered_alias_verification.errors.end());
}

static void test_pinned_kernel_corpus_manifest() {
    const ggml::hrx::kernel_corpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    REQUIRE(ggml::hrx::verify_kernel_corpus(corpus).valid());
    REQUIRE(std::string(corpus.upstream_revision) ==
            "c09218e7ca354654b6c66dddaf80f6294e4748bb+e809fd673bc46d47da5b07f59c0415fe379c4816");
    REQUIRE(corpus.kernels.size() == 133);
    REQUIRE(corpus.storage_transforms.size() == 21);
    REQUIRE(corpus.plan_case_count == 32);

    const std::array<int64_t, GGML_MAX_DIMS> q5_shape = { 512, 2048, 256, 1 };
    const ggml::hrx::kernel_storage_transform * q5_transform = ggml::hrx::match_kernel_storage_transform(
        corpus, "gfx1151", GGML_TYPE_Q5_K, q5_shape, true, "blk.0.ffn_down_exps.weight");
    REQUIRE(q5_transform != nullptr);
    REQUIRE(std::string(q5_transform->name) == "q5_k_expert_down_group4");
    REQUIRE(ggml::hrx::match_kernel_storage_transform(
                corpus, "gfx1151", GGML_TYPE_Q5_K, q5_shape, true, "blk.00.ffn_down_exps.weight") == nullptr);
    REQUIRE(ggml::hrx::match_kernel_storage_transform(
                corpus, "gfx1151", GGML_TYPE_Q5_K, q5_shape, true, "ffn_down_exps.weight") == nullptr);

    const std::array<int64_t, GGML_MAX_DIMS> q4_shape = { 5120, 17408, 1, 1 };
    const ggml::hrx::kernel_storage_transform * q4_transform = ggml::hrx::match_kernel_storage_transform(
        corpus, "gfx1151", GGML_TYPE_Q4_K, q4_shape, true, "");
    REQUIRE(q4_transform != nullptr);
    REQUIRE(std::string(q4_transform->name) == "q4_k_symi4_row64_k64");
    const ggml::hrx::kernel_storage_transform * q4_tensor_plane =
        ggml::hrx::resolve_kernel_storage_transform(
            corpus, "gfx1151", "q4_k_tensor_payload_header8", GGML_TYPE_Q4_K, q4_shape, true, "");
    REQUIRE(q4_tensor_plane != nullptr);
    REQUIRE(std::string(q4_tensor_plane->name) == "q4_k_tensor_payload_header8");
    const ggml::hrx::kernel_storage_transform * q4_symi4_tensor_plane =
        ggml::hrx::resolve_kernel_storage_transform(
            corpus, "gfx1151", "q4_k_symi4_tensor_payload_header8", GGML_TYPE_Q4_K, q4_shape, true, "");
    REQUIRE(q4_symi4_tensor_plane != nullptr);
    REQUIRE(std::string(q4_symi4_tensor_plane->name) == "q4_k_symi4_tensor_payload_header8");

    const std::array<int64_t, GGML_MAX_DIMS> q8_shape = { 4096, 2048, 1, 1 };
    const ggml::hrx::kernel_storage_transform * q8_transform = ggml::hrx::match_kernel_storage_transform(
        corpus, "gfx1151", GGML_TYPE_Q8_0, q8_shape, true, "");
    REQUIRE(q8_transform != nullptr);
    REQUIRE(std::string(q8_transform->name) == "q8_0_tile64_k64");
    const char *                           family  = "qwen3_moe";
    const char *                           name    = "ggml_linear_q6k_q8_1_x4";
    const uint64_t                         id      = ggml::hrx::kernel_catalog_id(family, name);
    const ggml::hrx::kernel_resolve_result gfx1100 = ggml::hrx::resolve_kernel_definition(
        corpus, "gfx1100", family, name, id, ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    const ggml::hrx::kernel_resolve_result gfx1151 = ggml::hrx::resolve_kernel_definition(
        corpus, "gfx1151", family, name, id, ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(gfx1100.found());
    REQUIRE(gfx1151.found());
    REQUIRE(std::string(gfx1100.definition->symbol) == "ggml_linear_q6k_q8_1_x4");
    REQUIRE(std::string(gfx1151.definition->symbol) == "ggml_linear_q6k_q8_1_x4_wave64_block4");

    std::vector<ggml::hrx::kernel_definition> mismatched_variants(corpus.kernels.begin(), corpus.kernels.end());
    const auto mismatched = std::find_if(mismatched_variants.begin(), mismatched_variants.end(),
                                         [](const ggml::hrx::kernel_definition & definition) {
                                             return std::string(definition.name) == "ggml_linear_q6k_q8_1_x4" &&
                                                    std::string(definition.target_selector) == "gfx1151";
                                         });
    REQUIRE(mismatched != mismatched_variants.end());
    mismatched->backend                          = "other-backend";
    ggml::hrx::kernel_corpus mismatched_corpus   = corpus;
    mismatched_corpus.kernels                    = { mismatched_variants.data(), mismatched_variants.size() };
    const ggml::hrx::VerificationResult mismatch = ggml::hrx::verify_kernel_corpus(mismatched_corpus);
    REQUIRE(std::find(mismatch.errors.begin(), mismatch.errors.end(),
                      "kernel target variants disagree on ABI for qwen3_moe:ggml_linear_q6k_q8_1_x4") !=
            mismatch.errors.end());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && std::string(argv[1]) == "--prove-qwen") {
        std::ifstream graph_file(argv[2]);
        REQUIRE(graph_file.good());
        const std::string graph_text((std::istreambuf_iterator<char>(graph_file)), std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::Graph::deserialize_json(graph_text);
        REQUIRE(graph.valid());
        const ggml::hrx::RoutedTransformerModel model =
            ggml::hrx::RoutedTransformerModel::analyze(ggml::hrx::GraphIndex(graph));
        REQUIRE(model.valid());
        const ggml::hrx::ProgramPlan reactive = ggml::hrx::build_reactive_plan(graph, "fixture-target");
        for (const std::string & error : reactive.errors) {
            std::fprintf(stderr, "program plan: %s\n", error.c_str());
        }
        REQUIRE(reactive.valid());
        REQUIRE(reactive.planner_identity.find("llm.routed_transformer@") != std::string::npos);
        REQUIRE(reactive.atom_fallback_count == 0);
        REQUIRE(reactive.schedule.roots.size() == 2);
        REQUIRE(std::all_of(reactive.schedule.roots.begin(), reactive.schedule.roots.end(),
                            [](const ggml::hrx::RootContract & root) {
                                return root.disposition == ggml::hrx::RootDisposition::Materialized;
                            }));
        REQUIRE(ggml::hrx::schedule_execution_kind_count(
                    reactive.schedule, ggml::hrx::KernelSpecialization::ExecutionKind::CpuFallback) == 0);
        if (reactive.schedule.workload.rfind("decode-", 0) == 0) {
            size_t split_dispatch_count = 0;
            size_t qkv_dispatch_count   = 0;
            for (const ggml::hrx::Invocation & invocation : reactive.schedule.invocations) {
                for (const ggml::hrx::Dispatch & dispatch : invocation.dispatches) {
                    if (dispatch.kernel.variant == "qwen3_moe_attention_qkv_postprocess_fused_decode") {
                        ++qkv_dispatch_count;
                        REQUIRE(dispatch.kernel.integer_parameters.at("query_weight_type") == GGML_TYPE_Q4_K);
                        REQUIRE(dispatch.kernel.integer_parameters.at("key_weight_type") == GGML_TYPE_Q4_K);
                        const int64_t value_type = dispatch.kernel.integer_parameters.at("value_weight_type");
                        REQUIRE((value_type == GGML_TYPE_Q4_K || value_type == GGML_TYPE_Q6_K));
                    }
                    if (dispatch.kernel.variant != "qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8") {
                        continue;
                    }
                    ++split_dispatch_count;
                    const int64_t tile_size = dispatch.kernel.integer_parameters.at("key_value_tile_size");
                    REQUIRE(tile_size > 0);
                    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") ==
                            model.key_value_token_count);
                    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_block_count") ==
                            (model.key_value_token_count + tile_size - 1) / tile_size);
                }
            }
            REQUIRE(qkv_dispatch_count == model.blocks.size());
            REQUIRE(split_dispatch_count == model.blocks.size());
            REQUIRE(ggml::hrx::schedule_dispatch_count(reactive.schedule) == model.blocks.size() * 6 + 5);
        } else {
            size_t dispatch_ordinal = 0;
            size_t qkv_fork_count   = 0;
            for (const ggml::hrx::Invocation & invocation : reactive.schedule.invocations) {
                if (invocation.stage == "attention.qkv_publication") {
                    REQUIRE(invocation.dispatches.size() == 4);
                    const std::vector<uint32_t> & predecessor = invocation.dispatches[0].dependencies;
                    REQUIRE(predecessor.size() == 1);
                    REQUIRE(invocation.dispatches[1].dependencies == predecessor);
                    REQUIRE(invocation.dispatches[2].dependencies == predecessor);
                    REQUIRE(invocation.dispatches[3].dependencies == std::vector<uint32_t>({
                                                                         static_cast<uint32_t>(dispatch_ordinal),
                                                                         static_cast<uint32_t>(dispatch_ordinal + 1),
                                                                         static_cast<uint32_t>(dispatch_ordinal + 2),
                                                                     }));
                    ++qkv_fork_count;
                }
                dispatch_ordinal += invocation.dispatches.size();
            }
            REQUIRE(qkv_fork_count == model.blocks.size());
        }
        const std::string         serialized = ggml::hrx::serialize_schedule_json(reactive.schedule);
        std::vector<std::string>  round_trip_errors;
        const ggml::hrx::Schedule round_trip = ggml::hrx::deserialize_schedule_json(serialized, round_trip_errors);
        REQUIRE(round_trip_errors.empty());
        REQUIRE(ggml::hrx::serialize_schedule_json(round_trip) == serialized);
        REQUIRE(ggml::hrx::verify_schedule(reactive.graph, round_trip).valid());

        REQUIRE(!reactive.fusion_search_text.empty());
        REQUIRE(!reactive.fusion_search_json.empty());
        REQUIRE(reactive.fusion_regions_dot.find("digraph fusion_regions") != std::string::npos);
        REQUIRE(reactive.logical_program_text.find("schema=ggml-hrx-logical-routed-transformer-v1") == 0);
        REQUIRE(reactive.logical_program_json.find("\"components\":[") != std::string::npos);
        REQUIRE(reactive.logical_program_dot.find("digraph logical_program") != std::string::npos);
        REQUIRE(reactive.semantic_witness.find(reactive.schedule.workload) != std::string::npos);
        REQUIRE(reactive.graph.values.size() > graph.values.size());
        REQUIRE(ggml::hrx::verify_resource_program(reactive.graph, reactive.schedule, reactive.resources).valid());
        // Kernel recipe parameters must be witnesses of the graph, not model
        // constants hidden in the selector. Check both logical top-k width and
        // its independently recovered physical row stride.
        const size_t             hidden_size    = model.hidden_size;
        const size_t             query_size     = model.query_size;
        const size_t             key_value_size = model.key_value_size;
        const size_t             expert_count   = model.expert_count;
        const ggml::hrx::Value & route_ids      = graph.values[model.blocks.front().values_by_role.router_route_ids];
        const size_t             route_count    = route_ids.access.shape[0];
        const size_t             route_stride   = route_ids.access.strides[1] / ggml_type_size(route_ids.type);
        size_t                   checked_routes = 0;
        size_t                   checked_expert_tables = 0;
        for (const ggml::hrx::Invocation & invocation : reactive.schedule.invocations) {
            for (const ggml::hrx::Dispatch & dispatch : invocation.dispatches) {
                const auto & config = dispatch.kernel.compile_parameters;
                if (config.count("qwen3_moe.model.hidden_size")) {
                    REQUIRE(config.at("qwen3_moe.model.hidden_size") == std::to_string(hidden_size));
                }
                if (config.count("qwen3_moe.attention.query_size")) {
                    REQUIRE(config.at("qwen3_moe.attention.query_size") == std::to_string(query_size));
                }
                if (config.count("qwen3_moe.attention.key_value_size")) {
                    REQUIRE(config.at("qwen3_moe.attention.key_value_size") == std::to_string(key_value_size));
                }
                if (config.count("qwen3_moe.router.expert_count")) {
                    REQUIRE(config.at("qwen3_moe.router.expert_count") == std::to_string(expert_count));
                }
                if (config.count("qwen3_moe.router.route_count")) {
                    REQUIRE(config.at("qwen3_moe.router.route_count") == std::to_string(route_count));
                }
                const auto stride = dispatch.kernel.integer_parameters.find("route_id_stride");
                if (stride != dispatch.kernel.integer_parameters.end()) {
                    REQUIRE(stride->second == static_cast<int64_t>(route_stride));
                    ++checked_routes;
                }
                if (dispatch.kernel.variant == "qwen3_moe_build_expert_table") {
                    REQUIRE(config.at("qwen3_moe.routed_gate_up.route_count") == std::to_string(route_count));
                    REQUIRE(config.at("qwen3_moe.routed_gate_up.expert_count") == std::to_string(expert_count));
                    ++checked_expert_tables;
                }
            }
        }
        REQUIRE(checked_routes != 0);
        if (reactive.schedule.workload.rfind("prefill-", 0) == 0) {
            REQUIRE(checked_expert_tables == model.blocks.size());
        }
        const ggml::hrx::kernel_corpus & executable_corpus = ggml::hrx::get_qwen_kernel_corpus();
        const ggml::hrx::CommandProgram  executable_commands =
            ggml::hrx::build_command_program(reactive, executable_corpus);
        const size_t kernel_command_count = std::count_if(
            executable_commands.commands.begin(), executable_commands.commands.end(),
            [](const ggml::hrx::Command & command) { return command.kind == ggml::hrx::CommandKind::Kernel; });
        const size_t copy_command_count = std::count_if(
            executable_commands.commands.begin(), executable_commands.commands.end(),
            [](const ggml::hrx::Command & command) { return command.kind == ggml::hrx::CommandKind::Copy; });
        const size_t fill_command_count = std::count_if(
            executable_commands.commands.begin(), executable_commands.commands.end(),
            [](const ggml::hrx::Command & command) { return command.kind == ggml::hrx::CommandKind::Fill; });
        REQUIRE(kernel_command_count == ggml::hrx::schedule_dispatch_count(reactive.schedule));
        REQUIRE(copy_command_count == 0);
        REQUIRE(fill_command_count == 0);
        REQUIRE(executable_commands.initializations.size() == 1);
        REQUIRE(executable_commands.initializations[0].data.size() == 256);
        const auto initialized_allocation =
            std::find_if(executable_commands.persistent_constants.allocations.begin(),
                         executable_commands.persistent_constants.allocations.end(),
                         [&](const ggml::hrx::PersistentConstantAllocation & allocation) {
                             return allocation.storage == executable_commands.initializations[0].storage;
                         });
        REQUIRE(initialized_allocation != executable_commands.persistent_constants.allocations.end());
        REQUIRE(initialized_allocation->size == 256);
        REQUIRE(executable_commands.persistent_constants.arena_size >= initialized_allocation->size);
        REQUIRE(std::none_of(executable_commands.transients.allocations.begin(),
                             executable_commands.transients.allocations.end(),
                             [&](const ggml::hrx::TransientAllocation & allocation) {
                                 return allocation.storage == initialized_allocation->storage;
                             }));
        size_t persistent_binding_count = 0;
        for (const ggml::hrx::Command & command : executable_commands.commands) {
            for (const ggml::hrx::CommandBinding & binding : command.bindings) {
                if (binding.storage != initialized_allocation->storage) {
                    continue;
                }
                REQUIRE(binding.origin == ggml::hrx::BindingOrigin::PersistentConstant);
                REQUIRE(binding.persistent_constant == initialized_allocation->id);
                ++persistent_binding_count;
            }
        }
        REQUIRE(persistent_binding_count != 0);
        REQUIRE(ggml::hrx::format_command_program(executable_commands).find("persistent-constants arena=256") !=
                std::string::npos);
        REQUIRE(
            ggml::hrx::serialize_command_program_json(executable_commands).find("\"origin\":\"persistent_constant\"") !=
            std::string::npos);
        REQUIRE(ggml::hrx::verify_command_program(reactive, executable_corpus, executable_commands).valid());
        ggml::hrx::CommandProgram missing_persistent = executable_commands;
        missing_persistent.persistent_constants.allocations.clear();
        REQUIRE(!ggml::hrx::verify_command_program(reactive, executable_corpus, missing_persistent).valid());
        ggml::hrx::CommandProgram missing_initialization = executable_commands;
        missing_initialization.initializations.clear();
        REQUIRE(!ggml::hrx::verify_command_program(reactive, executable_corpus, missing_initialization).valid());
        ggml::hrx::CommandProgram writable_persistent       = executable_commands;
        bool                      changed_persistent_access = false;
        for (ggml::hrx::Command & command : writable_persistent.commands) {
            for (ggml::hrx::CommandBinding & binding : command.bindings) {
                if (binding.origin != ggml::hrx::BindingOrigin::PersistentConstant) {
                    continue;
                }
                binding.access            = ggml::hrx::ResourceAccess::ReadWrite;
                changed_persistent_access = true;
                break;
            }
            if (changed_persistent_access) {
                break;
            }
        }
        REQUIRE(changed_persistent_access);
        REQUIRE(!ggml::hrx::verify_command_program(reactive, executable_corpus, writable_persistent).valid());
        ggml::hrx::CommandProgram      aliased_persistent = executable_commands;
        ggml::hrx::TransientAllocation alias;
        alias.id            = static_cast<uint32_t>(aliased_persistent.transients.allocations.size());
        alias.storage       = initialized_allocation->storage;
        alias.size          = initialized_allocation->size;
        alias.alignment     = initialized_allocation->alignment;
        alias.first_command = 0;
        alias.last_command  = 0;
        aliased_persistent.transients.allocations.push_back(alias);
        REQUIRE(!ggml::hrx::verify_command_program(reactive, executable_corpus, aliased_persistent).valid());

        ggml::hrx::Schedule missing_operation = reactive.schedule;
        missing_operation.invocations[1].covered_operations.pop_back();
        REQUIRE(!ggml::hrx::verify_schedule(reactive.graph, missing_operation).valid());
        ggml::hrx::Schedule wrong_dispatch_count = reactive.schedule;
        wrong_dispatch_count.expected_dispatch_count++;
        REQUIRE(!ggml::hrx::verify_schedule(reactive.graph, wrong_dispatch_count).valid());
        ggml::hrx::Schedule wrong_binding                            = reactive.schedule;
        wrong_binding.invocations[1].dispatches[0].bindings[0].value = reactive.graph.values.size();
        REQUIRE(!ggml::hrx::verify_schedule(reactive.graph, wrong_binding).valid());
        ggml::hrx::Schedule bad_root = reactive.schedule;
        bad_root.roots[0].value      = reactive.graph.values.size();
        REQUIRE(!ggml::hrx::verify_schedule(reactive.graph, bad_root).valid());
        ggml::hrx::Graph corrupted_graph                                                                  = graph;
        corrupted_graph.operations[model.blocks.front().operations_by_role.attention_query_projection].op = GGML_OP_ADD;
        REQUIRE(!ggml::hrx::RoutedTransformerModel::analyze(ggml::hrx::GraphIndex(corrupted_graph)).valid());
        ggml::hrx::Graph             mismatched_kv_graph = graph;
        const ggml::hrx::OperationId first_flash         = model.blocks.front().operations_by_role.attention_flash;
        const ggml::hrx::ValueId     first_mask          = mismatched_kv_graph.operations[first_flash].inputs[3];
        mismatched_kv_graph.values[first_mask].access.shape[0] -= 64;
        REQUIRE(!ggml::hrx::RoutedTransformerModel::analyze(ggml::hrx::GraphIndex(mismatched_kv_graph)).valid());
        ggml::hrx::Graph         mismatched_route_layout = graph;
        const ggml::hrx::ValueId first_routes            = model.blocks.front().values_by_role.router_route_ids;
        mismatched_route_layout.values[first_routes].access.strides[1] +=
            ggml_type_size(mismatched_route_layout.values[first_routes].type);
        const ggml::hrx::ProgramPlan rejected_routes =
            ggml::hrx::build_reactive_plan(mismatched_route_layout, "fixture-target");
        REQUIRE(!rejected_routes.valid());
        REQUIRE(std::any_of(
            rejected_routes.errors.begin(), rejected_routes.errors.end(),
            [](const std::string & error) { return error.find("route layout mismatch") != std::string::npos; }));
        std::ofstream schedule_file(argv[3], std::ios::trunc);
        REQUIRE(schedule_file.good());
        schedule_file << serialized << '\n';
        std::ofstream signature_file(argv[4], std::ios::trunc);
        REQUIRE(signature_file.good());
        signature_file << reactive.semantic_witness;
        std::printf("proved %s topology: %zu operations, %zu dispatches, zero CPU fallback and zero atom fallback\n",
                    reactive.schedule.workload.c_str(), graph.operations.size(),
                    ggml::hrx::schedule_dispatch_count(reactive.schedule));
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--materialize") {
        std::ifstream graph_file(argv[2]);
        REQUIRE(graph_file.good());
        const std::string graph_text((std::istreambuf_iterator<char>(graph_file)), std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::Graph::deserialize_json(graph_text);
        REQUIRE(graph.valid());
        const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(graph, "fixture-target");
        REQUIRE(plan.valid());
        std::ofstream output(argv[3], std::ios::trunc);
        REQUIRE(output.good());
        output << ggml::hrx::serialize_schedule_json(plan.schedule) << '\n';
        std::printf("materialized %zu invocations and %zu dispatches with %zu atom fallbacks for graph %s\n",
                    plan.schedule.invocations.size(), ggml::hrx::schedule_dispatch_count(plan.schedule),
                    plan.atom_fallback_count, graph.fingerprint.c_str());
        return 0;
    }
    if (argc == 3) {
        std::ifstream graph_file(argv[1]);
        std::ifstream schedule_file(argv[2]);
        REQUIRE(graph_file.good());
        REQUIRE(schedule_file.good());
        const std::string graph_text((std::istreambuf_iterator<char>(graph_file)), std::istreambuf_iterator<char>());
        const std::string schedule_text((std::istreambuf_iterator<char>(schedule_file)),
                                        std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::Graph::deserialize_json(graph_text);
        REQUIRE(graph.valid());
        std::vector<std::string>  errors;
        const ggml::hrx::Schedule schedule = ggml::hrx::deserialize_schedule_json(schedule_text, errors);
        REQUIRE(errors.empty());
        REQUIRE(ggml::hrx::verify_schedule(graph, schedule).valid());
        std::printf("verified %zu operations in %zu invocations for graph %s\n", graph.operations.size(),
                    schedule.invocations.size(), graph.fingerprint.c_str());
        return 0;
    }
    REQUIRE(argc == 1);
    test_deterministic_import_and_schedule();
    test_strided_and_empty_view_spans();
    test_set_rows_effects_and_views();
    test_reactive_cache_and_bindings();
    test_eager_capabilities_and_resource_verification();
    test_command_program_and_diagnostics();
    test_pinned_kernel_corpus_manifest();
    return 0;
}
