#include "command-program.h"
#include "graph-ir.h"
#include "kernel-corpus.h"
#include "matcher.h"
#include "optimizer.h"
#include "qwen-rules.h"
#include "qwen-program.h"
#include "reactive-plan.h"
#include "schedule.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

namespace {

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

struct Fixture {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;

    Fixture() {
        ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
        context = ggml_init(params);
        REQUIRE(context != nullptr);
        graph = ggml_new_graph_custom(context, 64, false);
    }

    ~Fixture() {
        ggml_free(context);
    }
};

static ggml::hrx::Graph make_arithmetic_graph(const char * prefix) {
    Fixture fixture;
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
    return ggml::hrx::import_graph(fixture.graph);
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

static void test_deterministic_import_and_matcher() {
    const ggml::hrx::Graph first = make_arithmetic_graph("first");
    const ggml::hrx::Graph second = make_arithmetic_graph("second");
    REQUIRE(first.valid());
    REQUIRE(first.operations.size() == 2);
    REQUIRE(first.fingerprint == second.fingerprint);
    const std::string graph_json = ggml::hrx::serialize_graph_json(first);
    REQUIRE(graph_json == ggml::hrx::serialize_graph_json(first));
    REQUIRE(graph_json.find("\"params\":") != std::string::npos);
    REQUIRE(graph_json.find("\"roots\":[") != std::string::npos);
    const ggml::hrx::Graph round_trip_graph = ggml::hrx::deserialize_graph_json(graph_json);
    REQUIRE(round_trip_graph.valid());
    REQUIRE(ggml::hrx::serialize_graph_json(round_trip_graph) == graph_json);

    ggml::hrx::MatchAutomaton automaton;
    automaton.name = "add-mul";
    automaton.root_state = 0;
    automaton.states = {
        { "mul", { { GGML_OP_MUL }, GGML_TYPE_F32, 2 }, { { ggml::hrx::Transition::Kind::InputProducer, 0, 1 } } },
        { "add", { { GGML_OP_ADD }, GGML_TYPE_F32, 2 }, {} },
    };
    const ggml::hrx::Match match = ggml::hrx::match_automaton(first, automaton, 1);
    REQUIRE(match.found());
    REQUIRE(match.state_operations[0] == 1);
    REQUIRE(match.state_operations[1] == 0);
    automaton.states[1].constraint.alternatives = { GGML_OP_SUB };
    REQUIRE(!ggml::hrx::match_automaton(first, automaton, 1).found());

    automaton.states[1].constraint.alternatives = { GGML_OP_ADD };
    ggml::hrx::MatchAutomaton add_only;
    add_only.name = "add";
    add_only.states = { { "add", { { GGML_OP_ADD }, GGML_TYPE_F32, 2 }, {} } };
    ggml::hrx::MatchAutomaton mul_only;
    mul_only.name = "mul";
    mul_only.states = { { "mul", { { GGML_OP_MUL }, GGML_TYPE_F32, 2 }, {} } };
    const std::vector<ggml::hrx::FusionRule> rules = {
        { automaton, { "test", "fused_add_mul", {},
            ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} }, 100 },
        { add_only, { "test", "add", {},
            ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} }, 1 },
        { mul_only, { "test", "mul", {},
            ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} }, 1 },
    };
    const ggml::hrx::Selection selection = ggml::hrx::select_regions(first, rules);
    REQUIRE(selection.regions.size() == 1);
    REQUIRE(selection.regions[0].operations.size() == 2);
    REQUIRE(selection.uncovered_operations.empty());
    REQUIRE(ggml::hrx::verify_schedule(first, ggml::hrx::materialize_schedule(first, rules, selection)).valid());
    const ggml::hrx::Schedule materialized = ggml::hrx::materialize_schedule(first, rules, selection);
    const std::string schedule_json = ggml::hrx::serialize_schedule_json(materialized);
    REQUIRE(schedule_json.find("fused_add_mul") != std::string::npos);
    std::vector<std::string> schedule_errors;
    const ggml::hrx::Schedule round_trip_schedule = ggml::hrx::deserialize_schedule_json(schedule_json, schedule_errors);
    REQUIRE(schedule_errors.empty());
    REQUIRE(ggml::hrx::verify_schedule(first, round_trip_schedule).valid());
    REQUIRE(ggml::hrx::serialize_schedule_json(round_trip_schedule) == schedule_json);
    ggml::hrx::Schedule wrong_dispatch_oracle = round_trip_schedule;
    wrong_dispatch_oracle.expected_dispatch_count = ggml::hrx::schedule_dispatch_count(wrong_dispatch_oracle) + 1;
    REQUIRE(!ggml::hrx::verify_schedule(first, wrong_dispatch_oracle).valid());

    ggml::hrx::MatchAutomaton consumer_pattern;
    consumer_pattern.name = "consumer-edge";
    consumer_pattern.require_internal_single_use = false;
    consumer_pattern.states = {
        { "producer_context", { { GGML_OP_ADD }, GGML_TYPE_F32, 2 }, {
            { ggml::hrx::Transition::Kind::OutputConsumer, 0, 1 },
        }, false },
        { "consumer", { { GGML_OP_MUL }, GGML_TYPE_F32, 2 }, {} },
    };
    const ggml::hrx::Match consumer_match = ggml::hrx::match_automaton(first, consumer_pattern, 0);
    REQUIRE(consumer_match.found());
    REQUIRE(consumer_match.covered_operations == std::vector<ggml::hrx::OperationId> { 1 });

    ggml::hrx::Schedule schedule;
    schedule.graph_fingerprint = first.fingerprint;
    ggml::hrx::Invocation invocation;
    invocation.kernel = { "test", "fused_add_mul", { { "width", 4 } },
        ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} };
    invocation.covered_operations = { 0, 1 };
    for (ggml::hrx::ValueId value = 0; value < first.values.size(); ++value) {
        if (first.values[value].producer == ggml::hrx::kInvalidId) {
            invocation.inputs.push_back({ "input" + std::to_string(value), value });
        }
    }
    invocation.outputs.push_back({ "result", first.operations.back().output });
    invocation.dispatches.push_back({ invocation.kernel, {
        invocation.inputs[0], invocation.inputs[1], invocation.inputs[2], invocation.outputs[0],
    }, {} });
    schedule.invocations.push_back(invocation);
    REQUIRE(ggml::hrx::verify_schedule(first, schedule).valid());
    schedule.invocations[0].covered_operations.pop_back();
    REQUIRE(!ggml::hrx::verify_schedule(first, schedule).valid());
}

static void test_set_rows_effects_and_views() {
    Fixture fixture;
    ggml_tensor * destination = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * rows = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 2);
    ggml_tensor * indices = ggml_new_tensor_1d(fixture.context, GGML_TYPE_I32, 2);
    ggml_set_input(destination);
    ggml_set_input(rows);
    ggml_set_input(indices);
    ggml_tensor * result = ggml_set_rows(fixture.context, destination, rows, indices);
    ggml_set_output(result);
    ggml_build_forward_expand(fixture.graph, result);

    const ggml::hrx::Graph graph = ggml::hrx::import_graph(fixture.graph);
    REQUIRE(graph.valid());
    REQUIRE(graph.operations.size() == 1);
    const ggml::hrx::Operation & operation = graph.operations[0];
    REQUIRE(operation.op == GGML_OP_SET_ROWS);
    REQUIRE(operation.inputs.size() == 3);
    const ggml::hrx::Value & output = graph.values[operation.output];
    REQUIRE(output.view_source != ggml::hrx::kInvalidId);
    REQUIRE(output.access.storage == graph.values[operation.inputs[2]].access.storage);
    REQUIRE(output.access.version == 1);
    const auto write = std::find_if(operation.effects.begin(), operation.effects.end(), [](const ggml::hrx::Effect & effect) {
        return effect.kind == ggml::hrx::EffectKind::Write;
    });
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
    const auto planned_mutation = std::find_if(mutation_plan.resources.uses.begin(), mutation_plan.resources.uses.end(),
        [&](const ggml::hrx::ResourceUse & use) { return use.storage == write->storage; });
    REQUIRE(planned_mutation != mutation_plan.resources.uses.end());
    REQUIRE(planned_mutation->access == ggml::hrx::ResourceAccess::ReadWrite);
    ggml::hrx::ResourceProgram missing_mutation_export = mutation_plan.resources;
    missing_mutation_export.resources[write->storage].exported = false;
    REQUIRE(!ggml::hrx::verify_resource_program(graph, mutation_plan.schedule, missing_mutation_export).valid());
    ggml::hrx::ResourceProgram bad_mutation_version = mutation_plan.resources;
    const auto mutation_use = std::find_if(bad_mutation_version.uses.begin(), bad_mutation_version.uses.end(),
        [&](const ggml::hrx::ResourceUse & use) { return use.storage == write->storage; });
    REQUIRE(mutation_use != bad_mutation_version.uses.end());
    mutation_use->after_version = graph.storages[write->storage].final_version + 1;
    REQUIRE(!ggml::hrx::verify_resource_program(graph, mutation_plan.schedule, bad_mutation_version).valid());

    ggml_tensor * read_view = ggml_view_tensor(fixture.context, destination);
    ggml_tensor * increment = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * read_after_write = ggml_add(fixture.context, read_view, increment);
    ggml_set_output(read_after_write);
    ggml_build_forward_expand(fixture.graph, read_after_write);
    const ggml::hrx::Graph effect_graph = ggml::hrx::import_graph(fixture.graph);
    REQUIRE(effect_graph.valid());

    ggml::hrx::MatchAutomaton effect_automaton;
    effect_automaton.name = "read-after-set-rows";
    effect_automaton.require_internal_single_use = false;
    effect_automaton.states = {
        { "consumer", { { GGML_OP_ADD }, GGML_TYPE_F32, 2 }, {
            { ggml::hrx::Transition::Kind::StorageWriter, 0, 1 },
        } },
        { "writer", { { GGML_OP_SET_ROWS }, GGML_TYPE_F32, 2 }, {} },
    };
    const ggml::hrx::Match effect_match = ggml::hrx::match_automaton(
        effect_graph, effect_automaton, static_cast<ggml::hrx::OperationId>(effect_graph.operations.size() - 1));
    REQUIRE(effect_match.found());
    REQUIRE(effect_graph.operations[effect_match.state_operations[1]].op == GGML_OP_SET_ROWS);

    const ggml::hrx::OperationId set_rows_id = effect_match.state_operations[1];
    const ggml::hrx::OperationId add_id = effect_match.state_operations[0];
    const ggml::hrx::OperationId view_id = graph.operations.size();
    ggml::hrx::Schedule reversed;
    reversed.graph_fingerprint = effect_graph.fingerprint;
    reversed.invocations = {
        { { "test", "consumer", {}, ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} }, { view_id, add_id },
          { { "destination", effect_graph.operations[view_id].inputs[0] }, { "increment", effect_graph.operations[add_id].inputs[1] } },
          { { "sum", effect_graph.operations[add_id].output } }, {}, "", -1 },
        { { "test", "writer", {}, ggml::hrx::KernelSpecialization::ExecutionKind::Native, {} }, { set_rows_id },
          { { "rows", operation.inputs[0] }, { "indices", operation.inputs[1] }, { "destination", operation.inputs[2] } },
          { { "updated", operation.output } }, {}, "", -1 },
    };
    REQUIRE(!ggml::hrx::verify_schedule(effect_graph, reversed).valid());
}

static void test_qwen_multi_output_rules() {
    ggml::hrx::Graph graph;
    auto leaf = [&](enum ggml_type type, std::vector<int64_t> shape) {
        ggml::hrx::Value value;
        value.id = graph.values.size();
        value.type = type;
        value.op = GGML_OP_NONE;
        value.access.storage = graph.storages.size();
        value.access.shape.fill(1);
        for (size_t i = 0; i < shape.size(); ++i) value.access.shape[i] = shape[i];
        graph.storages.push_back({ static_cast<ggml::hrx::StorageId>(graph.storages.size()), value.id });
        graph.values.push_back(value);
        return value.id;
    };
    auto operation = [&](enum ggml_op op, std::vector<ggml::hrx::ValueId> inputs, std::vector<int64_t> shape) {
        const ggml::hrx::ValueId output = leaf(GGML_TYPE_F32, std::move(shape));
        ggml::hrx::Operation node;
        node.id = graph.operations.size();
        node.op = op;
        node.inputs = std::move(inputs);
        node.output = output;
        graph.values[output].op = op;
        graph.values[output].producer = node.id;
        graph.operations.push_back(std::move(node));
        return output;
    };
    const ggml::hrx::ValueId activation = leaf(GGML_TYPE_F32, { 2048, 1 });
    const ggml::hrx::ValueId scale = leaf(GGML_TYPE_F32, { 2048 });
    const ggml::hrx::ValueId prepared = operation(GGML_OP_MUL, { activation, scale }, { 2048, 1 });
    auto projection = [&](int64_t width, int64_t heads, bool normalized) {
        ggml::hrx::ValueId weight = leaf(GGML_TYPE_Q4_K, { 2048, width });
        ggml::hrx::ValueId value = operation(GGML_OP_MUL_MAT, { weight, prepared }, { width, 1 });
        value = operation(GGML_OP_RESHAPE, { value }, { 128, heads });
        if (normalized) {
            value = operation(GGML_OP_RMS_NORM, { value }, { 128, heads });
            value = operation(GGML_OP_MUL, { value, leaf(GGML_TYPE_F32, { 128 }) }, { 128, heads });
            value = operation(GGML_OP_ROPE, { value, leaf(GGML_TYPE_I32, { 1 }) }, { 128, heads });
        }
        return value;
    };
    projection(4096, 32, true);
    ggml::hrx::ValueId key = projection(512, 4, true);
    key = operation(GGML_OP_VIEW, { key }, { 512, 1 });
    operation(GGML_OP_SET_ROWS, { key, leaf(GGML_TYPE_I64, { 1 }), leaf(GGML_TYPE_F16, { 512, 256 }) }, { 512, 256 });
    ggml::hrx::ValueId value = projection(512, 4, false);
    value = operation(GGML_OP_VIEW, { value }, { 512, 1 });
    operation(GGML_OP_SET_ROWS, { value, leaf(GGML_TYPE_I64, { 1 }), leaf(GGML_TYPE_F16, { 512, 256 }) }, { 512, 256 });

    const std::vector<ggml::hrx::FusionRule> rules = ggml::hrx::canonical_qwen3_moe_rules();
    const ggml::hrx::Selection selection = ggml::hrx::select_regions(graph, rules);
    REQUIRE(selection.regions.size() == 2);
    REQUIRE(selection.regions[0].operations.size() == 3);
    REQUIRE(selection.regions[1].operations.size() == 13);
    REQUIRE(selection.uncovered_operations == std::vector<ggml::hrx::OperationId> { 0 });
    const ggml::hrx::Schedule fallback_schedule = ggml::hrx::materialize_schedule_with_cpu_fallback(graph, rules, selection);
    graph.fingerprint.clear();
    REQUIRE(ggml::hrx::verify_schedule(graph, fallback_schedule).valid());
    REQUIRE(fallback_schedule.invocations.front().kernel.execution_kind ==
        ggml::hrx::KernelSpecialization::ExecutionKind::CpuFallback);
}

static void test_reactive_cache_and_bindings() {
    ggml::hrx::ReactivePlanCache cache;
    Fixture first;
    build_arithmetic_graph(first, "first-runtime");
    const ggml::hrx::ExecutionFrame first_frame = cache.prepare(first.graph, "test-target");
    REQUIRE(first_frame.valid());
    REQUIRE(first_frame.plan->schedule.invocations.size() == 2);
    REQUIRE(ggml::hrx::schedule_execution_kind_count(first_frame.plan->schedule,
        ggml::hrx::KernelSpecialization::ExecutionKind::NativeEager) == 2);
    REQUIRE(cache.stats().builds == 1);
    REQUIRE(cache.stats().hits == 0);

    Fixture second;
    build_arithmetic_graph(second, "renamed-runtime");
    const ggml::hrx::ExecutionFrame second_frame = cache.prepare(second.graph, "test-target");
    REQUIRE(second_frame.valid());
    REQUIRE(second_frame.plan == first_frame.plan);
    REQUIRE(cache.stats().builds == 1);
    REQUIRE(cache.stats().hits == 1);
    REQUIRE(first_frame.values.size() == second_frame.values.size());
    REQUIRE(first_frame.values.front() != second_frame.values.front());

    Fixture changed;
    ggml_tensor * x = ggml_new_tensor_2d(changed.context, GGML_TYPE_F32, 4, 9);
    ggml_tensor * y = ggml_new_tensor_2d(changed.context, GGML_TYPE_F32, 4, 9);
    ggml_set_input(x);
    ggml_set_input(y);
    ggml_tensor * add = ggml_add(changed.context, x, y);
    ggml_set_output(add);
    ggml_build_forward_expand(changed.graph, add);
    REQUIRE(cache.prepare(changed.graph, "test-target").valid());
    REQUIRE(cache.stats().builds == 2);

    ggml::hrx::ReactivePlanCache concurrent_cache;
    std::atomic<int> valid_frames = 0;
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([&] {
            if (concurrent_cache.prepare(first.graph, "concurrent-target").valid()) ++valid_frames;
        });
    }
    for (std::thread & worker : workers) worker.join();
    REQUIRE(valid_frames == 4);
    REQUIRE(concurrent_cache.stats().builds == 1);
    REQUIRE(concurrent_cache.stats().hits == 3);
}

static void test_eager_capabilities_and_resource_verification() {
    REQUIRE(ggml::hrx::eager_capability_declared(GGML_OP_NONE));
    for (enum ggml_op op : {
        GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_GET_ROWS, GGML_OP_GLU, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID,
        GGML_OP_PERMUTE, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_ROPE, GGML_OP_SET_ROWS,
        GGML_OP_SOFT_MAX, GGML_OP_SUM_ROWS, GGML_OP_VIEW,
    }) REQUIRE(ggml::hrx::eager_capability_declared(op));
    REQUIRE(!ggml::hrx::eager_capability_declared(GGML_OP_CONV_2D));

    Fixture fixture;
    build_arithmetic_graph(fixture, "resource");
    const ggml::hrx::ImportedGraph imported = ggml::hrx::import_graph_with_bindings(fixture.graph);
    REQUIRE(imported.graph.valid());
    const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(imported.graph, "test-target");
    REQUIRE(plan.valid());
    REQUIRE(ggml::hrx::verify_resource_program(plan.graph, plan.schedule, plan.resources).valid());
    const ggml::hrx::StorageId add_storage = plan.graph.values[plan.graph.operations[0].output].access.storage;
    const ggml::hrx::ResourceContract & add_resource = plan.resources.resources[add_storage];
    REQUIRE(add_resource.elidable);
    REQUIRE(add_resource.first_invocation != UINT32_MAX);
    const auto add_write = std::find_if(plan.resources.uses.begin(), plan.resources.uses.end(),
        [&](const ggml::hrx::ResourceUse & use) {
            return use.storage == add_storage &&
                (use.access == ggml::hrx::ResourceAccess::Write || use.access == ggml::hrx::ResourceAccess::ReadWrite);
        });
    REQUIRE(add_write != plan.resources.uses.end());

    ggml::hrx::ResourceProgram missing_import = plan.resources;
    const auto external = std::find_if(missing_import.resources.begin(), missing_import.resources.end(),
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
    unknown.fingerprint = "unknown-op";
    REQUIRE(!ggml::hrx::build_reactive_plan(unknown, "test-target").valid());
}

static ggml::hrx::KernelCorpus make_test_corpus(const ggml::hrx::ProgramPlan & plan) {
    static std::vector<ggml::hrx::KernelDefinition> kernels;
    static std::vector<std::string> sources;
    static std::vector<std::string> digests;
    static std::vector<std::vector<ggml::hrx::KernelBindingDefinition>> bindings;
    static std::vector<ggml::hrx::KernelSource> source_records;
    static std::vector<ggml::hrx::KernelSourceRef> primary_sources;
    const size_t dispatch_count = ggml::hrx::schedule_dispatch_count(plan.schedule);
    kernels.clear();
    sources.clear();
    digests.clear();
    bindings.clear();
    source_records.clear();
    primary_sources.clear();
    kernels.reserve(dispatch_count);
    sources.reserve(dispatch_count);
    digests.reserve(dispatch_count);
    bindings.reserve(dispatch_count);
    source_records.reserve(dispatch_count);
    primary_sources.reserve(dispatch_count);

    ggml::hrx::KernelCorpus corpus;
    corpus.upstream_revision = "test-revision";
    corpus.corpus_digest = "test-corpus-sha256";
    corpus.recipe_digest = "test-build-bazel-sha256";
    corpus.plan_case_count = 1;
    for (const ggml::hrx::Invocation & invocation : plan.schedule.invocations) {
        for (const ggml::hrx::Dispatch & dispatch : invocation.dispatches) {
            if (std::find_if(kernels.begin(), kernels.end(), [&](const ggml::hrx::KernelDefinition & item) {
                    return item.family != nullptr && item.name != nullptr &&
                        dispatch.kernel.family == item.family && dispatch.kernel.variant == item.name;
                }) != kernels.end()) continue;
            sources.push_back("test/" + dispatch.kernel.variant + ".loom");
            digests.push_back("sha256-" + dispatch.kernel.variant);
            source_records.emplace_back();
            source_records.back().source.data = sources.back().c_str();
            source_records.back().source.length = sources.back().size();
            source_records.back().source.format = ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT;
            source_records.back().dependencies = nullptr;
            source_records.back().dependency_count = 0;
            primary_sources.emplace_back();
            primary_sources.back().path = sources.back().c_str();
            primary_sources.back().contents = &source_records.back();
            bindings.emplace_back();
            bindings.back().reserve(dispatch.bindings.size());
            for (const ggml::hrx::TensorBinding & binding : dispatch.bindings) {
                bindings.back().push_back({ binding.role.c_str(), ggml::hrx::ResourceAccess::ReadWrite });
            }
            ggml::hrx::KernelDefinition kernel;
            kernel.family = dispatch.kernel.family.c_str();
            kernel.name = dispatch.kernel.variant.c_str();
            kernel.id = ggml::hrx::kernel_catalog_id(kernel.family, kernel.name);
            kernel.source = sources.back().c_str();
            kernel.symbol = dispatch.kernel.variant.c_str();
            kernel.target = plan.target.c_str();
            kernel.source_digest = digests.back().c_str();
            kernel.bindings = { bindings.back().data(), bindings.back().size() };
            kernel.compile_recipe.mode = "direct";
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
    const ggml::hrx::ImportedGraph imported = ggml::hrx::import_graph_with_bindings(fixture.graph);
    const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(imported.graph, "gfx1151");
    REQUIRE(plan.valid());
    const ggml::hrx::KernelCorpus corpus = make_test_corpus(plan);
    REQUIRE(ggml::hrx::verify_kernel_corpus(corpus).valid());
    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(plan, corpus);
    for (const std::string & error : commands.errors) std::fprintf(stderr, "command program: %s\n", error.c_str());
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

    ggml::hrx::KernelCorpus missing_kernel = corpus;
    --missing_kernel.kernels.count;
    REQUIRE(!ggml::hrx::build_command_program(plan, missing_kernel).valid());
    ggml::hrx::CommandProgram bad_range = commands;
    bad_range.commands.front().bindings.front().length = plan.graph.storages[bad_range.commands.front().bindings.front().storage].size + 1;
    REQUIRE(!ggml::hrx::verify_command_program(plan, corpus, bad_range).valid());
    ggml::hrx::CommandProgram bad_dependency = commands;
    bad_dependency.commands.back().dependencies = { bad_dependency.commands.back().ordinal };
    REQUIRE(!ggml::hrx::verify_command_program(plan, corpus, bad_dependency).valid());
    ggml::hrx::CommandProgram bad_abi = commands;
    bad_abi.commands.front().bindings.pop_back();
    REQUIRE(!ggml::hrx::verify_command_program(plan, corpus, bad_abi).valid());

    ggml::hrx::BindingSnapshot snapshot;
    snapshot.device_identity = "gfx1151:0";
    uint64_t identity = 1;
    for (const ggml::hrx::ResourceContract & resource : plan.resources.resources) {
        if (resource.elidable) continue;
        snapshot.bindings.push_back({ resource.storage, identity++, 1, resource.size, 0, resource.size });
    }
    REQUIRE(ggml::hrx::verify_binding_snapshot(plan, snapshot).valid());
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
}

static void test_embedded_kernel_corpus() {
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus("gfx1151");
    REQUIRE(&corpus == &ggml::hrx::get_qwen_kernel_corpus("gfx1151"));
    REQUIRE(ggml::hrx::verify_kernel_corpus(corpus).valid());
    REQUIRE(std::strcmp(corpus.upstream_revision, "b01fe3eb2cddfedad982be873239bc365dccd67f") == 0);
    REQUIRE(std::strcmp(corpus.recipe_digest, "542255e2e245e96ced8744315223e8aeeaeb5e075280930a2fcbc5760cf5551d") == 0);
    REQUIRE(corpus.kernels.size() == 39);
    REQUIRE(corpus.plan_case_count == 24);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && std::string(argv[1]) == "--prove-qwen") {
        std::ifstream graph_file(argv[2]);
        REQUIRE(graph_file.good());
        const std::string graph_text((std::istreambuf_iterator<char>(graph_file)), std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::deserialize_graph_json(graph_text);
        REQUIRE(graph.valid());
        const ggml::hrx::QwenProgramProof proof = ggml::hrx::recover_owned_qwen3_moe_program(graph);
        for (const std::string & error : proof.errors) std::fprintf(stderr, "Qwen proof: %s\n", error.c_str());
        REQUIRE(proof.recognized());
        const ggml::hrx::VerificationResult verification = ggml::hrx::verify_owned_qwen3_moe_program(graph, proof);
        for (const std::string & error : verification.errors) std::fprintf(stderr, "verification: %s\n", error.c_str());
        REQUIRE(verification.valid());
        REQUIRE(proof.structurally_sufficient());
        REQUIRE(proof.natively_complete());
        REQUIRE(proof.root_seams.empty());
        REQUIRE(proof.schedule.roots.size() == 2);
        REQUIRE(std::all_of(proof.schedule.roots.begin(), proof.schedule.roots.end(), [](const ggml::hrx::RootContract & root) {
            return root.disposition == ggml::hrx::RootDisposition::Materialized;
        }));
        // Regression counts are for the SHA-pinned Q4_K_M model lock. Dynamic
        // quant variants have a different per-layer weight mix and are not
        // interchangeable graph fixtures.
        REQUIRE(proof.native_gaps.empty());
        REQUIRE(ggml::hrx::schedule_execution_kind_count(proof.schedule,
            ggml::hrx::KernelSpecialization::ExecutionKind::CpuFallback) == 0);
        if (proof.schedule.workload.rfind("decode-", 0) == 0) {
            size_t split_dispatch_count = 0;
            size_t qkv_dispatch_count = 0;
            size_t q6_value_dispatch_count = 0;
            for (const ggml::hrx::Invocation & invocation : proof.schedule.invocations) {
                for (const ggml::hrx::Dispatch & dispatch : invocation.dispatches) {
                    if (dispatch.kernel.variant == "qwen3_moe_attention_qkv_quantized") {
                        ++qkv_dispatch_count;
                        REQUIRE(dispatch.kernel.integer_parameters.at("query_weight_type") == GGML_TYPE_Q4_K);
                        REQUIRE(dispatch.kernel.integer_parameters.at("key_weight_type") == GGML_TYPE_Q4_K);
                        const int64_t value_type = dispatch.kernel.integer_parameters.at("value_weight_type");
                        REQUIRE((value_type == GGML_TYPE_Q4_K || value_type == GGML_TYPE_Q6_K));
                        q6_value_dispatch_count += value_type == GGML_TYPE_Q6_K;
                    }
                    if (dispatch.kernel.variant != "qwen3_moe_flash_attention_decode_split_f32_f16_wmma") continue;
                    ++split_dispatch_count;
                    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == 768);
                    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_tile_size") == 64);
                    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_block_count") == 12);
                }
            }
            REQUIRE(qkv_dispatch_count == 48);
            REQUIRE(q6_value_dispatch_count == 24);
            REQUIRE(split_dispatch_count == 48);
        }
        const std::string serialized = ggml::hrx::serialize_schedule_json(proof.schedule);
        std::vector<std::string> round_trip_errors;
        const ggml::hrx::Schedule round_trip = ggml::hrx::deserialize_schedule_json(serialized, round_trip_errors);
        REQUIRE(round_trip_errors.empty());
        REQUIRE(ggml::hrx::serialize_schedule_json(round_trip) == serialized);
        REQUIRE(ggml::hrx::verify_schedule(graph, round_trip).valid());

        const ggml::hrx::ProgramPlan reactive = ggml::hrx::build_reactive_plan(graph, "fixture-target");
        for (const std::string & error : reactive.errors) std::fprintf(stderr, "reactive plan: %s\n", error.c_str());
        REQUIRE(reactive.valid());
        REQUIRE(reactive.planner_identity.find("llm.routed_transformer@") != std::string::npos);
        REQUIRE(reactive.legacy_oracle_equivalent);
        REQUIRE(!reactive.fusion_search_text.empty());
        REQUIRE(!reactive.fusion_search_json.empty());
        REQUIRE(reactive.fusion_regions_dot.find("digraph fusion_regions") != std::string::npos);
        REQUIRE(reactive.semantic_witness.find(proof.schedule.workload) != std::string::npos);
        REQUIRE(reactive.graph.values.size() > graph.values.size());
        REQUIRE(ggml::hrx::verify_resource_program(reactive.graph, reactive.schedule, reactive.resources).valid());
        // Kernel recipe parameters must be witnesses of the graph, not model
        // constants hidden in the selector. Check both logical top-k width and
        // its independently recovered physical row stride.
        const auto output_value = [&](size_t operation) -> const ggml::hrx::Value & {
            return graph.values[graph.operations[operation].output];
        };
        const size_t hidden_size = output_value(0).access.shape[0];
        const size_t query_size = output_value(3).access.shape[0];
        const size_t key_value_size = output_value(8).access.shape[0];
        const size_t expert_count = output_value(31).access.shape[0];
        const ggml::hrx::Value & route_ids = output_value(35);
        const size_t route_count = route_ids.access.shape[0];
        const size_t route_stride = route_ids.access.strides[1] / ggml_type_size(route_ids.type);
        size_t checked_routes = 0;
        for (const ggml::hrx::Invocation & invocation : reactive.schedule.invocations) {
            for (const ggml::hrx::Dispatch & dispatch : invocation.dispatches) {
                const auto & config = dispatch.kernel.compile_parameters;
                if (config.count("qwen3_moe.model.hidden_size"))
                    REQUIRE(config.at("qwen3_moe.model.hidden_size") == std::to_string(hidden_size));
                if (config.count("qwen3_moe.attention.query_size"))
                    REQUIRE(config.at("qwen3_moe.attention.query_size") == std::to_string(query_size));
                if (config.count("qwen3_moe.attention.key_value_size"))
                    REQUIRE(config.at("qwen3_moe.attention.key_value_size") == std::to_string(key_value_size));
                if (config.count("qwen3_moe.router.expert_count"))
                    REQUIRE(config.at("qwen3_moe.router.expert_count") == std::to_string(expert_count));
                if (config.count("qwen3_moe.router.route_count"))
                    REQUIRE(config.at("qwen3_moe.router.route_count") == std::to_string(route_count));
                const auto stride = dispatch.kernel.integer_parameters.find("route_id_stride");
                if (stride != dispatch.kernel.integer_parameters.end()) {
                    REQUIRE(stride->second == static_cast<int64_t>(route_stride));
                    ++checked_routes;
                }
            }
        }
        REQUIRE(checked_routes != 0);
        const ggml::hrx::KernelCorpus & executable_corpus = ggml::hrx::get_qwen_kernel_corpus("gfx1151");
        const ggml::hrx::CommandProgram executable_commands =
            ggml::hrx::build_command_program(reactive, executable_corpus);
        const size_t kernel_command_count = std::count_if(
            executable_commands.commands.begin(), executable_commands.commands.end(),
            [](const ggml::hrx::Command & command) {
                return command.kind == ggml::hrx::CommandKind::Kernel;
            });
        const size_t copy_command_count = std::count_if(
            executable_commands.commands.begin(), executable_commands.commands.end(),
            [](const ggml::hrx::Command & command) {
                return command.kind == ggml::hrx::CommandKind::Copy;
            });
        const size_t fill_command_count = std::count_if(
            executable_commands.commands.begin(), executable_commands.commands.end(),
            [](const ggml::hrx::Command & command) {
                return command.kind == ggml::hrx::CommandKind::Fill;
            });
        REQUIRE(kernel_command_count == ggml::hrx::schedule_dispatch_count(reactive.schedule));
        REQUIRE(copy_command_count == 1);
        REQUIRE(fill_command_count == (proof.schedule.workload.rfind("decode", 0) == 0 ? 48 : 0));
        REQUIRE(executable_commands.initializations.size() == 1);
        REQUIRE(executable_commands.initializations[0].data.size() == 256);
        REQUIRE(ggml::hrx::verify_command_program(reactive, executable_corpus, executable_commands).valid());

        ggml::hrx::QwenProgramProof missing_operation = proof;
        missing_operation.schedule.invocations[1].covered_operations.pop_back();
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, missing_operation).valid());
        ggml::hrx::QwenProgramProof wrong_dispatch_count = proof;
        wrong_dispatch_count.schedule.invocations[1].dispatches.pop_back();
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, wrong_dispatch_count).valid());
        ggml::hrx::QwenProgramProof wrong_kernel = proof;
        wrong_kernel.schedule.invocations[1].dispatches[0].kernel.variant = "wrong_kernel";
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, wrong_kernel).valid());
        ggml::hrx::QwenProgramProof wrong_binding = proof;
        wrong_binding.schedule.invocations[1].dispatches[0].bindings[0].value = graph.values.size();
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, wrong_binding).valid());
        ggml::hrx::QwenProgramProof wrong_dependency = proof;
        wrong_dependency.schedule.invocations[1].dispatches[0].dependencies.clear();
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, wrong_dependency).valid());
        ggml::hrx::QwenProgramProof cpu_injection = proof;
        cpu_injection.schedule.invocations[1].dispatches[0].kernel.execution_kind =
            ggml::hrx::KernelSpecialization::ExecutionKind::CpuFallback;
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, cpu_injection).valid());
        ggml::hrx::QwenProgramProof bad_root = proof;
        bad_root.schedule.roots[0].value = graph.values.size();
        REQUIRE(!ggml::hrx::verify_owned_qwen3_moe_program(graph, bad_root).valid());
        ggml::hrx::Graph corrupted_graph = graph;
        corrupted_graph.operations[3].op = GGML_OP_ADD;
        REQUIRE(!ggml::hrx::recover_owned_qwen3_moe_program(corrupted_graph).recognized());
        ggml::hrx::Graph mismatched_kv_graph = graph;
        const ggml::hrx::ValueId first_mask = mismatched_kv_graph.operations[25].inputs[3];
        mismatched_kv_graph.values[first_mask].access.shape[0] -= 64;
        REQUIRE(!ggml::hrx::recover_owned_qwen3_moe_program(mismatched_kv_graph).recognized());
        ggml::hrx::Graph mismatched_route_layout = graph;
        const ggml::hrx::ValueId first_routes = mismatched_route_layout.operations[35].output;
        mismatched_route_layout.values[first_routes].access.strides[1] += ggml_type_size(
            mismatched_route_layout.values[first_routes].type);
        const ggml::hrx::ProgramPlan rejected_routes = ggml::hrx::build_reactive_plan(mismatched_route_layout, "fixture-target");
        REQUIRE(!rejected_routes.valid());
        REQUIRE(std::any_of(rejected_routes.errors.begin(), rejected_routes.errors.end(), [](const std::string & error) {
            return error.find("route layout mismatch") != std::string::npos;
        }));
        std::ofstream schedule_file(argv[3], std::ios::trunc);
        REQUIRE(schedule_file.good());
        schedule_file << serialized << '\n';
        std::ofstream signature_file(argv[4], std::ios::trunc);
        REQUIRE(signature_file.good());
        signature_file << ggml::hrx::qwen_program_signature(proof);
        std::printf("proved %s topology: %zu operations, %zu dispatches, zero CPU fallback, %zu native gaps, %zu root seams\n",
            proof.schedule.workload.c_str(), graph.operations.size(), ggml::hrx::schedule_dispatch_count(proof.schedule),
            proof.native_gaps.size(), proof.root_seams.size());
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--materialize") {
        std::ifstream graph_file(argv[2]);
        REQUIRE(graph_file.good());
        const std::string graph_text((std::istreambuf_iterator<char>(graph_file)), std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::deserialize_graph_json(graph_text);
        REQUIRE(graph.valid());
        const std::vector<ggml::hrx::FusionRule> rules = ggml::hrx::canonical_qwen3_moe_rules();
        const ggml::hrx::Selection selection = ggml::hrx::select_regions(graph, rules);
        const ggml::hrx::Schedule schedule = ggml::hrx::materialize_schedule_with_cpu_fallback(graph, rules, selection);
        REQUIRE(ggml::hrx::verify_schedule(graph, schedule).valid());
        std::ofstream output(argv[3], std::ios::trunc);
        REQUIRE(output.good());
        output << ggml::hrx::serialize_schedule_json(schedule) << '\n';
        std::printf("materialized %zu regions and %zu provisional dispatches with %zu CPU fallbacks for graph %s\n",
            schedule.invocations.size(), ggml::hrx::schedule_dispatch_count(schedule),
            selection.uncovered_operations.size(), graph.fingerprint.c_str());
        return 0;
    }
    if (argc == 3) {
        std::ifstream graph_file(argv[1]);
        std::ifstream schedule_file(argv[2]);
        REQUIRE(graph_file.good());
        REQUIRE(schedule_file.good());
        const std::string graph_text((std::istreambuf_iterator<char>(graph_file)), std::istreambuf_iterator<char>());
        const std::string schedule_text((std::istreambuf_iterator<char>(schedule_file)), std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::deserialize_graph_json(graph_text);
        REQUIRE(graph.valid());
        std::vector<std::string> errors;
        const ggml::hrx::Schedule schedule = ggml::hrx::deserialize_schedule_json(schedule_text, errors);
        REQUIRE(errors.empty());
        REQUIRE(ggml::hrx::verify_schedule(graph, schedule).valid());
        std::printf("verified %zu operations in %zu invocations for graph %s\n",
            graph.operations.size(), schedule.invocations.size(), graph.fingerprint.c_str());
        return 0;
    }
    REQUIRE(argc == 1);
    test_deterministic_import_and_matcher();
    test_set_rows_effects_and_views();
    test_qwen_multi_output_rules();
    test_reactive_cache_and_bindings();
    test_eager_capabilities_and_resource_verification();
    test_command_program_and_diagnostics();
    test_embedded_kernel_corpus();
    return 0;
}
