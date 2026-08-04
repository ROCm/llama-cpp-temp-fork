#include "fusion-search.h"
#include "routed-transformer.h"
#include "routed-transformer-bindings.h"
#include "routed-transformer-program.h"
#include "qwen-bindings.h"
#include "qwen-program.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

class GraphBuilder {
public:
    ggml::hrx::ValueId input() {
        ggml::hrx::Value value;
        value.id = graph.values.size();
        value.type = GGML_TYPE_F32;
        value.op = GGML_OP_NONE;
        value.access.storage = graph.storages.size();
        value.access.shape = { 4, 1, 1, 1 };
        value.access.strides = { 4, 16, 16, 16 };
        value.boundary = ggml::hrx::BoundaryKind::Input;
        graph.storages.push_back({ static_cast<ggml::hrx::StorageId>(graph.storages.size()), value.id,
                                   16, true, false, false, 0 });
        graph.values.push_back(value);
        return value.id;
    }

    ggml::hrx::OperationId op(enum ggml_op kind, std::vector<ggml::hrx::ValueId> inputs) {
        ggml::hrx::Value output;
        output.id = graph.values.size();
        output.type = GGML_TYPE_F32;
        output.op = kind;
        output.access.storage = graph.storages.size();
        output.access.shape = { 4, 1, 1, 1 };
        output.access.strides = { 4, 16, 16, 16 };
        output.producer = graph.operations.size();
        graph.storages.push_back({ static_cast<ggml::hrx::StorageId>(graph.storages.size()), output.id,
                                   16, false, false, false, 0 });
        graph.values.push_back(output);
        ggml::hrx::Operation operation;
        operation.id = graph.operations.size();
        operation.op = kind;
        operation.inputs = std::move(inputs);
        operation.output = output.id;
        graph.operations.push_back(operation);
        return operation.id;
    }

    ggml::hrx::ValueId output(ggml::hrx::OperationId operation) {
        graph.values[graph.operations[operation].output].boundary = ggml::hrx::BoundaryKind::Output;
        graph.roots.push_back(graph.operations[operation].output);
        return graph.operations[operation].output;
    }

    ggml::hrx::Graph graph;
};

static ggml::hrx::FusionCandidate candidate(const ggml::hrx::GraphIndex & index,
                                             std::string family,
                                             std::vector<ggml::hrx::OperationId> operations,
                                             int reference_dispatches,
                                             int planned_dispatches) {
    ggml::hrx::FusionCandidate result;
    result.provider = "test";
    result.family = std::move(family);
    result.key = result.provider + ":" + result.family;
    result.hero = operations.back();
    result.operations = std::move(operations);
    result.materialized_outputs = index.boundary(result.operations).outputs;
    result.economics.reference_dispatches = reference_dispatches;
    result.economics.planned_dispatches = planned_dispatches;
    return result;
}

class TestProvider final : public ggml::hrx::FusionProvider {
public:
    explicit TestProvider(std::vector<ggml::hrx::FusionCandidate> seeds,
                          std::vector<ggml::hrx::FusionCandidate> expansions = {})
        : seeds_(std::move(seeds)), expansions_(std::move(expansions)) {}

    const char * id() const override { return "test"; }
    const char * revision() const override { return "1"; }

    ggml::hrx::Decision discover(const ggml::hrx::GraphIndex &, ggml::hrx::FactDatabase & facts) const override {
        return facts.observe("test.width", int64_t(4), { "shape", 0 });
    }

    void seed(const ggml::hrx::GraphIndex &, const ggml::hrx::FactDatabase &,
              std::vector<ggml::hrx::FusionCandidate> & candidates) const override {
        candidates = seeds_;
    }

    void expand(const ggml::hrx::GraphIndex &, const ggml::hrx::FactDatabase &,
                const ggml::hrx::FusionCandidate & seed,
                std::vector<ggml::hrx::FusionCandidate> & expansions) const override {
        if (seed.family == "seed") expansions = expansions_;
    }

private:
    std::vector<ggml::hrx::FusionCandidate> seeds_;
    std::vector<ggml::hrx::FusionCandidate> expansions_;
};

static void test_index_boundary_and_legality() {
    GraphBuilder builder;
    const auto x = builder.input();
    const auto y = builder.input();
    const auto a = builder.op(GGML_OP_ADD, { x, y });
    const auto b = builder.op(GGML_OP_MUL, { builder.graph.operations[a].output, y });
    const auto c = builder.op(GGML_OP_ADD, { builder.graph.operations[b].output, y });
    builder.output(c);
    const ggml::hrx::GraphIndex index(builder.graph);
    REQUIRE(index.valid());
    REQUIRE(index.consumers(builder.graph.operations[a].output) == std::vector<ggml::hrx::OperationId> { b });
    REQUIRE(index.predecessors(c) == std::vector<ggml::hrx::OperationId> { b });

    const ggml::hrx::RegionBoundary full = index.boundary({ a, b, c });
    REQUIRE(full.inputs.size() == 2);
    REQUIRE(full.outputs == std::vector<ggml::hrx::ValueId> { builder.graph.operations[c].output });
    REQUIRE(index.validate_region({ a, b, c }, full.outputs).allowed);
    REQUIRE(index.validate_region({ a, c }, { builder.graph.operations[c].output }).reason ==
            ggml::hrx::DecisionReason::DisconnectedRegion);
    REQUIRE(index.validate_region({ a, b, c }, {}).reason == ggml::hrx::DecisionReason::MissingMaterialization);
    REQUIRE(index.validate_region({ a, b, c }, { static_cast<ggml::hrx::ValueId>(999) }).reason ==
            ggml::hrx::DecisionReason::InvalidValue);
}

static void test_storage_versions_are_dependencies() {
    GraphBuilder builder;
    const auto state = builder.input();
    const auto x = builder.input();
    const auto writer = builder.op(GGML_OP_ADD, { x, x });
    const auto reader = builder.op(GGML_OP_MUL, { x, x });
    const auto storage = builder.graph.values[state].access.storage;
    builder.graph.storages[storage].mutable_state = true;
    builder.graph.storages[storage].final_version = 1;
    builder.graph.operations[writer].effects.push_back({
        ggml::hrx::EffectKind::Write, storage, 0, 1, 0, 16, true });
    builder.graph.operations[reader].effects.push_back({
        ggml::hrx::EffectKind::Read, storage, 1, 1, 0, 16, true });
    builder.output(reader);
    const ggml::hrx::GraphIndex index(builder.graph);
    REQUIRE(index.valid());
    REQUIRE(index.storage_writer(storage, 1) == writer);
    REQUIRE(std::find(index.predecessors(reader).begin(), index.predecessors(reader).end(), writer) !=
            index.predecessors(reader).end());
    REQUIRE(index.validate_region({ writer, reader }, index.boundary({ writer, reader }).outputs).allowed);
}

static void test_contracted_cycle() {
    GraphBuilder builder;
    const auto x = builder.input();
    const auto a = builder.op(GGML_OP_ADD, { x, x });
    const auto b = builder.op(GGML_OP_MUL, { builder.graph.operations[a].output, x });
    const auto c = builder.op(GGML_OP_ADD, { builder.graph.operations[b].output, x });
    builder.output(c);
    const ggml::hrx::GraphIndex index(builder.graph);
    const auto decision = index.validate_region({ a, c }, index.boundary({ a, c }).outputs);
    // The set is both disconnected under an induced-subgraph definition and
    // non-convex. Connectivity is intentionally reported first.
    REQUIRE(decision.reason == ggml::hrx::DecisionReason::DisconnectedRegion);

    std::vector<size_t> order;
    const auto regions = std::vector<std::vector<ggml::hrx::OperationId>> { { a, c }, { b } };
    REQUIRE(index.topologically_order_regions(regions, order).reason == ggml::hrx::DecisionReason::ContractedCycle);
}

static void test_priority_growth_and_overlap() {
    GraphBuilder builder;
    const auto x = builder.input();
    const auto a = builder.op(GGML_OP_ADD, { x, x });
    const auto b = builder.op(GGML_OP_MUL, { builder.graph.operations[a].output, x });
    builder.output(b);
    const ggml::hrx::GraphIndex index(builder.graph);

    auto seed = candidate(index, "seed", { a }, 2, 1);
    auto tail = candidate(index, "tail", { b }, 2, 1);
    auto fused = candidate(index, "fused", { a, b }, 4, 1);
    ggml::hrx::PlannerConfiguration configuration;
    configuration.add_provider(std::make_shared<TestProvider>(
        std::vector<ggml::hrx::FusionCandidate> { seed, tail },
        std::vector<ggml::hrx::FusionCandidate> { fused }));
    ggml::hrx::SearchOptions options;
    options.require_complete_coverage = true;
    options.record_trace = true;
    const ggml::hrx::SearchResult result = ggml::hrx::search_fusions(index, configuration, options);
    REQUIRE(result.valid());
    REQUIRE(result.selected.size() == 1);
    REQUIRE(result.selected.front().family == "fused");
    REQUIRE(result.uncovered_operations.empty());
    REQUIRE(result.report.expanded == 1);
    REQUIRE(result.report.invalidated >= 1);
    REQUIRE(ggml::hrx::format_search_report(result).find("family=fused") != std::string::npos);
    REQUIRE(ggml::hrx::serialize_search_report_json(result).find("ggml-hrx-fusion-search-v1") != std::string::npos);
}

static void test_fact_disagreement() {
    ggml::hrx::FactDatabase facts;
    REQUIRE(facts.observe("llm.hidden_size", int64_t(2048), { "layer", 1 }).allowed);
    const auto conflict = facts.observe("llm.hidden_size", int64_t(4096), { "layer", 2 });
    REQUIRE(!conflict.allowed);
    REQUIRE(conflict.reason == ggml::hrx::DecisionReason::InconsistentFact);
    REQUIRE(conflict.implicated_ids == std::vector<uint32_t>({ 1, 2 }));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2) {
        std::ifstream input(argv[1]);
        REQUIRE(input.good());
        const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        const ggml::hrx::Graph graph = ggml::hrx::deserialize_graph_json(text);
        REQUIRE(graph.valid());
        const ggml::hrx::GraphIndex index(graph);
        const ggml::hrx::RoutedTransformerModel model = ggml::hrx::analyze_routed_transformer(index);
        for (const std::string & error : model.errors) std::fprintf(stderr, "analysis: %s\n", error.c_str());
        REQUIRE(model.valid());
        REQUIRE(model.preamble_operations.size() + model.endpoint_operations.size() +
            [&] { size_t count = 0; for (const auto & block : model.blocks) count += block.operations.size(); return count; }() ==
            graph.operations.size());
        const ggml::hrx::SearchResult result = ggml::hrx::search_fusions(
            index, ggml::hrx::make_structural_routed_transformer_planner(), { true, true });
        for (const std::string & error : result.errors) std::fprintf(stderr, "search: %s\n", error.c_str());
        REQUIRE(result.valid());
        REQUIRE(result.uncovered_operations.empty());
        auto planned_dispatches = [](const ggml::hrx::SearchResult & search) {
            size_t count = 0;
            for (const auto & selected : search.selected) {
                REQUIRE(selected.economics.planned_dispatches >= 0);
                count += static_cast<size_t>(selected.economics.planned_dispatches);
            }
            return count;
        };
        ggml::hrx::RoutedTransformerRecipeCatalog future_catalog;
        using namespace ggml::hrx::routed_transformer_recipes;
        future_catalog.available = {
            kDecodeQkvPostprocess, kDecodeOutputNextQ8, kDecodeRouterTopK,
            kDecodeGateUpNextQ8, kDecodeDownNextQ8,
            kPrefillExpertPartition, kPrefillDownNextNorm,
        };
        const ggml::hrx::SearchResult future = ggml::hrx::search_fusions(
            index, ggml::hrx::make_structural_routed_transformer_planner(future_catalog), { true, true });
        for (const std::string & error : future.errors) std::fprintf(stderr, "future search: %s\n", error.c_str());
        REQUIRE(future.valid());
        REQUIRE(future.uncovered_operations.empty());
        const size_t current_dispatches = planned_dispatches(result);
        const size_t future_dispatches = planned_dispatches(future);
        // This verifies composition: enabling independently described recipes
        // removes one publication per applicable component. It intentionally
        // derives the expected delta from recovered blocks instead of pinning a
        // model/corpus dispatch total.
        const size_t expected_reduction = model.query_token_count == 1
            ? model.blocks.size() * 5
            : (model.blocks.size() - 1) * 2;
        REQUIRE(current_dispatches == future_dispatches + expected_reduction);
        const ggml::hrx::QwenProgramProof legacy = ggml::hrx::recover_owned_qwen3_moe_program(graph);
        ggml::hrx::RoutedTransformerProgramProof structural =
            ggml::hrx::recover_structural_routed_transformer_program(graph);
        if (model.output_token_count != 1) {
            REQUIRE(!legacy.recognized());
            REQUIRE(!structural.valid());
            REQUIRE(std::any_of(structural.errors.begin(), structural.errors.end(), [](const std::string & error) {
                return error.find("demanded output token") != std::string::npos;
            }));
            std::printf("routed-transformer blocks=%zu components=%zu Tq=%lld Tout=%lld Tkv=%lld (no current recipe)\n",
                        model.blocks.size(), result.selected.size(),
                        static_cast<long long>(model.query_token_count),
                        static_cast<long long>(model.output_token_count),
                        static_cast<long long>(model.key_value_token_count));
            return 0;
        }
        REQUIRE(legacy.recognized());
        for (const std::string & error : structural.errors) std::fprintf(stderr, "program: %s\n", error.c_str());
        REQUIRE(structural.valid());
        REQUIRE(structural.schedule.workload == legacy.schedule.workload);
        REQUIRE(ggml::hrx::schedule_dispatch_count(structural.schedule) ==
                ggml::hrx::schedule_dispatch_count(legacy.schedule));
        auto flatten = [](const ggml::hrx::Schedule & schedule) {
            std::vector<const ggml::hrx::Dispatch *> dispatches;
            for (const auto & invocation : schedule.invocations) {
                for (const auto & dispatch : invocation.dispatches) dispatches.push_back(&dispatch);
            }
            return dispatches;
        };
        const auto legacy_dispatches = flatten(legacy.schedule);
        const auto structural_dispatches = flatten(structural.schedule);
        REQUIRE(legacy_dispatches.size() == structural_dispatches.size());
        for (size_t i = 0; i < legacy_dispatches.size(); ++i) {
            REQUIRE(legacy_dispatches[i]->kernel.family == structural_dispatches[i]->kernel.family);
            REQUIRE(legacy_dispatches[i]->kernel.variant == structural_dispatches[i]->kernel.variant);
            REQUIRE(legacy_dispatches[i]->kernel.integer_parameters == structural_dispatches[i]->kernel.integer_parameters);
            REQUIRE(legacy_dispatches[i]->dependencies == structural_dispatches[i]->dependencies);
        }
        ggml::hrx::Graph legacy_bound_graph = graph;
        ggml::hrx::Schedule legacy_bound_schedule = legacy.schedule;
        REQUIRE(ggml::hrx::materialize_qwen3_moe_dispatch_bindings(
            legacy_bound_graph, legacy_bound_schedule).valid());
        ggml::hrx::Graph structural_bound_graph = graph;
        REQUIRE(ggml::hrx::materialize_routed_transformer_dispatch_bindings(
            structural_bound_graph, structural.schedule).valid());
        REQUIRE(legacy_bound_graph.values.size() == structural_bound_graph.values.size());
        REQUIRE(legacy_bound_graph.storages.size() == structural_bound_graph.storages.size());
        const auto legacy_bound_dispatches = flatten(legacy_bound_schedule);
        const auto structural_bound_dispatches = flatten(structural.schedule);
        REQUIRE(legacy_bound_dispatches.size() == structural_bound_dispatches.size());
        for (size_t i = 0; i < legacy_bound_dispatches.size(); ++i) {
            const auto & expected = *legacy_bound_dispatches[i];
            const auto & actual = *structural_bound_dispatches[i];
            REQUIRE(expected.kernel.family == actual.kernel.family);
            REQUIRE(expected.kernel.variant == actual.kernel.variant);
            REQUIRE(expected.kernel.integer_parameters == actual.kernel.integer_parameters);
            REQUIRE(expected.kernel.compile_parameters == actual.kernel.compile_parameters);
            REQUIRE(expected.kernel.execution_kind == actual.kernel.execution_kind);
            REQUIRE(expected.bindings == actual.bindings);
            REQUIRE(expected.dependencies == actual.dependencies);
        }
        const ggml::hrx::VerificationResult verification =
            ggml::hrx::verify_schedule(structural_bound_graph, structural.schedule);
        for (const std::string & error : verification.errors) std::fprintf(stderr, "verification: %s\n", error.c_str());
        REQUIRE(verification.valid());
        std::printf("routed-transformer blocks=%zu components=%zu Tq=%lld Tout=%lld Tkv=%lld\n",
                    model.blocks.size(), result.selected.size(),
                    static_cast<long long>(model.query_token_count),
                    static_cast<long long>(model.output_token_count),
                    static_cast<long long>(model.key_value_token_count));
        return 0;
    }
    REQUIRE(argc == 1);
    test_index_boundary_and_legality();
    test_storage_versions_are_dependencies();
    test_contracted_cycle();
    test_priority_growth_and_overlap();
    test_fact_disagreement();
    return 0;
}
