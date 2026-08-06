#include "fusion-search.h"
#include "reactive-plan.h"
#include "routed-transformer.h"
#include "routed-transformer-bindings.h"
#include "routed-transformer-program.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
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
    const ggml::hrx::SearchResult result = ggml::hrx::SearchResult::search(index, configuration, options);
    REQUIRE(result.valid());
    REQUIRE(result.selected.size() == 1);
    REQUIRE(result.selected.front().family == "fused");
    REQUIRE(result.uncovered_operations.empty());
    REQUIRE(result.report.expanded == 1);
    REQUIRE(result.report.invalidated >= 1);
    REQUIRE(ggml::hrx::SearchResult::format_report(result).find("family=fused") != std::string::npos);
    REQUIRE(ggml::hrx::SearchResult::serialize_report_json(result).find("ggml-hrx-fusion-search-v1") != std::string::npos);
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
        const ggml::hrx::Graph graph = ggml::hrx::Graph::deserialize_json(text);
        REQUIRE(graph.valid());
        const ggml::hrx::GraphIndex index(graph);
        const ggml::hrx::RoutedTransformerModel model = ggml::hrx::RoutedTransformerModel::analyze(index);
        for (const std::string & error : model.errors) std::fprintf(stderr, "analysis: %s\n", error.c_str());
        REQUIRE(model.valid());
        REQUIRE(ggml::hrx::RoutedTransformerModel::verify(index, model).valid());
        REQUIRE(ggml::hrx::RoutedTransformerModel::format(model).find("schema=ggml-hrx-logical-routed-transformer-v1") == 0);
        REQUIRE(ggml::hrx::RoutedTransformerModel::serialize_json(model).find("\"components\":[") != std::string::npos);
        REQUIRE(ggml::hrx::RoutedTransformerModel::dot(model).find("digraph logical_program") != std::string::npos);
        REQUIRE(model.unraised_operations.empty());
        for (size_t block = 0; block < model.blocks.size(); ++block) {
            REQUIRE(model.blocks[block].components.size() == 7);
            for (size_t component = 0; component < model.blocks[block].components.size(); ++component) {
                const auto expected = static_cast<ggml::hrx::LogicalComponentId>(1 + block * 7 + component);
                REQUIRE(model.blocks[block].components[component].id == expected);
            }
        }
        REQUIRE(model.preamble_operations.size() + model.endpoint_operations.size() +
            [&] { size_t count = 0; for (const auto & block : model.blocks) count += block.operations.size(); return count; }() ==
            graph.operations.size());
        const ggml::hrx::SearchResult result = ggml::hrx::SearchResult::search(
            index, ggml::hrx::RoutedTransformerProvider::make_planner(), { true, true });
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
        const ggml::hrx::SearchResult future = ggml::hrx::SearchResult::search(
            index, ggml::hrx::RoutedTransformerProvider::make_planner(future_catalog), { true, true });
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
        ggml::hrx::RoutedTransformerProgramProof structural =
            ggml::hrx::RoutedTransformerProgramProof::recover(graph);
        for (const std::string & error : structural.errors) std::fprintf(stderr, "program: %s\n", error.c_str());
        REQUIRE(structural.valid());
        ggml::hrx::Graph structural_bound_graph = graph;
        REQUIRE(ggml::hrx::RoutedTransformerProgramProof::materialize_dispatch_bindings(
            structural_bound_graph, structural.schedule, *structural.logical_program).valid());
        if (model.query_token_count != 1) {
            const auto gather = std::find_if(structural.schedule.invocations.begin(), structural.schedule.invocations.end(),
                [](const ggml::hrx::Invocation & invocation) {
                    return std::any_of(invocation.dispatches.begin(), invocation.dispatches.end(),
                        [](const ggml::hrx::Dispatch & dispatch) {
                            return dispatch.kernel.variant == "ggml_gather_add_f32";
                        });
                });
            REQUIRE(gather != structural.schedule.invocations.end());
            const auto & dispatch = *std::find_if(gather->dispatches.begin(), gather->dispatches.end(),
                [](const ggml::hrx::Dispatch & item) { return item.kernel.variant == "ggml_gather_add_f32"; });
            REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == model.output_token_count);
            REQUIRE(dispatch.kernel.integer_parameters.at("source_token_count") == model.query_token_count);
            REQUIRE(dispatch.kernel.integer_parameters.at("output_token_count") == model.output_token_count);
        }
        std::set<ggml::hrx::LogicalComponentId> selected_components;
        for (const auto & selected : structural.search.selected) {
            REQUIRE(!selected.logical_components.empty());
            selected_components.insert(selected.logical_components.begin(), selected.logical_components.end());
        }
        std::set<ggml::hrx::LogicalComponentId> emitted_components;
        for (const auto & invocation : structural.schedule.invocations) {
            REQUIRE(!invocation.recipe.empty());
            REQUIRE(!invocation.logical_components.empty());
            emitted_components.insert(invocation.logical_components.begin(), invocation.logical_components.end());
        }
        REQUIRE(emitted_components == selected_components);
        const ggml::hrx::VerificationResult verification =
            ggml::hrx::verify_schedule(structural_bound_graph, structural.schedule);
        for (const std::string & error : verification.errors) std::fprintf(stderr, "verification: %s\n", error.c_str());
        REQUIRE(verification.valid());

        // An unfamiliar tail is represented explicitly in the logical IR and
        // lowered through the atom emitter. Existing component recipes remain
        // intact; partial workload knowledge does not silently drop the op or
        // force a second whole-model matcher.
        ggml::hrx::Graph extended = graph;
        const ggml::hrx::ValueId tail_input = extended.roots.front();
        ggml::hrx::Value atom_output = extended.values[tail_input];
        atom_output.id = static_cast<ggml::hrx::ValueId>(extended.values.size());
        atom_output.name = "test.unfamiliar_tail";
        atom_output.op = GGML_OP_CLAMP;
        atom_output.producer = static_cast<ggml::hrx::OperationId>(extended.operations.size());
        atom_output.boundary = ggml::hrx::BoundaryKind::Internal;
        atom_output.access.storage = static_cast<ggml::hrx::StorageId>(extended.storages.size());
        extended.storages.push_back({ atom_output.access.storage, atom_output.id,
                                      extended.storages[extended.values[tail_input].access.storage].size,
                                      false, false, false, 0 });
        extended.values.push_back(atom_output);
        ggml::hrx::Operation atom;
        atom.id = static_cast<ggml::hrx::OperationId>(extended.operations.size());
        atom.original_ordinal = atom.id;
        atom.op = GGML_OP_CLAMP;
        atom.inputs = { tail_input };
        atom.output = atom_output.id;
        extended.operations.push_back(std::move(atom));
        const ggml::hrx::RoutedTransformerProgramProof extended_program =
            ggml::hrx::RoutedTransformerProgramProof::recover(extended);
        REQUIRE(extended_program.valid());
        REQUIRE(extended_program.logical_program->fallback_components.size() == 1);
        REQUIRE(std::count_if(extended_program.schedule.invocations.begin(), extended_program.schedule.invocations.end(),
            [](const ggml::hrx::Invocation & invocation) {
                return invocation.recipe == "atom.CLAMP" &&
                       invocation.kernel.execution_kind == ggml::hrx::KernelSpecialization::ExecutionKind::NativeEager;
            }) == 1);
        const ggml::hrx::ProgramPlan extended_plan = ggml::hrx::build_reactive_plan(extended, "fixture-target");
        REQUIRE(extended_plan.valid());
        REQUIRE(extended_plan.atom_fallback_count == 1);
        REQUIRE(extended_plan.warnings.size() == 1);
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
