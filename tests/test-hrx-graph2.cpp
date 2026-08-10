#include "domains/llm-patterns.h"
#include "domains/llm-recipes.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph-plan.h"
#include "kernel-corpus.h"
#include "matcher.h"
#include "planner.h"
#include "program-selection.h"
#include "transitional-command-program.h"
#include "transitional-program.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

struct graph_fixture {
    ggml_context * context = nullptr;
    ggml_cgraph *  graph   = nullptr;

    graph_fixture() {
        ggml_init_params parameters = { 1024 * 1024, nullptr, true };
        context                     = ggml_init(parameters);
        REQUIRE(context != nullptr);
        graph = ggml_new_graph_custom(context, 16, false);
    }

    ~graph_fixture() { ggml_free(context); }
};

static void build_arithmetic_graph(graph_fixture & fixture) {
    ggml_tensor * lhs   = ggml_new_tensor_1d(fixture.context, GGML_TYPE_F32, 8);
    ggml_tensor * rhs   = ggml_new_tensor_1d(fixture.context, GGML_TYPE_F32, 8);
    ggml_tensor * scale = ggml_new_tensor_1d(fixture.context, GGML_TYPE_F32, 8);
    ggml_set_input(lhs);
    ggml_set_input(rhs);
    ggml_set_input(scale);
    ggml_tensor * sum    = ggml_add(fixture.context, lhs, rhs);
    ggml_tensor * result = ggml_mul(fixture.context, sum, scale);
    ggml_set_output(result);
    ggml_build_forward_expand(fixture.graph, result);
}

static void test_fixture(const char * path) {
    std::ifstream input(path);
    REQUIRE(input.good());
    const std::string     text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ggml::hrx::graph_plan plan = ggml::hrx::graph_plan::deserialize_json(text);
    REQUIRE(plan.valid());
    ggml::hrx::pattern_registry patterns;
    ggml::hrx::llm_patterns::register_patterns(patterns);
    if (!ggml::hrx::matcher::recognize(plan, patterns)) {
        std::fprintf(stderr, "%s", plan.format().c_str());
        std::abort();
    }

    std::map<ggml::hrx::region_kind, size_t> counts;
    for (ggml::hrx::region_id id : plan.active_regions()) {
        ++counts[plan.regions()[id].kind];
    }
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_EMBEDDING] == 1);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_ATTENTION_PREPARE] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_ATTENTION_QKV] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_ATTENTION] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_ATTENTION_OUTPUT] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_ROUTER_SELECTION] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_EXPERT_GATE_UP] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_EXPERT_DOWN] == 48);
    REQUIRE(counts[ggml::hrx::region_kind::REGION_KIND_ENDPOINT] == 1);
    REQUIRE(plan.active_regions().size() == 338);
    REQUIRE(plan.groups().size() == 48);
    for (const ggml::hrx::semantic_group & group : plan.groups()) {
        REQUIRE(group.members.size() == 7);
    }
    REQUIRE(plan.diagnostics().matcher_probes <= plan.operations().size() * 8);
    REQUIRE(ggml::hrx::planner::select_recipes(plan, "gfx1151"));
    const ggml::hrx::program_selection selection = ggml::hrx::program_selection::select(plan, "gfx1151");
    REQUIRE(selection.valid());
    REQUIRE(selection.implementation == ggml::hrx::program_implementation::PROGRAM_IMPLEMENTATION_TRANSITIONAL);
    REQUIRE(selection.coverage.operations.size() == plan.operations().size());
    REQUIRE(plan.verify(true));
    const ggml::hrx::ProgramPlan transitional = ggml::hrx::build_transitional_program(plan, selection, "gfx1151", 17);
    REQUIRE(transitional.valid());
    std::vector<const ggml_tensor *> source_roots(plan.storages().size());
    for (size_t i = 0; i < source_roots.size(); ++i) {
        source_roots[i] = reinterpret_cast<const ggml_tensor *>(i + 1);
    }
    std::vector<const ggml_tensor *> program_roots;
    std::vector<std::string>         binding_errors;
    REQUIRE(ggml::hrx::transitional_program_adapter::bind_storage_roots(plan, transitional, source_roots, program_roots,
                                                                        binding_errors));
    REQUIRE(program_roots.size() == transitional.graph.storages.size());
    REQUIRE(std::equal(source_roots.begin(), source_roots.end(), program_roots.begin()));
    REQUIRE(std::all_of(program_roots.begin() + source_roots.size(), program_roots.end(),
                        [](const ggml_tensor * tensor) { return tensor == nullptr; }));
    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(transitional, ggml::hrx::get_qwen_kernel_corpus());
    REQUIRE(commands.valid());
    const ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(transitional, ggml::hrx::get_qwen_kernel_corpus(), commands);
    REQUIRE(verification.valid());
    const int64_t query_tokens        = plan.facts().at("query_token_count");
    const size_t  expected_dispatches = query_tokens == 1 ? 293 : (query_tokens == 512 ? 632 : 679);
    REQUIRE(ggml::hrx::schedule_dispatch_count(transitional.schedule) == expected_dispatches);
    REQUIRE(commands.commands.size() == (query_tokens == 1 ? 293 : 679));
}

static void test_degenerate_pipeline() {
    std::ostringstream fixture;
    fixture << "{\"version\":1,\"storages\":["
            << "{\"id\":0,\"root\":0,\"size\":4,\"external\":true,\"weight\":false,\"mutable\":false,\"version\":0},"
            << "{\"id\":1,\"root\":1,\"size\":4,\"external\":true,\"weight\":false,\"mutable\":false,\"version\":0},"
            << "{\"id\":2,\"root\":2,\"size\":4,\"external\":false,\"weight\":false,\"mutable\":false,\"version\":0}],"
            << "\"values\":["
            << "{\"id\":0,\"type\":\"f32\",\"op\":\"NONE\",\"storage\":0,\"version\":0,\"offset\":0,\"flags\":1,"
               "\"producer\":4294967295,\"view_source\":4294967295,\"boundary\":\"input\",\"shape\":[1,1,1,1],"
               "\"strides\":[4,4,4,4],\"name\":\"lhs\"},"
            << "{\"id\":1,\"type\":\"f32\",\"op\":\"NONE\",\"storage\":1,\"version\":0,\"offset\":0,\"flags\":1,"
               "\"producer\":4294967295,\"view_source\":4294967295,\"boundary\":\"input\",\"shape\":[1,1,1,1],"
               "\"strides\":[4,4,4,4],\"name\":\"rhs\"},"
            << "{\"id\":2,\"type\":\"f32\",\"op\":\"ADD\",\"storage\":2,\"version\":0,\"offset\":0,\"flags\":2,"
               "\"producer\":0,\"view_source\":4294967295,\"boundary\":\"output\",\"shape\":[1,1,1,1],\"strides\":[4,4,"
               "4,4],\"name\":\"sum\"}],"
            << "\"operations\":[{\"id\":0,\"op\":\"ADD\",\"output\":2,\"ordinal\":0,\"params\":\""
            << std::string(GGML_MAX_OP_PARAMS * 2, '0')
            << "\",\"inputs\":[0,1],\"effects\":[]}],\"roots\":[2],\"errors\":[]}";

    ggml::hrx::graph_plan plan = ggml::hrx::graph_plan::deserialize_json(fixture.str());
    REQUIRE(plan.valid());
    ggml::hrx::pattern_registry patterns;
    ggml::hrx::llm_patterns::register_patterns(patterns);
    REQUIRE(ggml::hrx::matcher::recognize(plan, patterns));
    REQUIRE(plan.active_regions().size() == 1);
    REQUIRE(plan.regions()[plan.active_regions().front()].kind == ggml::hrx::region_kind::REGION_KIND_ATOM);
    REQUIRE(ggml::hrx::planner::select_recipes(plan, "gfx1151"));
    const ggml::hrx::program_selection selection = ggml::hrx::program_selection::select(plan, "gfx1151");
    REQUIRE(selection.valid());
    REQUIRE(selection.roots.size() == 1);
    REQUIRE(selection.roots.front() == "primitive");
    REQUIRE(selection.coverage.operations.size() == 1);
    REQUIRE(plan.verify(true));
}

static void test_uid_cache() {
    ggml::hrx::graph_plan_cache cache;
    REQUIRE(!cache.prepare(nullptr, "gfx1151").valid());

    graph_fixture first;
    build_arithmetic_graph(first);
    first.graph->uid                            = 41;
    const ggml::hrx::graph_execution_frame cold = cache.prepare(first.graph, "gfx1151");
    REQUIRE(cold.valid());
    REQUIRE(cache.stats().builds == 1);
    const ggml::hrx::graph_execution_frame hot = cache.prepare(first.graph, "gfx1151");
    REQUIRE(hot.valid());
    REQUIRE(hot.plan == cold.plan);
    REQUIRE(hot.values.front() == cold.values.front());
    REQUIRE(cache.stats().hits == 1);

    // A scheduler may rebuild the cgraph object while preserving its UID. The
    // frozen plan is reused, but stable leaf/node slots rebind live tensors.
    graph_fixture rebound;
    build_arithmetic_graph(rebound);
    rebound.graph->uid                                   = 41;
    const ggml::hrx::graph_execution_frame rebound_frame = cache.prepare(rebound.graph, "gfx1151");
    for (const std::string & error : rebound_frame.errors) {
        std::fprintf(stderr, "cache error: %s\n", error.c_str());
    }
    REQUIRE(rebound_frame.valid());
    REQUIRE(rebound_frame.plan == cold.plan);
    REQUIRE(rebound_frame.values.front() != cold.values.front());
    REQUIRE(cache.stats().hits == 2);

    REQUIRE(!cache.prepare(first.graph, "gfx1100").valid());
    graph_fixture uncached;
    build_arithmetic_graph(uncached);
    REQUIRE(uncached.graph->uid == 0);
    const ggml::hrx::graph_execution_frame zero_first  = cache.prepare(uncached.graph, "gfx1151");
    const ggml::hrx::graph_execution_frame zero_second = cache.prepare(uncached.graph, "gfx1151");
    REQUIRE(zero_first.valid());
    REQUIRE(zero_second.valid());
    REQUIRE(zero_first.plan != zero_second.plan);
}

static void test_mutation_import() {
    graph_fixture fixture;
    ggml_tensor * destination = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 8);
    ggml_tensor * rows        = ggml_new_tensor_2d(fixture.context, GGML_TYPE_F32, 4, 2);
    ggml_tensor * indices     = ggml_new_tensor_1d(fixture.context, GGML_TYPE_I32, 2);
    ggml_set_input(destination);
    ggml_set_input(rows);
    ggml_set_input(indices);
    ggml_tensor * result = ggml_set_rows(fixture.context, destination, rows, indices);
    ggml_set_output(result);
    ggml_build_forward_expand(fixture.graph, result);

    const ggml::hrx::graph_plan plan = ggml::hrx::graph_plan::import(fixture.graph);
    REQUIRE(plan.valid());
    REQUIRE(plan.operations().size() == 1);
    const ggml::hrx::graph_op & operation = plan.operations().front();
    REQUIRE(operation.storage_semantics == ggml::hrx::operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_MUTATE);
    REQUIRE(operation.inputs.size() == 3);
    const ggml::hrx::graph_value & output = plan.values()[operation.output];
    REQUIRE(output.view_source != ggml::hrx::ID_INVALID);
    REQUIRE(output.access.storage == plan.values()[operation.inputs[2]].access.storage);
    REQUIRE(output.access.version == 1);
    const auto write =
        std::find_if(operation.effects.begin(), operation.effects.end(), [](const ggml::hrx::graph_effect & effect) {
            return effect.kind == ggml::hrx::effect_kind::EFFECT_KIND_WRITE;
        });
    REQUIRE(write != operation.effects.end());
    REQUIRE(write->size == plan.storages()[write->storage].size);
}

static void test_program_catalog_contract() {
    float    rms_epsilon      = 0.000001f;
    uint32_t rms_epsilon_bits = 0;
    std::memcpy(&rms_epsilon_bits, &rms_epsilon, sizeof(rms_epsilon_bits));
    std::map<std::string, int64_t> facts = {
        { "query_token_count",        1                },
        { "key_value_token_count",    576              },
        { "output_token_count",       1                },
        { "layer_count",              48               },
        { "hidden_size",              2048             },
        { "head_size",                128              },
        { "query_size",               4096             },
        { "key_value_size",           512              },
        { "query_head_count",         32               },
        { "key_value_head_count",     4                },
        { "expert_count",             128              },
        { "route_count",              8                },
        { "expert_intermediate_size", 768              },
        { "vocabulary_count",         151936           },
        { "rms_epsilon_bits",         rms_epsilon_bits },
    };
    REQUIRE(ggml::hrx::llm_recipes::supports_qwen3_30b_decode_576(facts));
    facts["key_value_token_count"] = 575;
    REQUIRE(!ggml::hrx::llm_recipes::supports_qwen3_30b_decode_576(facts));
    facts["key_value_token_count"] = 576;
    facts.erase("route_count");
    REQUIRE(!ggml::hrx::llm_recipes::supports_qwen3_30b_decode_576(facts));
}

int main(int argc, char ** argv) {
    test_degenerate_pipeline();
    test_uid_cache();
    test_mutation_import();
    test_program_catalog_contract();
    for (int i = 1; i < argc; ++i) {
        test_fixture(argv[i]);
    }
    return 0;
}
