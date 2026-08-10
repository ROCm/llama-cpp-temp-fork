#include "domains/llm-patterns.h"
#include "graph-plan.h"
#include "kernel-corpus-json.h"
#include "kernel-corpus.h"
#include "planner.h"
#include "program-selection.h"
#include "tool-utils.h"
#include "transitional-command-program.h"
#include "transitional-program.h"
#include "transitional-schedule.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using ggml::hrx::tool::write_file;

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: ggml-hrx-dump-program normalized-graph.json target output-dir\n";
        return 2;
    }

    try {
        const std::filesystem::path graph_path       = argv[1];
        const std::string           target           = argv[2];
        const std::filesystem::path output_directory = argv[3];

        const std::string graph_text = ggml::hrx::tool::read_file(graph_path);
        if (graph_text.empty()) {
            throw std::runtime_error("cannot read " + graph_path.string());
        }
        ggml::hrx::graph_plan graph = ggml::hrx::graph_plan::deserialize_json(graph_text);
        if (!graph.valid()) {
            throw std::runtime_error("invalid normalized graph: " + (graph.diagnostics().errors.empty() ?
                                                                         std::string("unknown error") :
                                                                         graph.diagnostics().errors.front()));
        }
        ggml::hrx::pattern_registry patterns;
        ggml::hrx::llm_patterns::register_patterns(patterns);
        ggml::hrx::matcher::recognize(graph, patterns);
        ggml::hrx::planner::select_recipes(graph, target);
        const ggml::hrx::program_selection selection = ggml::hrx::program_selection::select(graph, target);
        const ggml::hrx::ProgramPlan       plan = ggml::hrx::build_transitional_program(graph, selection, target, 0);
        if (!plan.valid()) {
            throw std::runtime_error("cannot recover program: " +
                                     (plan.errors.empty() ? std::string("unknown error") : plan.errors.front()));
        }

        const ggml::hrx::kernel_corpus &    corpus       = ggml::hrx::get_qwen_kernel_corpus();
        const ggml::hrx::CommandProgram     commands     = ggml::hrx::build_command_program(plan, corpus);
        const ggml::hrx::VerificationResult verification = ggml::hrx::verify_command_program(plan, corpus, commands);

        std::filesystem::create_directories(output_directory);
        const std::string & readable_program = plan.semantic_witness;
        write_file(output_directory / "program.txt", readable_program);
        write_file(output_directory / "graph-plan.txt", graph.format());
        write_file(output_directory / "graph-plan.json", graph.serialize_json());
        write_file(output_directory / "graph-plan.dot", graph.dot());
        write_file(output_directory / "program-selection.txt", selection.format());
        write_file(output_directory / "program-selection.json", selection.serialize_json());
        write_file(output_directory / "program-summary.txt", readable_program);
        write_file(output_directory / "semantic-witness.txt", plan.semantic_witness);
        write_file(output_directory / "program.json", ggml::hrx::serialize_schedule_json(plan.schedule));
        if (!plan.fusion_search_text.empty()) {
            write_file(output_directory / "fusion-search.txt", plan.fusion_search_text);
            write_file(output_directory / "fusion-search.json", plan.fusion_search_json);
            write_file(output_directory / "fusion-regions.dot", plan.fusion_regions_dot);
        }
        if (!plan.logical_program_text.empty()) {
            write_file(output_directory / "logical-program.txt", plan.logical_program_text);
            write_file(output_directory / "logical-program.json", plan.logical_program_json);
            write_file(output_directory / "logical-program.dot", plan.logical_program_dot);
        }
        write_file(output_directory / "resources.txt", ggml::hrx::format_resource_program(plan.resources));
        write_file(output_directory / "kernels.txt", ggml::hrx::format_kernel_corpus(corpus));
        write_file(output_directory / "kernels.json", ggml::hrx::serialize_kernel_corpus_json(corpus));
        write_file(output_directory / "commands.txt", ggml::hrx::format_command_program(commands));
        write_file(output_directory / "commands.json", ggml::hrx::serialize_command_program_json(commands));
        write_file(output_directory / "commands.dot", ggml::hrx::command_program_dot(commands));

        std::ostringstream status;
        status << "schema=ggml-hrx-plan-diagnostics-v1\n"
               << "workload=" << plan.schedule.workload << '\n'
               << "target=" << target << '\n'
               << "graph=uid-0\n"
               << "planner=" << plan.planner_identity << '\n'
               << "atom_fallbacks=" << plan.atom_fallback_count << '\n'
               << "operations=" << graph.operations().size() << '\n'
               << "dispatches=" << ggml::hrx::schedule_dispatch_count(plan.schedule) << '\n'
               << "commands=" << commands.commands.size() << '\n'
               << "valid=" << (verification.valid() ? "true" : "false") << '\n'
               << ggml::hrx::format_verification_summary(verification.errors);
        for (const std::string & warning : plan.warnings) {
            status << "warning=" << warning << '\n';
        }
        write_file(output_directory / "status.txt", status.str());
        write_file(output_directory / "verification-errors.txt",
                   ggml::hrx::format_verification_errors(verification.errors));

        std::cout << "dumped " << plan.schedule.workload << " graph=uid-0"
                  << " dispatches=" << ggml::hrx::schedule_dispatch_count(plan.schedule)
                  << " commands=" << commands.commands.size() << " valid=" << (verification.valid() ? "true" : "false")
                  << " to " << output_directory << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ggml-hrx-dump-program: " << error.what() << '\n';
        return 1;
    }
}
