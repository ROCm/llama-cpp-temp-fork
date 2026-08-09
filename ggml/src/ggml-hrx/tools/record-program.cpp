#include "command-program.h"
#include "executable-program.h"
#include "graph-ir.h"
#include "hrx_runtime.h"
#include "kernel-corpus.h"
#include "reactive-plan.h"
#include "tool-utils.h"
#include "transfer-manager.h"
#include "weight-residency.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ggml::hrx::tool::check_status;
using ggml::hrx::tool::write_file;

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: ggml-hrx-record-program normalized-graph.json target output-dir\n";
        return 2;
    }
    hrx_device_t device      = nullptr;
    hrx_stream_t stream      = nullptr;
    bool         initialized = false;
    try {
        const std::filesystem::path graph_path       = argv[1];
        const std::string           target           = argv[2];
        const std::filesystem::path output_directory = argv[3];
        const std::string           graph_text       = ggml::hrx::tool::read_file(graph_path);
        if (graph_text.empty()) {
            throw std::runtime_error("cannot read " + graph_path.string());
        }
        const ggml::hrx::Graph graph = ggml::hrx::Graph::deserialize_json(graph_text);
        if (!graph.valid()) {
            throw std::runtime_error("normalized graph is invalid");
        }
        const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(graph, target);
        if (!plan.valid()) {
            throw std::runtime_error("reactive plan is invalid");
        }
        const ggml::hrx::kernel_corpus & corpus   = ggml::hrx::get_kernel_corpus();
        const ggml::hrx::CommandProgram  commands = ggml::hrx::build_command_program(plan, corpus);
        if (!commands.valid()) {
            throw std::runtime_error(commands.errors.front());
        }

        check_status(hrx_gpu_initialize(0), "initialize GPU");
        initialized = true;
        check_status(hrx_gpu_device_get(0, &device), "get GPU device");
        check_status(hrx_stream_create(device, 0, &stream), "create stream");
        ggml::hrx::executable_preparation_options options;
        options.target       = target;
        size_t recorder_size = 1;
        for (const ggml::hrx::ResourceContract & resource : plan.resources.resources) {
            if (!resource.elidable) {
                recorder_size = std::max(recorder_size, resource.size);
            }
        }
        if (recorder_size > options.recorder_buffer_limit) {
            throw std::runtime_error("largest imported binding exceeds bounded recorder limit");
        }
        hrx_buffer_t recorder_buffer = nullptr;
        check_status(hrx_buffer_allocate(stream, recorder_size, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT,
                                         &recorder_buffer),
                     "allocate recorder buffer");
        ggml::hrx::executable_bindings executable_bindings;
        executable_bindings.snapshot.device_identity = target;
        for (const ggml::hrx::ResourceContract & resource : plan.resources.resources) {
            if (resource.elidable) {
                continue;
            }
            executable_bindings.snapshot.bindings.push_back(
                { resource.storage, 1, 1, recorder_size, 0, resource.size });
            ggml::hrx::executable_buffer_binding binding;
            binding.storage         = resource.storage;
            binding.buffer          = recorder_buffer;
            binding.buffer_identity = 1;
            binding.generation      = 1;
            binding.capacity        = recorder_size;
            binding.length          = resource.size;
            executable_bindings.storages.push_back(std::move(binding));
        }
        {
            ggml::hrx::transfer_manager transfers(device);
            if (!transfers.valid()) {
                throw std::runtime_error("transfer manager initialization failed: " + transfers.initialization_error());
            }
            ggml::hrx::executable_artifact_repository artifacts;
            ggml::hrx::weight_residency_cache         weights(device);
            ggml::hrx::prepared_executable_program    prepared = ggml::hrx::prepare_executable_program(
                device, stream, transfers, weights, artifacts, plan, corpus, commands, executable_bindings, options);
            std::filesystem::create_directories(output_directory);
            write_file(output_directory / "prepared.txt", prepared.format());
            write_file(output_directory / "prepared.json", prepared.serialize_json());
            const std::filesystem::path artifact_directory = output_directory / "artifacts";
            for (size_t i = 0; i < prepared.artifacts().size(); ++i) {
                const ggml::hrx::prepared_artifact_diagnostic & artifact = prepared.artifacts()[i];
                const std::filesystem::path                     directory =
                    artifact_directory / (std::to_string(i) + "-" + artifact.kernel_id);
                std::filesystem::create_directories(directory);
                write_file(directory / "manifest.json", artifact.manifest_json);
                write_file(directory / "compile-report.json", artifact.compile_report_json);
                write_file(directory / "final.loom", artifact.final_module_text);
                std::ostringstream launch;
                launch << "key=" << artifact.key << '\n'
                       << "workgroups=" << artifact.workgroup_count[0] << ',' << artifact.workgroup_count[1] << ','
                       << artifact.workgroup_count[2] << '\n'
                       << "workgroup_size=" << artifact.workgroup_size[0] << ',' << artifact.workgroup_size[1] << ','
                       << artifact.workgroup_size[2] << '\n'
                       << "subgroup_size=" << artifact.subgroup_size << '\n'
                       << "constant_bytes=" << artifact.constant_bytes << '\n'
                       << "binding_count=" << artifact.binding_count << '\n';
                write_file(directory / "launch.txt", launch.str());
            }
            std::string status = "schema=ggml-hrx-record-status-v1\nworkload=" + commands.workload +
                                 "\ntarget=" + target + "\ncommands=" + std::to_string(commands.commands.size()) +
                                 "\nnodes=" + std::to_string(prepared.node_count()) +
                                 "\nartifacts=" + std::to_string(prepared.artifact_count()) +
                                 "\ninstantiated=" + (prepared.valid() ? std::string("true") : std::string("false")) +
                                 "\nlaunched=false\n";
            write_file(output_directory / "status.txt", status);
            if (!prepared.valid()) {
                std::cerr << prepared.format();
                throw std::runtime_error("program preparation failed");
            }
            std::cout << "recorded workload=" << commands.workload << " target=" << target
                      << " artifacts=" << prepared.artifact_count() << " nodes=" << prepared.node_count()
                      << " instantiated=true launched=false\n";
        }
        hrx_buffer_release(recorder_buffer);
        hrx_stream_release(stream);
        stream = nullptr;
        hrx_device_release(device);
        device = nullptr;
        check_status(hrx_gpu_shutdown(), "shutdown GPU");
        initialized = false;
        return 0;
    } catch (const std::exception & error) {
        if (stream != nullptr) {
            hrx_stream_release(stream);
        }
        if (device != nullptr) {
            hrx_device_release(device);
        }
        if (initialized) {
            hrx_status_t status = hrx_gpu_shutdown();
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
        }
        std::cerr << "ggml-hrx-record-program: " << error.what() << '\n';
        return 1;
    }
}
