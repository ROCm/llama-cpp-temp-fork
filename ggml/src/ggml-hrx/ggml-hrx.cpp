#include "ggml-hrx.h"

#include "graph/command-program.h"
#include "graph/graph-ir.h"
#include "graph/reactive-plan.h"
#include "executable-program.h"
#include "kernel-corpus.h"
#include "kernel-corpus-json.h"
#include "transfer-manager.h"
#include "weight-residency.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "hrx_runtime.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace {

static constexpr size_t GGML_HRX_ALIGNMENT = 256;
static constexpr size_t GGML_HRX_DEBUG_MAXIMUM_SNAPSHOT_BINDING_BYTES = 64ull * 1024ull * 1024ull;
// GGML represents tensor locations as host pointers and derives view/arena
// offsets with ordinary pointer arithmetic. Device-local HRX buffers have no
// host address to return, so expose a non-null sentinel base solely as an
// offset coordinate system. tensor->data is never dereferenced for an HRX
// buffer: every upload, download, copy, and executable binding subtracts this
// base and applies the resulting byte offset to the buffer's HRX handle. The
// same sentinel can therefore be shared by allocations; the owning buffer
// context supplies identity and bounds.
static constexpr uintptr_t GGML_HRX_FAKE_PTR_BASE = 0x1000;
static std::atomic<uint64_t> g_allocation_generation { 1 };

struct ggml_backend_hrx_device_context;

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device;
    std::string name;
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device;
    hrx_buffer_t buffer;
    uint8_t * base;
    uint64_t identity;
    uint64_t generation;
};

struct ggml_backend_hrx_device_context {
    hrx_device_t device = nullptr;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    ggml_backend_buffer_type buft = {};
    ggml_backend_hrx_buffer_type_context buft_context = {};
    ggml::hrx::ReactivePlanCache plan_cache;
    std::unique_ptr<ggml::hrx::TransferManager> transfers;
    std::mutex active_stream_mutex;
    hrx_stream_t active_stream = nullptr;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context * device;
    hrx_stream_t stream;
    std::string name;
    struct DiagnosticOptions {
        std::filesystem::path directory;
        std::string level = "summary";
        bool graph_oracle = false;
    } diagnostics;
    struct ExecutionDebugOptions {
        size_t command_limit = SIZE_MAX;
        bool serialize_commands = false;
        bool split_commands = false;
        bool synchronize_launch = false;
        std::string workload;
        std::string sanitizer;
        std::string sanitizer_reporting;
    } execution_debug;
    std::mutex oracle_mutex;
    std::unordered_map<std::string, std::string> oracle_witnesses;
    const ggml::hrx::KernelCorpus * corpus = nullptr;
    std::unique_ptr<ggml::hrx::WeightResidencyCache> weights;
    ggml::hrx::ExecutableArtifactRepository artifacts;
    std::mutex execution_mutex;
    std::unordered_map<std::string, std::unique_ptr<ggml::hrx::PreparedExecutableProgram>> executables;
    std::vector<ggml::hrx::PreparedExecutableProgram *> pending_completions;
    uint64_t executable_builds = 0;
    uint64_t executable_hits = 0;
    uint64_t launches = 0;
};

struct ggml_backend_hrx_reg_context {
    bool initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx_reg_context() {
        for (auto & context : device_contexts) {
            if (context->device != nullptr) {
                hrx_device_release(context->device);
            }
        }
        if (initialized) {
            hrx_status_t status = hrx_gpu_shutdown();
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
        }
    }
};

static bool hrx_check(hrx_status_t status, const char * expression, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expression, message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

static std::optional<std::string> environment_string(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return std::nullopt;
    return std::string(value);
}

static std::string environment_string(const char * name, std::string default_value) {
    const std::optional<std::string> value = environment_string(name);
    return value ? *value : std::move(default_value);
}

static bool environment_enabled(const char * name) {
    const std::optional<std::string> value = environment_string(name);
    return value && *value != "0";
}

static size_t environment_size(const char * name, size_t default_value) {
    const std::optional<std::string> value = environment_string(name);
    if (!value) return default_value;
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value->c_str(), &end, 10);
    if (errno != 0 || end == value->c_str() || *end != '\0' || parsed > SIZE_MAX) {
        GGML_LOG_ERROR("invalid %s='%s'; expected a non-negative command count\n", name, value->c_str());
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

static bool debug_applies_to_workload(
        const ggml_backend_hrx_context::ExecutionDebugOptions & options,
        const std::string & workload) {
    return options.workload.empty() || workload == options.workload ||
        (workload.size() > options.workload.size() &&
         workload.compare(0, options.workload.size(), options.workload) == 0 &&
         workload[options.workload.size()] == '-');
}

#define HRX_CHECK(expression) hrx_check((expression), #expression, __FILE__, __LINE__)

static std::optional<std::string> device_string_property(
        hrx_device_t device, hrx_device_property_t property, const char * property_name) {
    // libhrx currently has no size-query form for string properties. Retry a
    // bounded dynamic buffer on the one status that specifically means the
    // destination was too small, rather than baking an ABI-dependent limit
    // into backend registration.
    std::vector<char> buffer(64);
    while (buffer.size() <= 4096) {
        hrx_status_t status = hrx_device_get_property(device, property, buffer.data(), buffer.size());
        if (hrx_status_is_ok(status)) return std::string(buffer.data());
        if (hrx_status_code(status) != HRX_STATUS_OUT_OF_RANGE) {
            hrx_check(status, property_name, __FILE__, __LINE__);
            return std::nullopt;
        }
        hrx_status_ignore(status);
        buffer.resize(buffer.size() * 2);
    }
    GGML_LOG_ERROR("%s exceeds the maximum supported property string length\n", property_name);
    return std::nullopt;
}

static ggml_guid_t ggml_backend_hrx_guid() {
    static ggml_guid guid = {
        0xd2, 0x3d, 0x72, 0x83, 0xb2, 0x82, 0x4d, 0xe0, 0x8a, 0x3e, 0x21, 0x1d, 0x68, 0x87, 0x2f, 0x4b,
    };
    return &guid;
}

static ggml_backend_hrx_device_context * device_context(ggml_backend_dev_t device) {
    return static_cast<ggml_backend_hrx_device_context *>(device->context);
}

static ggml_backend_hrx_buffer_context * buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

static size_t tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static const char * buffer_type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->name.c_str();
}

static void buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = buffer_context(buffer);
    if (context->buffer != nullptr) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * buffer_base(ggml_backend_buffer_t buffer) {
    return buffer_context(buffer)->base;
}

static void buffer_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(destination_offset <= buffer->size && size <= buffer->size - destination_offset);
    const std::string error = context->device->transfers->fill(
        context->buffer, destination_offset, size, &value, sizeof(value));
    if (!error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
}

static void buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(destination_offset <= buffer->size && size <= buffer->size - destination_offset);
    const std::string error = context->device->transfers->upload(
        data, context->buffer, destination_offset, size);
    if (!error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
}

static void buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    const size_t source_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(source_offset <= buffer->size && size <= buffer->size - source_offset);
    hrx_stream_t producer = nullptr;
    {
        std::lock_guard<std::mutex> lock(context->device->active_stream_mutex);
        producer = context->device->active_stream;
        if (producer != nullptr) hrx_stream_retain(producer);
    }
    const std::string error = context->device->transfers->download(
        producer, context->buffer, source_offset, data, size);
    if (producer != nullptr) hrx_stream_release(producer);
    if (!error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
}

static bool buffer_copy(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * destination) {
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer == nullptr || source_buffer->iface.get_base != buffer_base) {
        return false;
    }
    auto * source_context = buffer_context(source_buffer);
    auto * destination_context = buffer_context(buffer);
    if (source_context->device != destination_context->device) {
        return false;
    }
    const size_t source_offset = tensor_offset(source_context, source);
    const size_t destination_offset = tensor_offset(destination_context, destination);
    const size_t size = ggml_nbytes(source);
    if (source_offset > source_buffer->size || size > source_buffer->size - source_offset ||
        destination_offset > buffer->size || size > buffer->size - destination_offset) return false;
    hrx_stream_t producer = nullptr;
    {
        std::lock_guard<std::mutex> lock(destination_context->device->active_stream_mutex);
        producer = destination_context->device->active_stream;
        if (producer != nullptr) hrx_stream_retain(producer);
    }
    const std::string error = destination_context->device->transfers->copy(
        producer, source_context->buffer, source_offset,
        destination_context->buffer, destination_offset, size);
    if (producer != nullptr) hrx_stream_release(producer);
    if (!error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
    return error.empty();
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer->size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    const std::string error = context->device->transfers->fill(
        context->buffer, 0, buffer->size, &value, sizeof(value));
    if (!error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
}

static const ggml_backend_buffer_i buffer_i = {
    buffer_free, buffer_base, nullptr, buffer_memset, buffer_set, buffer_get, nullptr, nullptr, buffer_copy, buffer_clear, nullptr,
};

static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * type_context = static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_MEMORY_ACCESS_ALL, HRX_BUFFER_USAGE_DEFAULT, 0,
    };
    hrx_buffer_t allocation = nullptr;
    if (size > 0 &&
        !HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(type_context->device->device), params, size, &allocation))) {
        return nullptr;
    }
    const uint64_t generation = g_allocation_generation.fetch_add(1);
    auto * context = new (std::nothrow) ggml_backend_hrx_buffer_context {
        type_context->device, allocation, reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE),
        generation, generation,
    };
    if (context == nullptr) {
        if (allocation != nullptr) {
            hrx_buffer_release(allocation);
        }
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, buffer_i, context, size);
}

static size_t buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t buffer_max_size(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->device->memory_total;
}

static const ggml_backend_buffer_type_i buffer_type_i = {
    buffer_type_name, buffer_alloc, buffer_alignment, buffer_max_size, nullptr, nullptr,
};

static void dump_graph(const ggml_backend_hrx_context::DiagnosticOptions & options,
                       const ggml_cgraph * graph, const char * mode, const char * stage) {
    if (options.directory.empty()) {
        return;
    }
    static std::atomic<uint64_t> sequence = 0;
    static std::mutex mutex;
    const uint64_t id = sequence.fetch_add(1);
    try {
        const ggml::hrx::Graph normalized = ggml::hrx::ImportedGraph::import(graph).graph;
        const std::filesystem::path & directory = options.directory;
        std::filesystem::create_directories(directory);
        const std::string stem = std::to_string(id) + "-uid-" + std::to_string(graph->uid) + "-" + mode + "-" + stage;
        const std::filesystem::path dot_path = std::filesystem::path(directory) / (stem + ".dot");
        ggml_graph_dump_dot(graph, graph, dot_path.string().c_str());
        std::lock_guard<std::mutex> lock(mutex);
        const std::filesystem::path normalized_directory = std::filesystem::path(directory) / "normalized";
        std::filesystem::create_directories(normalized_directory);
        const std::filesystem::path json_path = normalized_directory / (normalized.fingerprint + ".json");
        if (!std::filesystem::exists(json_path)) {
            const std::filesystem::path temporary_path = json_path.string() + ".tmp";
            std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
            output << ggml::hrx::Graph::serialize_json(normalized) << '\n';
            output.close();
            std::filesystem::rename(temporary_path, json_path);
        }
        std::ofstream manifest(std::filesystem::path(directory) / "manifest.tsv", std::ios::app);
        manifest << id << '\t' << graph->uid << '\t' << mode << '\t' << stage << '\t'
                 << graph->n_nodes << '\t' << graph->n_leafs << '\t' << normalized.fingerprint << '\t'
                 << dot_path.filename().string() << '\t' << std::filesystem::relative(json_path, directory).string() << '\n';
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: graph dump failed: %s\n", __func__, error.what());
    }
}

static void write_atomic(const std::filesystem::path & path, const std::string & contents) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create diagnostic file " + temporary.string());
    output << contents;
    if (contents.empty() || contents.back() != '\n') output << '\n';
    output.close();
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        if (!std::filesystem::exists(path)) throw std::runtime_error("cannot publish diagnostic file " + path.string());
    }
}

static void write_atomic(const std::filesystem::path & path, const std::vector<uint8_t> & contents) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create diagnostic file " + temporary.string());
    output.write(reinterpret_cast<const char *>(contents.data()),
                 static_cast<std::streamsize>(contents.size()));
    output.close();
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        if (!std::filesystem::exists(path)) throw std::runtime_error("cannot publish diagnostic file " + path.string());
    }
}

static void dump_plan(const ggml_backend_hrx_context::DiagnosticOptions & options,
                      const ggml::hrx::ProgramPlan & plan,
                      const ggml::hrx::KernelCorpus & corpus) {
    if (options.directory.empty()) return;
    static std::mutex mutex;
    try {
        std::lock_guard<std::mutex> lock(mutex);
        std::string target = plan.target;
        std::replace_if(target.begin(), target.end(), [](char ch) { return !std::isalnum(static_cast<unsigned char>(ch)); }, '_');
        const std::filesystem::path directory = options.directory / "plans" /
            (plan.schedule.workload + "-" + plan.graph.fingerprint + "-" + target);
        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(plan, corpus);
        const ggml::hrx::VerificationResult command_verification =
            ggml::hrx::verify_command_program(plan, corpus, commands);
        write_atomic(directory / "program.txt", plan.semantic_witness);
        write_atomic(directory / "semantic-witness.txt", plan.semantic_witness);
        write_atomic(directory / "program.json", ggml::hrx::serialize_schedule_json(plan.schedule));
        if (!plan.fusion_search_text.empty()) {
            write_atomic(directory / "fusion-search.txt", plan.fusion_search_text);
            write_atomic(directory / "fusion-search.json", plan.fusion_search_json);
            write_atomic(directory / "fusion-regions.dot", plan.fusion_regions_dot);
        }
        if (!plan.logical_program_text.empty()) {
            write_atomic(directory / "logical-program.txt", plan.logical_program_text);
            write_atomic(directory / "logical-program.json", plan.logical_program_json);
            write_atomic(directory / "logical-program.dot", plan.logical_program_dot);
        }
        write_atomic(directory / "resources.txt", ggml::hrx::format_resource_program(plan.resources));
        write_atomic(directory / "kernels.txt", ggml::hrx::format_kernel_corpus(corpus));
        write_atomic(directory / "kernels.json", ggml::hrx::serialize_kernel_corpus_json(corpus));
        write_atomic(directory / "commands.txt", ggml::hrx::format_command_program(commands));
        write_atomic(directory / "commands.json", ggml::hrx::serialize_command_program_json(commands));
        write_atomic(directory / "commands.dot", ggml::hrx::command_program_dot(commands));
        std::ostringstream status;
        status << "schema=ggml-hrx-plan-diagnostics-v1\nlevel=" << options.level << "\nvalid="
               << (command_verification.valid() ? "true" : "false") << '\n';
        status << "planner=" << plan.planner_identity << '\n'
               << "atom_fallbacks=" << plan.atom_fallback_count << '\n';
        for (const std::string & warning : plan.warnings) status << "warning=" << warning << '\n';
        status << ggml::hrx::format_verification_summary(command_verification.errors);
        write_atomic(directory / "status.txt", status.str());
        write_atomic(directory / "verification-errors.txt", ggml::hrx::format_verification_errors(command_verification.errors));
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: plan dump failed: %s\n", __func__, error.what());
    }
}

static void dump_execution_state(
    const ggml_backend_hrx_context::DiagnosticOptions & options,
    const ggml::hrx::ProgramPlan & plan, const ggml::hrx::ExecutableBindings & bindings,
    const ggml::hrx::PreparedExecutableProgram & executable,
    const ggml::hrx::TransferManagerStats & transfer_stats,
    const ggml::hrx::WeightResidencyStats & weight_stats, const char * state,
    const std::string & detail, uint64_t builds, uint64_t hits, uint64_t launches) {
    if (options.directory.empty()) return;
    try {
        std::string target = plan.target;
        std::replace_if(target.begin(), target.end(), [](char ch) {
            return !std::isalnum(static_cast<unsigned char>(ch));
        }, '_');
        const std::filesystem::path directory = options.directory / "runtime" /
            (plan.schedule.workload + "-" + plan.graph.fingerprint + "-" + target + "-" +
             ggml::hrx::fingerprint_bindings(bindings.snapshot).value);
        write_atomic(directory / "bindings.txt", ggml::hrx::format_binding_snapshot(bindings.snapshot, false));
        write_atomic(directory / "bindings.json", ggml::hrx::serialize_binding_snapshot_json(bindings.snapshot, false));
        write_atomic(directory / "bindings-detail.txt", bindings.format());
        write_atomic(directory / "bindings-detail.json", bindings.serialize_json());
        write_atomic(directory / "executable.txt", executable.format());
        write_atomic(directory / "executable.json", executable.serialize_json());
        write_atomic(directory / "transfers.txt", transfer_stats.format());
        write_atomic(directory / "weights.txt", ggml::hrx::format_weight_residency_stats(weight_stats));
        std::ostringstream status;
        status << "schema=ggml-hrx-runtime-status-v1\nstate=" << state
               << "\nvalid=" << (executable.valid() ? "true" : "false")
               << "\nbuilds=" << builds << "\nhits=" << hits << "\nlaunches=" << launches
               << "\nretained_bytes=" << executable.retained_bytes() << '\n';
        status << "borrowed_device_weight_bytes=" << executable.borrowed_device_weight_bytes() << '\n'
               << "resident_host_weight_bytes=" << executable.resident_host_weight_bytes() << '\n'
               << "resident_weight_allocations=" << weight_stats.allocation_count << '\n'
               << "resident_weight_bytes=" << weight_stats.resident_bytes << '\n';
        if (!detail.empty()) status << "detail=" << detail << '\n';
        write_atomic(directory / "status.txt", status.str());
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: runtime dump failed: %s\n", __func__, error.what());
    }
}

static enum ggml_backend_graph_claim_result graph_claim(ggml_backend_t backend, const ggml_cgraph * graph, enum ggml_backend_graph_claim_mode mode) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (!context->diagnostics.graph_oracle || graph->n_nodes == 0) {
        return GGML_BACKEND_GRAPH_CLAIM_DECLINED;
    }
    dump_graph(context->diagnostics, graph, mode == GGML_BACKEND_GRAPH_CLAIM_MODE_MEASURE ? "measure" : "execute", "raw-oracle");
    if (mode == GGML_BACKEND_GRAPH_CLAIM_MODE_EXECUTE) {
        // Use the same executable-reachability import as the reactive path.
        // The raw pre-placement graph carries an additional, unused leaf list;
        // importing that list perturbs ValueIds and therefore the otherwise
        // identical ABI ordering of invocation boundary bindings.
        const ggml::hrx::Graph normalized = ggml::hrx::ImportedGraph::import(graph).graph;
        const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(normalized, context->device->architecture);
        if (!plan.valid()) {
            GGML_LOG_WARN("%s: diagnostic oracle could not build a plan: %s\n", __func__, plan.errors.front().c_str());
        } else {
            std::lock_guard<std::mutex> lock(context->oracle_mutex);
            context->oracle_witnesses[plan.schedule.workload] = plan.semantic_witness;
            GGML_LOG_WARN("%s: recorded diagnostic oracle workload '%s' with %zu operations, %zu values, %zu roots, and %zu dispatches\n",
                __func__, plan.schedule.workload.c_str(), plan.graph.operations.size(), plan.graph.values.size(), plan.graph.roots.size(),
                ggml::hrx::schedule_dispatch_count(plan.schedule));
        }
    }
    // This hook is now only a pre-placement diagnostic oracle. Ordinary
    // supports_op placement and graph_compute are the launch architecture.
    return GGML_BACKEND_GRAPH_CLAIM_DECLINED;
}

static const char * backend_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static bool tensor_hrx_binding(const ggml_tensor * tensor,
                               ggml_backend_hrx_buffer_context ** out_context,
                               size_t * out_offset) {
    if (tensor == nullptr) return false;
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || buffer->iface.get_base != buffer_base) return false;
    auto * context = buffer_context(buffer);
    const size_t offset = tensor_offset(context, tensor);
    if (context->buffer == nullptr || offset > buffer->size || ggml_nbytes(tensor) > buffer->size - offset) return false;
    *out_context = context;
    *out_offset = offset;
    return true;
}

static void backend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor,
                                     const void * data, size_t offset, size_t size) {
    auto * backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    ggml_backend_hrx_buffer_context * context = nullptr;
    size_t tensor_base = 0;
    if (!tensor_hrx_binding(tensor, &context, &tensor_base) || offset > ggml_nbytes(tensor) ||
        size > ggml_nbytes(tensor) - offset) {
        GGML_LOG_ERROR("%s: invalid or failed HRX tensor upload\n", __func__);
        return;
    }
    std::string error = backend_context->device->transfers->upload(
        data, context->buffer, tensor_base + offset, size);
    if (error.empty()) error = backend_context->device->transfers->join(backend_context->stream);
    if (!error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
}

static bool backend_copy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                      const ggml_tensor * source, ggml_tensor * destination) {
    GGML_UNUSED(backend_src);
    auto * destination_backend = static_cast<ggml_backend_hrx_context *>(backend_dst->context);
    ggml_backend_hrx_buffer_context * destination_context = nullptr;
    size_t destination_offset = 0;
    if (!tensor_hrx_binding(destination, &destination_context, &destination_offset)) return false;
    ggml_backend_hrx_buffer_context * source_context = nullptr;
    size_t source_offset = 0;
    if (tensor_hrx_binding(source, &source_context, &source_offset)) {
        if (source_context->device != destination_context->device) return false;
        std::string error = destination_backend->device->transfers->copy(
            destination_backend->stream, source_context->buffer, source_offset,
            destination_context->buffer, destination_offset, ggml_nbytes(source));
        if (error.empty()) error = destination_backend->device->transfers->join(destination_backend->stream);
        return error.empty();
    }
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer != nullptr && ggml_backend_buffer_is_host(source_buffer)) {
        std::string error = destination_backend->device->transfers->upload(
            source->data, destination_context->buffer, destination_offset, ggml_nbytes(source));
        if (error.empty()) error = destination_backend->device->transfers->join(destination_backend->stream);
        return error.empty();
    }
    return false;
}

static void backend_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    HRX_CHECK(hrx_stream_synchronize(context->stream));
    {
        std::lock_guard<std::mutex> lock(context->device->active_stream_mutex);
        if (context->device->active_stream == context->stream) context->device->active_stream = nullptr;
    }
    hrx_stream_release(context->stream);
    delete context;
    delete backend;
}

static void backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (!HRX_CHECK(hrx_stream_synchronize(context->stream))) return;
    const std::string transfer_error = context->device->transfers->collect();
    if (!transfer_error.empty()) GGML_LOG_ERROR("%s: %s\n", __func__, transfer_error.c_str());
    std::lock_guard<std::mutex> lock(context->execution_mutex);
    for (ggml::hrx::PreparedExecutableProgram * executable : context->pending_completions) {
        const ggml::hrx::ErrorResult error = executable->complete_after_synchronize();
        if (error) GGML_LOG_ERROR("%s: %s\n", __func__, error->c_str());
    }
    context->pending_completions.clear();
}

static ggml::hrx::ExecutableBindings resolve_executable_bindings(
    const ggml_backend_hrx_context & context, const ggml::hrx::ExecutionFrame & frame,
    std::vector<std::string> & errors) {
    ggml::hrx::ExecutableBindings result;
    result.snapshot.device_identity = context.device->name + ":" + context.device->architecture;
    if (frame.storage_roots.size() != frame.plan->graph.storages.size()) {
        errors.push_back("runtime storage-root count does not match the normalized graph");
        return result;
    }
    for (const ggml::hrx::ResourceContract & resource : frame.plan->resources.resources) {
        if (resource.elidable) continue;
        const ggml_tensor * root = frame.storage_roots[resource.storage];
        if (root == nullptr || root->buffer == nullptr || root->data == nullptr) {
            errors.push_back("storage " + std::to_string(resource.storage) + " has no live tensor allocation");
            continue;
        }
        ggml::hrx::ExecutableBufferBinding binding;
        binding.storage = resource.storage;
        binding.length = resource.size;
        binding.weight = resource.weight;
        binding.mutable_state = resource.mutable_state;
        binding.exported = resource.exported;
        ggml_backend_hrx_buffer_context * hrx_context = nullptr;
        size_t root_offset = 0;
        if (tensor_hrx_binding(root, &hrx_context, &root_offset)) {
            binding.buffer = hrx_context->buffer;
            binding.buffer_identity = hrx_context->identity;
            binding.generation = hrx_context->generation;
            binding.capacity = root->buffer->size;
            binding.offset = root_offset;
        } else if (ggml_backend_buffer_is_host(root->buffer)) {
            void * base = ggml_backend_buffer_get_base(root->buffer);
            const size_t capacity = ggml_backend_buffer_get_size(root->buffer);
            if (base == nullptr || static_cast<const uint8_t *>(root->data) < static_cast<const uint8_t *>(base)) {
                errors.push_back("host storage " + std::to_string(resource.storage) + " has no valid backing base");
                continue;
            }
            const size_t host_offset = static_cast<size_t>(
                static_cast<const uint8_t *>(root->data) - static_cast<const uint8_t *>(base));
            if (host_offset > capacity || resource.size > capacity - host_offset) {
                errors.push_back("host storage " + std::to_string(resource.storage) + " exceeds its backing allocation");
                continue;
            }
            binding.host_data = base;
            const uint64_t buffer_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(root->buffer));
            const uint64_t base_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
            binding.buffer_identity = buffer_address ^ (base_address + 0x9e3779b97f4a7c15ull +
                                                        (buffer_address << 6) + (buffer_address >> 2));
            binding.generation = 1;
            binding.capacity = capacity;
            binding.offset = host_offset;
            binding.initialize_from_host = resource.weight;
            binding.upload_before_launch = (resource.imported && !resource.weight) || resource.mutable_state;
            binding.download_after_completion = resource.exported || resource.mutable_state;
        } else {
            errors.push_back("storage " + std::to_string(resource.storage) +
                             " belongs to an unsupported foreign buffer");
            continue;
        }
        result.snapshot.bindings.push_back({ binding.storage, binding.buffer_identity,
            binding.generation, binding.capacity, binding.offset, binding.length });
        result.storages.push_back(binding);
    }
    const ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_binding_snapshot(*frame.plan, result.snapshot);
    errors.insert(errors.end(), verification.errors.begin(), verification.errors.end());
    return result;
}

// Converts one eagerly presented GGML graph into a cached, allocation-specific
// executable and submits it asynchronously. Cold execution recovers and
// verifies the semantic plan, compiles/records its command program, and caches
// that frozen executable. Warm execution resolves current tensor allocations,
// rebinds the cached commands, and launches with predictable host work.
static enum ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);

    // Publish the compute stream used by this backend instance. Synchronous
    // readback and device copies use it as the producer timeline.
    {
        std::lock_guard<std::mutex> lock(context->device->active_stream_mutex);
        context->device->active_stream = context->stream;
    }

    // Buffer uploads and fills are recorded on TransferManager's dedicated
    // stream, not this compute stream. Publish any pending transfer batch and
    // insert its timeline point as a device-side wait here. join() does not
    // wait on the host; it establishes the required cross-stream ordering.
    const std::string transfer_join_error = context->device->transfers->join(context->stream);
    if (!transfer_join_error.empty()) {
        GGML_LOG_ERROR("%s: cannot join pending transfers: %s\n", __func__, transfer_join_error.c_str());
        return GGML_STATUS_FAILED;
    }

    // Normalize the eager graph and recover or look up its frozen schedule.
    dump_graph(context->diagnostics, graph, "execute", "reactive-split");
    ggml::hrx::ExecutionFrame frame = context->device->plan_cache.prepare(graph, context->device->architecture);
    if (!frame.valid()) {
        GGML_LOG_ERROR("%s: reactive plan preparation failed: %s\n", __func__, frame.errors.empty() ? "unknown error" : frame.errors.front().c_str());
        return GGML_STATUS_FAILED;
    }

    // When the optional pre-placement oracle is enabled, require the eager
    // graph to recover the exact semantic witness captured by that oracle.
    if (context->diagnostics.graph_oracle) {
        std::lock_guard<std::mutex> lock(context->oracle_mutex);
        auto oracle = context->oracle_witnesses.find(frame.plan->schedule.workload);
        if (oracle == context->oracle_witnesses.end()) {
            GGML_LOG_ERROR("%s: no diagnostic oracle witness for %s\n", __func__, frame.plan->schedule.workload.c_str());
            return GGML_STATUS_FAILED;
        }
        if (oracle->second != frame.plan->semantic_witness) {
            GGML_LOG_ERROR("%s: raw and reactive schedule witnesses disagree for %s\n", __func__, frame.plan->schedule.workload.c_str());
            std::istringstream raw(oracle->second);
            std::istringstream reactive(frame.plan->semantic_witness);
            std::string raw_line;
            std::string reactive_line;
            size_t line = 0;
            while (std::getline(raw, raw_line) && std::getline(reactive, reactive_line)) {
                ++line;
                if (raw_line != reactive_line) {
                    GGML_LOG_ERROR("%s: first witness difference at line %zu: raw='%s' reactive='%s'\n",
                        __func__, line, raw_line.c_str(), reactive_line.c_str());
                    break;
                }
            }
            return GGML_STATUS_FAILED;
        }
    }

    // Plan reports are cold-path diagnostics; dump_plan internally becomes a
    // no-op unless a diagnostic directory was configured at initialization.
    const ggml::hrx::PlanCacheStats stats = context->device->plan_cache.stats();
    GGML_LOG_WARN("%s: verified reactive %s plan with %zu operations, %zu dispatches, cache builds=%llu hits=%llu\n",
        __func__, frame.plan->schedule.workload.c_str(), frame.plan->graph.operations.size(),
        ggml::hrx::schedule_dispatch_count(frame.plan->schedule),
        static_cast<unsigned long long>(stats.builds), static_cast<unsigned long long>(stats.hits));
    dump_plan(context->diagnostics, *frame.plan, *context->corpus);

    // Resolve semantic graph storages to this call's concrete HRX/host
    // allocations. The fingerprint below prevents reusing an executable with
    // stale buffer identities, generations, or offsets.
    std::vector<std::string> binding_errors;
    ggml::hrx::ExecutableBindings bindings = resolve_executable_bindings(*context, frame, binding_errors);
    if (!binding_errors.empty()) {
        GGML_LOG_ERROR("%s: live binding resolution failed: %s\n", __func__, binding_errors.front().c_str());
        return GGML_STATUS_FAILED;
    }
    const ggml::hrx::AllocationFingerprint allocation = ggml::hrx::fingerprint_bindings(bindings.snapshot);

    // Collapse initialization-time debug configuration to simple local values
    // before constructing the executable cache key.
    const bool execution_debug_enabled = debug_applies_to_workload(
        context->execution_debug, frame.plan->schedule.workload);
    const size_t debug_command_limit = execution_debug_enabled ?
        context->execution_debug.command_limit : SIZE_MAX;
    const bool debug_serialize_commands = execution_debug_enabled && context->execution_debug.serialize_commands;
    const bool debug_split_commands = execution_debug_enabled && context->execution_debug.split_commands;
    const bool debug_synchronize = execution_debug_enabled && context->execution_debug.synchronize_launch;
    const std::string debug_sanitizer = execution_debug_enabled ? context->execution_debug.sanitizer : std::string();
    const std::string debug_sanitizer_reporting = execution_debug_enabled ?
        context->execution_debug.sanitizer_reporting : std::string();
    const std::string executable_key = frame.plan->graph.fingerprint + '|' +
        frame.plan->schedule.workload + '|' + frame.plan->target + '|' +
        context->corpus->recipe_digest + '|' + context->corpus->corpus_digest + '|' + allocation.value +
        "|debug_limit=" + std::to_string(debug_command_limit) +
        "|debug_serialize=" + std::to_string(debug_serialize_commands) +
        "|debug_split=" + std::to_string(debug_split_commands) +
        "|sanitizer=" + debug_sanitizer +
        "|sanitizer_reporting=" + debug_sanitizer_reporting;

    // Serialize executable cache mutation and launch. Prepared programs retain
    // mutable bindings and completion state, so they are not concurrently
    // rebound or submitted from multiple callers.
    std::lock_guard<std::mutex> lock(context->execution_mutex);
    ggml::hrx::PreparedExecutableProgram * executable = nullptr;
    auto found = context->executables.find(executable_key);
    if (found != context->executables.end()) {
        executable = found->second.get();
        ++context->executable_hits;
    } else {
        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(*frame.plan, *context->corpus);
        if (!commands.valid()) {
            GGML_LOG_ERROR("%s: command construction failed: %s\n", __func__, commands.errors.front().c_str());
            return GGML_STATUS_FAILED;
        }
        ggml::hrx::ExecutablePreparationOptions options;
        options.target = frame.plan->target;
        options.command_limit = debug_command_limit;
        options.serialize_commands = debug_serialize_commands;
        options.split_commands = debug_split_commands;
        options.sanitizer = debug_sanitizer;
        options.sanitizer_reporting = debug_sanitizer_reporting;
        auto prepared = std::make_unique<ggml::hrx::PreparedExecutableProgram>(
            ggml::hrx::prepare_executable_program(context->device->device, context->stream,
                *context->device->transfers,
                *context->weights, context->artifacts, *frame.plan, *context->corpus, commands, bindings, options));
        if (!prepared->valid()) {
            dump_execution_state(context->diagnostics, *frame.plan, bindings, *prepared,
                context->device->transfers->stats(),
                context->weights->stats(),
                "prepare_failed", prepared->errors().empty() ? "unknown error" : prepared->errors().front(),
                context->executable_builds, context->executable_hits, context->launches);
            GGML_LOG_ERROR("%s: executable preparation failed: %s\n", __func__,
                prepared->errors().empty() ? "unknown error" : prepared->errors().front().c_str());
            return GGML_STATUS_FAILED;
        }
        executable = prepared.get();
        context->executables.emplace(executable_key, std::move(prepared));
        ++context->executable_builds;
        GGML_LOG_WARN("%s: prepared %s executable: nodes=%zu artifacts=%zu retained=%zu bytes builds=%llu\n",
            __func__, frame.plan->schedule.workload.c_str(), executable->node_count(),
            executable->artifact_count(), executable->retained_bytes(),
            static_cast<unsigned long long>(context->executable_builds));
    }

    // Patch allocation-dependent handles into the prepared commands, then
    // submit the whole program asynchronously to the compute stream.
    const ggml::hrx::ErrorResult rebind_error = executable->rebind(bindings);
    if (rebind_error) {
        GGML_LOG_ERROR("%s: executable rebinding failed: %s\n", __func__, rebind_error->c_str());
        return GGML_STATUS_FAILED;
    }
    const ggml::hrx::ErrorResult launch_error = executable->launch(context->stream);
    if (launch_error) {
        dump_execution_state(context->diagnostics, *frame.plan, bindings, *executable,
            context->device->transfers->stats(),
            context->weights->stats(),
            "launch_failed", *launch_error, context->executable_builds, context->executable_hits, context->launches);
        GGML_LOG_ERROR("%s: HRX graph launch failed: %s\n", __func__, launch_error->c_str());
        return GGML_STATUS_FAILED;
    }

    // Debug prefix/synchronization modes deliberately replace the normal async
    // contract so failures can be localized to a command boundary.
    if (debug_synchronize || executable->command_prefix()) {
        if (!HRX_CHECK(hrx_stream_synchronize(context->stream))) {
            dump_execution_state(context->diagnostics, *frame.plan, bindings, *executable,
                context->device->transfers->stats(), context->weights->stats(),
                "debug_synchronize_failed", "stream synchronization failed",
                context->executable_builds, context->executable_hits, context->launches);
            return GGML_STATUS_FAILED;
        }
        if (executable->command_prefix()) {
            std::vector<uint8_t> transient_snapshot;
            const ggml::hrx::ErrorResult snapshot_error = executable->snapshot_transients(transient_snapshot);
            if (snapshot_error) {
                GGML_LOG_ERROR("%s: debug-prefix transient snapshot failed: %s\n", __func__, snapshot_error->c_str());
                return GGML_STATUS_FAILED;
            }
            std::vector<ggml::hrx::PreparedBindingSnapshot> output_snapshots;
            const ggml::hrx::ErrorResult output_snapshot_error =
                executable->snapshot_last_command_outputs(
                    output_snapshots, GGML_HRX_DEBUG_MAXIMUM_SNAPSHOT_BINDING_BYTES);
            if (output_snapshot_error) {
                GGML_LOG_ERROR("%s: debug-prefix output snapshot failed: %s\n",
                    __func__, output_snapshot_error->c_str());
                return GGML_STATUS_FAILED;
            }
            const ggml::hrx::ErrorResult completion_error = executable->complete_after_synchronize();
            if (completion_error) {
                GGML_LOG_ERROR("%s: debug-prefix readback failed: %s\n", __func__, completion_error->c_str());
                return GGML_STATUS_FAILED;
            }
            if (!context->diagnostics.directory.empty()) {
                const std::filesystem::path snapshot_directory =
                    context->diagnostics.directory / "snapshots" / frame.plan->schedule.workload;
                const std::string stem = "prefix-" + std::to_string(executable->commands().size());
                try {
                    write_atomic(snapshot_directory / (stem + "-transients.bin"), transient_snapshot);
                    std::ostringstream metadata;
                    metadata << "schema=ggml-hrx-transient-snapshot-v1\n"
                             << "workload=" << frame.plan->schedule.workload << '\n'
                             << "commands=" << executable->commands().size() << '\n'
                             << "bytes=" << transient_snapshot.size() << '\n'
                             << "allocation_fingerprint=" << executable->allocation_fingerprint().value << '\n';
                    for (size_t i = 0; i < output_snapshots.size(); ++i) {
                        std::string name = output_snapshots[i].name;
                        std::replace_if(name.begin(), name.end(), [](char character) {
                            return !std::isalnum(static_cast<unsigned char>(character)) && character != '-' && character != '_';
                        }, '_');
                        const std::string file_name = stem + "-output-" + std::to_string(i) + "-" + name + ".bin";
                        metadata << "output=" << i << '\t' << output_snapshots[i].name << '\t'
                                 << ggml::hrx::resource_access_name(output_snapshots[i].access) << '\t'
                                 << output_snapshots[i].length << '\t';
                        if (output_snapshots[i].bytes.empty() && output_snapshots[i].length != 0) {
                            metadata << "omitted:binding-exceeds-" <<
                                GGML_HRX_DEBUG_MAXIMUM_SNAPSHOT_BINDING_BYTES << "-bytes\n";
                        } else {
                            write_atomic(snapshot_directory / file_name, output_snapshots[i].bytes);
                            metadata << file_name << '\n';
                        }
                    }
                    write_atomic(snapshot_directory / (stem + ".txt"), metadata.str());
                } catch (const std::exception & error) {
                    GGML_LOG_ERROR("%s: debug-prefix snapshot write failed: %s\n", __func__, error.what());
                    return GGML_STATUS_FAILED;
                }
            }
            dump_execution_state(context->diagnostics, *frame.plan, bindings, *executable,
                context->device->transfers->stats(), context->weights->stats(),
                "debug_prefix_complete", "intentional stop after synchronized command prefix",
                context->executable_builds, context->executable_hits, context->launches);
            GGML_LOG_WARN("%s: synchronized commands [0, %zu); exposing diagnostic partial outputs\n",
                __func__, executable->commands().size());
            return GGML_STATUS_SUCCESS;
        }
    }

    // Retain completion bookkeeping until backend_synchronize observes the
    // stream, publishes host-visible results, and releases transient state.
    if (std::find(context->pending_completions.begin(), context->pending_completions.end(), executable) ==
        context->pending_completions.end()) {
        context->pending_completions.push_back(executable);
    }
    ++context->launches;
    dump_execution_state(context->diagnostics, *frame.plan, bindings, *executable,
        context->device->transfers->stats(),
        context->weights->stats(),
        "submitted", {}, context->executable_builds, context->executable_hits, context->launches);
    GGML_LOG_DEBUG("%s: submitted %s launch=%llu cache_hits=%llu\n", __func__,
        frame.plan->schedule.workload.c_str(), static_cast<unsigned long long>(context->launches),
        static_cast<unsigned long long>(context->executable_hits));
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i backend_i = {
    backend_name, backend_free,
    backend_set_tensor_async, nullptr, nullptr, nullptr, backend_copy_tensor_async,
    backend_synchronize,
    nullptr, nullptr, nullptr, nullptr,
    graph_compute,
    nullptr, nullptr, nullptr,
    graph_claim,
};

static const char * device_name(ggml_backend_dev_t device) {
    return device_context(device)->name.c_str();
}

static const char * device_description(ggml_backend_dev_t device) {
    return device_context(device)->description.c_str();
}

static void device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    *free = device_context(device)->memory_total;
    *total = device_context(device)->memory_total;
}

static enum ggml_backend_dev_type device_type(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void device_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    props->name = device_name(device);
    props->description = device_description(device);
    device_memory(device, &props->memory_free, &props->memory_total);
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps = { false, false, false, false };
}

static ggml_backend_t device_init(ggml_backend_dev_t device, const char * parameters) {
    GGML_UNUSED(parameters);
    auto * device_ctx = device_context(device);
    hrx_stream_t stream = nullptr;
    if (!HRX_CHECK(hrx_stream_create(device_ctx->device, 0, &stream))) {
        return nullptr;
    }
    // Snapshot environment configuration once. Steady-state graph execution
    // only reads the parsed context fields below.
    const std::optional<std::string> dump_directory = environment_string("GGML_HRX_DUMP_GRAPH_DIR");
    const std::string dump_level = environment_string("GGML_HRX_DUMP_LEVEL", "summary");
    const std::optional<std::string> sanitizer = environment_string("GGML_HRX_LOOM_SANITIZER");
    const std::optional<std::string> sanitizer_reporting =
        environment_string("GGML_HRX_LOOM_SANITIZER_REPORTING");
    const std::optional<std::string> debug_workload = environment_string("GGML_HRX_DEBUG_WORKLOAD");
    auto * context = new (std::nothrow) ggml_backend_hrx_context;
    if (context != nullptr) {
        context->device = device_ctx;
        context->stream = stream;
        context->name = device_ctx->name;
        context->corpus = &ggml::hrx::get_qwen_kernel_corpus(device_ctx->architecture.c_str());
        const ggml::hrx::VerificationResult corpus_verification = ggml::hrx::verify_kernel_corpus(*context->corpus);
        if (!corpus_verification.valid()) {
            GGML_LOG_ERROR("%s: embedded kernel corpus validation failed: %s\n", __func__, corpus_verification.errors.front().c_str());
            delete context;
            context = nullptr;
        }
        if (context != nullptr) {
            context->weights = std::make_unique<ggml::hrx::WeightResidencyCache>(device_ctx->device);
            if (!context->weights->valid()) {
                GGML_LOG_ERROR("%s: weight residency initialization failed: %s\n", __func__,
                    context->weights->initialization_error().c_str());
                delete context;
                context = nullptr;
            }
        }
    }
    if (context != nullptr) {
        context->diagnostics.graph_oracle = environment_enabled("GGML_HRX_GRAPH_ORACLE");
        if (dump_directory) context->diagnostics.directory = *dump_directory;
        context->diagnostics.level = dump_level;
        context->execution_debug.command_limit = environment_size("GGML_HRX_DEBUG_COMMAND_LIMIT", SIZE_MAX);
        context->execution_debug.serialize_commands = environment_enabled("GGML_HRX_DEBUG_SERIALIZE_COMMANDS");
        context->execution_debug.split_commands = environment_enabled("GGML_HRX_DEBUG_SPLIT_COMMANDS");
        context->execution_debug.synchronize_launch = environment_enabled("GGML_HRX_DEBUG_SYNCHRONIZE");
        if (debug_workload) context->execution_debug.workload = *debug_workload;
        if (sanitizer) context->execution_debug.sanitizer = *sanitizer;
        if (sanitizer_reporting) context->execution_debug.sanitizer_reporting = *sanitizer_reporting;
        if (context->execution_debug.command_limit != SIZE_MAX || context->execution_debug.serialize_commands ||
            context->execution_debug.split_commands || context->execution_debug.synchronize_launch ||
            !context->execution_debug.sanitizer.empty()) {
            GGML_LOG_WARN("HRX execution debugging: workload=%s limit=%s serialize=%d split=%d sync=%d sanitizer=%s\n",
                context->execution_debug.workload.empty() ? "all" : context->execution_debug.workload.c_str(),
                context->execution_debug.command_limit == SIZE_MAX ? "all" :
                    std::to_string(context->execution_debug.command_limit).c_str(),
                context->execution_debug.serialize_commands, context->execution_debug.split_commands,
                context->execution_debug.synchronize_launch,
                context->execution_debug.sanitizer.empty() ? "none" : context->execution_debug.sanitizer.c_str());
        }
    }
    auto * backend = context != nullptr ? new (std::nothrow) ggml_backend { ggml_backend_hrx_guid(), backend_i, device, context } : nullptr;
    if (backend == nullptr) {
        delete context;
        hrx_stream_release(stream);
    }
    return backend;
}

static ggml_backend_buffer_type_t device_buffer_type(ggml_backend_dev_t device) {
    return &device_context(device)->buft;
}

static bool device_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    return op != nullptr && ggml::hrx::eager_capability_declared(op->op);
}

static bool device_supports_buffer_type(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    // Host-backed tensors are valid imports. The residency planner, not the
    // outer scheduler, will eventually decide whether and when to stage them.
    return buft == &device_context(device)->buft || ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i device_i = {
    device_name, device_description, device_memory, device_type, device_props, device_init, device_buffer_type,
    nullptr, nullptr, device_supports_op, device_supports_buffer_type, nullptr, nullptr, nullptr, nullptr,
};

static const char * registry_name(ggml_backend_reg_t registry) {
    GGML_UNUSED(registry);
    return "HRX";
}

static size_t registry_device_count(ggml_backend_reg_t registry) {
    return static_cast<ggml_backend_hrx_reg_context *>(registry->context)->devices.size();
}

static ggml_backend_dev_t registry_device(ggml_backend_reg_t registry, size_t index) {
    auto * context = static_cast<ggml_backend_hrx_reg_context *>(registry->context);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * registry_proc(ggml_backend_reg_t registry, const char * name) {
    GGML_UNUSED(registry);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i registry_i = { registry_name, registry_device_count, registry_device, registry_proc };

static std::unique_ptr<ggml_backend_hrx_reg_context> create_registry_context() {
    auto context = std::make_unique<ggml_backend_hrx_reg_context>();
    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }
    int count = 0;
    if (!HRX_CHECK(hrx_gpu_device_count(&count))) {
        return context;
    }
    context->device_contexts.reserve(count);
    context->devices.reserve(count);
    for (int i = 0; i < count; ++i) {
        hrx_device_t hrx_device = nullptr;
        if (!HRX_CHECK(hrx_gpu_device_get(i, &hrx_device)) || hrx_device == nullptr) {
            continue;
        }
        hrx_device_retain(hrx_device);
        auto device_ctx = std::make_unique<ggml_backend_hrx_device_context>();
        device_ctx->device = hrx_device;
        device_ctx->name = "HRX" + std::to_string(i);
        const std::optional<std::string> name = device_string_property(
            hrx_device, HRX_DEVICE_PROPERTY_NAME, "query HRX device name");
        const std::optional<std::string> architecture = device_string_property(
            hrx_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, "query HRX device architecture");
        if (!name || !architecture) {
            hrx_device_release(hrx_device);
            continue;
        }
        uint64_t memory = 0;
        if (!HRX_CHECK(hrx_device_get_property(
                hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory, sizeof(memory)))) {
            hrx_device_release(hrx_device);
            continue;
        }
        device_ctx->memory_total = static_cast<size_t>(memory);
        device_ctx->description = *name + " (" + *architecture + ")";
        device_ctx->architecture = *architecture;
        device_ctx->transfers = std::make_unique<ggml::hrx::TransferManager>(hrx_device);
        if (!device_ctx->transfers->valid()) {
            GGML_LOG_ERROR("%s: transfer manager initialization failed: %s\n", __func__,
                device_ctx->transfers->initialization_error().c_str());
            hrx_device_release(hrx_device);
            continue;
        }
        device_ctx->buft_context = { device_ctx.get(), device_ctx->name };
        device_ctx->buft = { buffer_type_i, nullptr, &device_ctx->buft_context };
        context->device_contexts.emplace_back(std::move(device_ctx));
        context->devices.push_back({ device_i, nullptr, context->device_contexts.back().get() });
        context->device_contexts.back()->buft.device = &context->devices.back();
    }
    return context;
}

} // namespace

ggml_backend_reg_t ggml_backend_hrx_reg() {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context = create_registry_context();
    static ggml_backend_reg registry = { GGML_BACKEND_API_VERSION, registry_i, context.get() };
    for (auto & device : context->devices) {
        device.reg = &registry;
    }
    return &registry;
}

ggml_backend_t ggml_backend_hrx_init(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(registry)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, device), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count() {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_hrx_reg()));
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    return device < ggml_backend_reg_dev_count(registry) ? ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(registry, device)) : nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
