#include "executable-program.h"

#include "kernel-corpus.h"
#include "loom-jit.h"
#include "nlohmann/json.hpp"
#include "transfer-manager.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {

struct artifact_record {
    ~artifact_record() {
        if (executable != nullptr) {
            hrx_executable_release(executable);
        }
    }

    hrx_executable_t                executable     = nullptr;
    uint32_t                        export_ordinal = 0;
    hrx_executable_export_info_t    export_info    = {};
    ggml_hrx_loom_jit_launch_config launch;
    prepared_artifact_diagnostic    diagnostic;
};

struct executable_artifact_repository::impl {
    std::mutex                                                        mutex;
    std::unordered_map<std::string, std::shared_ptr<artifact_record>> artifacts;
};

executable_artifact_repository::executable_artifact_repository() : impl_(new impl()) {}

executable_artifact_repository::~executable_artifact_repository() = default;

namespace {

constexpr size_t direct_host_input_max_bytes = 512;

ggml_hrx_loom_jit_source_format to_jit_source_format(kernel_source_format format) {
    switch (format) {
        case KERNEL_SOURCE_FORMAT_TEXT:
            return ggml_hrx_loom_jit_source_format::GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
        case KERNEL_SOURCE_FORMAT_BINARY:
            return ggml_hrx_loom_jit_source_format::GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE;
    }
    return ggml_hrx_loom_jit_source_format::GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
}

const TransientAllocation * find_transient(const CommandProgram & commands, uint32_t id) {
    const auto it = std::find_if(commands.transients.allocations.begin(), commands.transients.allocations.end(),
                                 [&](const TransientAllocation & allocation) { return allocation.id == id; });
    return it == commands.transients.allocations.end() ? nullptr : &*it;
}

const PersistentConstantAllocation * find_persistent_constant(const CommandProgram & commands, uint32_t id) {
    const auto it =
        std::find_if(commands.persistent_constants.allocations.begin(), commands.persistent_constants.allocations.end(),
                     [&](const PersistentConstantAllocation & allocation) { return allocation.id == id; });
    return it == commands.persistent_constants.allocations.end() ? nullptr : &*it;
}

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::string join_key(const std::map<std::string, std::string> & values) {
    std::ostringstream out;
    for (const auto & value : values) {
        out << '|' << value.first << '=' << value.second;
    }
    return out.str();
}

}  // namespace

packed_kernel_constants pack_kernel_constants(const kernel_definition & definition, const Command & command) {
    packed_kernel_constants result;
    for (const kernel_scalar_definition & parameter : definition.launch_parameters) {
        const auto value = command.kernel.integer_parameters.find(parameter.name != nullptr ? parameter.name : "");
        if (value == command.kernel.integer_parameters.end()) {
            result.errors.push_back("missing launch scalar " +
                                    std::string(parameter.name != nullptr ? parameter.name : ""));
            continue;
        }
        const char * type = parameter.type != nullptr ? parameter.type : "";
        if (std::strcmp(type, "f32") == 0) {
            if (value->second < 0 || static_cast<uint64_t>(value->second) > std::numeric_limits<uint32_t>::max()) {
                result.errors.push_back("launch scalar " +
                                        std::string(parameter.name != nullptr ? parameter.name : "") +
                                        " does not contain an f32 ABI payload");
                continue;
            }
            append_u32(result.bytes, static_cast<uint32_t>(value->second));
            continue;
        }
        if (std::strcmp(type, "index") != 0) {
            result.errors.push_back("unsupported launch scalar type " + std::string(type) + " for " +
                                    std::string(parameter.name != nullptr ? parameter.name : ""));
            continue;
        }
        if (value->second < 0 || static_cast<uint64_t>(value->second) > std::numeric_limits<uint32_t>::max()) {
            result.errors.push_back("launch scalar " + std::string(parameter.name != nullptr ? parameter.name : "") +
                                    " does not fit the index ABI");
            continue;
        }
        append_u32(result.bytes, static_cast<uint32_t>(value->second));
    }
    if (!result.errors.empty()) {
        result.bytes.clear();
    }
    return result;
}

std::string kernel_artifact_key(const kernel_definition & definition,
                                const Command &           command,
                                const std::string &       target) {
    std::ostringstream out;
    out << target << '|' << definition.family << '/' << definition.name << '#' << definition.id << '|'
        << definition.source << '|' << definition.symbol << "|recipe=" << definition.compile_recipe.mode;
    if (definition.compile_recipe.link_module != nullptr && definition.compile_recipe.link_module[0] != 0) {
        out << ':' << definition.compile_recipe.link_module;
    }
    for (const kernel_scalar_definition & parameter : definition.workload_parameters) {
        const auto value = command.kernel.integer_parameters.find(parameter.name != nullptr ? parameter.name : "");
        out << '|' << parameter.name << '=';
        if (value == command.kernel.integer_parameters.end()) {
            out << "<missing>";
        } else {
            out << value->second;
        }
    }
    out << join_key(command.kernel.compile_parameters);
    return out.str();
}

static const char * binding_class_name(const executable_buffer_binding & binding) {
    if (binding.buffer != nullptr) {
        if (binding.weight) {
            return "borrowed_device_weight";
        }
        if (binding.mutable_state) {
            return "device_mutable_state";
        }
        return "device";
    }
    if (binding.weight) {
        return "resident_host_weight";
    }
    if (binding.upload_before_launch && binding.download_after_completion) {
        return "host_input_output";
    }
    if (binding.upload_before_launch) {
        return "host_input";
    }
    if (binding.download_after_completion) {
        return "host_output";
    }
    return "host";
}

struct prepared_executable_program::impl {
    struct host_staging_binding {
        StorageId    storage       = kInvalidId;
        hrx_buffer_t buffer        = nullptr;
        void *       host_data     = nullptr;
        void *       mapped_data   = nullptr;
        size_t       length        = 0;
        bool         upload        = false;
        bool         initialize    = false;
        bool         download      = false;
        bool         direct_mapped = false;
    };

    struct debug_binding {
        std::string      name;
        ResourceAccess   access = ResourceAccess::Read;
        hrx_buffer_ref_t ref    = {};
    };

    ~impl() {
        if (graph_exec != nullptr) {
            hrx_graph_exec_release(graph_exec);
        }
        if (graph != nullptr) {
            hrx_graph_release(graph);
        }
        if (transient_buffer != nullptr) {
            hrx_buffer_release(transient_buffer);
        }
        if (persistent_constant_buffer != nullptr) {
            hrx_buffer_release(persistent_constant_buffer);
        }
        for (host_staging_binding & staging : host_staging) {
            if (staging.buffer != nullptr) {
                hrx_buffer_release(staging.buffer);
            }
        }
        for (hrx_buffer_t buffer : retained_buffers) {
            hrx_buffer_release(buffer);
        }
    }

    hrx_graph_t                                   graph                      = nullptr;
    hrx_graph_exec_t                              graph_exec                 = nullptr;
    hrx_buffer_t                                  transient_buffer           = nullptr;
    hrx_buffer_t                                  persistent_constant_buffer = nullptr;
    hrx_device_t                                  device                     = nullptr;
    transfer_manager *                            transfers                  = nullptr;
    std::vector<host_staging_binding>             host_staging;
    std::vector<weight_residency_lease>           resident_weights;
    std::vector<hrx_buffer_t>                     retained_buffers;
    std::vector<std::shared_ptr<artifact_record>> retained_artifacts;
    std::vector<debug_binding>                    last_command_outputs;
    bool                                          launch_in_flight = false;
};

prepared_executable_program::prepared_executable_program() : impl_(new impl()) {}

prepared_executable_program::~prepared_executable_program()                                                   = default;
prepared_executable_program::prepared_executable_program(prepared_executable_program &&) noexcept             = default;
prepared_executable_program & prepared_executable_program::operator=(prepared_executable_program &&) noexcept = default;

error_result prepared_executable_program::rebind(const executable_bindings &   bindings,
                                                 const AllocationFingerprint & fingerprint) {
    if (!valid()) {
        return "cannot rebind an invalid prepared executable";
    }
    if (fingerprint != allocation_fingerprint_) {
        return "live allocation fingerprint does not match prepared executable";
    }
    for (impl::host_staging_binding & staging : impl_->host_staging) {
        const auto found =
            std::find_if(bindings.storages.begin(), bindings.storages.end(),
                         [&](const executable_buffer_binding & binding) { return binding.storage == staging.storage; });
        if (found == bindings.storages.end() || found->buffer != nullptr || found->host_data == nullptr ||
            found->offset > found->capacity || found->length > found->capacity - found->offset ||
            found->length != staging.length || found->upload_before_launch != staging.upload ||
            found->download_after_completion != staging.download) {
            return "live host binding does not match prepared storage " + std::to_string(staging.storage);
        }
        staging.host_data = static_cast<uint8_t *>(found->host_data) + found->offset;
    }
    return {};
}

error_result prepared_executable_program::launch(hrx_stream_t stream) {
    if (!valid() || stream == nullptr) {
        return "cannot launch an invalid prepared executable";
    }
    if (impl_->transfers == nullptr) {
        return "prepared executable has no transfer manager";
    }
    const bool has_direct_uploads =
        std::any_of(impl_->host_staging.begin(), impl_->host_staging.end(),
                    [](const impl::host_staging_binding & staging) { return staging.upload && staging.direct_mapped; });
    if (impl_->launch_in_flight && has_direct_uploads) {
        if (error_result synchronize_error = take_status(hrx_stream_synchronize(stream))) {
            return "synchronize reusable direct host bindings: " + *synchronize_error;
        }
        if (error_result completion_error = complete_after_synchronize()) {
            return "complete reusable direct host bindings: " + *completion_error;
        }
    }
    const bool has_transfer_uploads = std::any_of(
        impl_->host_staging.begin(), impl_->host_staging.end(),
        [](const impl::host_staging_binding & staging) { return staging.upload && !staging.direct_mapped; });
    if (impl_->launch_in_flight && has_transfer_uploads) {
        std::string error = impl_->transfers->wait_for_producer(stream);
        if (!error.empty()) {
            return "order reusable launch bindings: " + error;
        }
    }
    for (const impl::host_staging_binding & staging : impl_->host_staging) {
        if (!staging.upload) {
            continue;
        }
        if (staging.direct_mapped) {
            std::memcpy(staging.mapped_data, staging.host_data, staging.length);
            continue;
        }
        std::string error = impl_->transfers->upload(staging.host_data, staging.buffer, 0, staging.length);
        if (!error.empty()) {
            return "upload host staging: " + error;
        }
    }
    std::string error = impl_->transfers->join(stream);
    if (!error.empty()) {
        return "join executable uploads: " + error;
    }
    if (error_result launch_error = take_status(hrx_graph_exec_launch(impl_->graph_exec, stream))) {
        return launch_error;
    }
    impl_->launch_in_flight = true;
    return {};
}

error_result prepared_executable_program::complete_after_synchronize() {
    if (!valid()) {
        return "cannot complete an invalid prepared executable";
    }
    if (impl_->transfers == nullptr) {
        return "prepared executable has no transfer manager";
    }
    for (const impl::host_staging_binding & staging : impl_->host_staging) {
        if (!staging.download) {
            continue;
        }
        std::string error = impl_->transfers->download(nullptr, staging.buffer, 0, staging.host_data, staging.length);
        if (!error.empty()) {
            return "download host staging: " + error;
        }
    }
    impl_->launch_in_flight = false;
    return {};
}

error_result prepared_executable_program::snapshot_transients(std::vector<uint8_t> & bytes) {
    bytes.clear();
    if (!valid()) {
        return "cannot snapshot an invalid prepared executable";
    }
    if (impl_->transfers == nullptr) {
        return "prepared executable has no transfer manager";
    }
    if (transient_bytes_ == 0) {
        return {};
    }
    if (impl_->transient_buffer == nullptr) {
        return "prepared executable has no transient allocation";
    }
    bytes.resize(transient_bytes_);
    const std::string error =
        impl_->transfers->download(nullptr, impl_->transient_buffer, 0, bytes.data(), bytes.size());
    if (!error.empty()) {
        bytes.clear();
        return "snapshot transient arena: " + error;
    }
    return {};
}

error_result prepared_executable_program::snapshot_last_command_outputs(
    std::vector<prepared_binding_snapshot> & snapshots,
    size_t                                   maximum_binding_bytes) {
    snapshots.clear();
    if (!valid()) {
        return "cannot snapshot an invalid prepared executable";
    }
    if (impl_->transfers == nullptr) {
        return "prepared executable has no transfer manager";
    }
    for (const impl::debug_binding & binding : impl_->last_command_outputs) {
        prepared_binding_snapshot snapshot;
        snapshot.name   = binding.name;
        snapshot.access = binding.access;
        snapshot.length = binding.ref.length;
        if (binding.ref.length > maximum_binding_bytes) {
            snapshots.push_back(std::move(snapshot));
            continue;
        }
        snapshot.bytes.resize(binding.ref.length);
        const std::string error = impl_->transfers->download(nullptr, binding.ref.buffer, binding.ref.offset,
                                                             snapshot.bytes.data(), snapshot.bytes.size());
        if (!error.empty()) {
            snapshots.clear();
            return "snapshot command output " + binding.name + ": " + error;
        }
        snapshots.push_back(std::move(snapshot));
    }
    return {};
}

void prepared_executable_program::abandon_after_synchronize() {
    if (impl_ != nullptr) {
        impl_->launch_in_flight = false;
    }
}

class executable_program_preparer {
  public:
    executable_program_preparer(hrx_device_t                           device,
                                hrx_stream_t                           stream,
                                transfer_manager &                     transfers,
                                weight_residency_cache &               weights,
                                executable_artifact_repository &       artifact_repository,
                                const ProgramPlan &                    plan,
                                const kernel_corpus &                  corpus,
                                const CommandProgram &                 commands,
                                const executable_bindings &            bindings,
                                const executable_preparation_options & options) :
        result(),
        impl(*result.impl_),
        device(device),
        stream(stream),
        transfers(transfers),
        weights(weights),
        artifact_repository(artifact_repository),
        plan(plan),
        corpus(corpus),
        commands(commands),
        bindings(bindings),
        options(options),
        record_command_count(std::min(options.command_limit, commands.commands.size())),
        command_artifacts(commands.commands.size()),
        command_constants(commands.commands.size()) {}

    prepared_executable_program run();

  private:
    bool validate_and_initialize();
    bool allocate_program_buffers();
    bool compile_artifacts();
    bool bind_storage();
    bool record_graph();
    bool resolve_binding(const CommandBinding & binding, hrx_buffer_ref_t & result_ref);

    prepared_executable_program                     result;
    prepared_executable_program::impl &             impl;
    hrx_device_t                                    device;
    hrx_stream_t                                    stream;
    transfer_manager &                              transfers;
    weight_residency_cache &                        weights;
    executable_artifact_repository &                artifact_repository;
    const ProgramPlan &                             plan;
    const kernel_corpus &                           corpus;
    const CommandProgram &                          commands;
    const executable_bindings &                     bindings;
    const executable_preparation_options &          options;
    size_t                                          record_command_count;
    error_result                                    error;
    std::vector<std::shared_ptr<artifact_record>>   command_artifacts;
    std::vector<std::vector<uint8_t>>               command_constants;
    std::unordered_map<StorageId, hrx_buffer_ref_t> storage_refs;
};

bool executable_program_preparer::validate_and_initialize() {
    if (device == nullptr || stream == nullptr) {
        result.errors_.push_back("device and stream are required");
    }
    if (!transfers.valid()) {
        result.errors_.push_back("valid transfer manager is required: " + transfers.initialization_error());
    }
    if (!weights.valid()) {
        result.errors_.push_back("valid weight residency cache is required: " + weights.initialization_error());
    }
    if (!commands.valid()) {
        result.errors_.insert(result.errors_.end(), commands.errors.begin(), commands.errors.end());
    }
    if (commands.target != options.target) {
        result.errors_.push_back("preparation target does not match command program");
    }
    const VerificationResult verification = verify_command_program(plan, corpus, commands);
    result.errors_.insert(result.errors_.end(), verification.errors.begin(), verification.errors.end());
    const VerificationResult binding_verification = verify_binding_snapshot(plan, bindings.snapshot);
    result.errors_.insert(result.errors_.end(), binding_verification.errors.begin(), binding_verification.errors.end());
    if (bindings.storages.size() != bindings.snapshot.bindings.size()) {
        result.errors_.push_back("executable binding table does not match binding snapshot");
    }
    if (!result.errors_.empty()) {
        return false;
    }

    impl.device                    = device;
    impl.transfers                 = &transfers;
    result.source_command_count_   = commands.commands.size();
    result.command_prefix_         = record_command_count != commands.commands.size();
    result.split_commands_         = options.split_commands;
    result.serialized_commands_    = options.serialize_commands || options.split_commands;
    result.allocation_fingerprint_ = fingerprint_bindings(bindings.snapshot);
    return true;
}

bool executable_program_preparer::allocate_program_buffers() {
    if (commands.transients.arena_size != 0) {
        error = take_status(hrx_buffer_allocate(stream, commands.transients.arena_size, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                                HRX_BUFFER_USAGE_DEFAULT, &impl.transient_buffer));
        if (error) {
            result.errors_.push_back("allocate transient arena: " + *error);
            return false;
        }
    }
    result.transient_bytes_ = commands.transients.arena_size;
    if (commands.persistent_constants.arena_size != 0) {
        error = take_status(hrx_buffer_allocate(stream, commands.persistent_constants.arena_size,
                                                HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT,
                                                &impl.persistent_constant_buffer));
        if (error) {
            result.errors_.push_back("allocate persistent constant arena: " + *error);
            return false;
        }
    }
    result.persistent_constant_bytes_ = commands.persistent_constants.arena_size;
    for (const ConstantInitialization & initialization : commands.initializations) {
        const auto allocation = std::find_if(
            commands.persistent_constants.allocations.begin(), commands.persistent_constants.allocations.end(),
            [&](const PersistentConstantAllocation & item) { return item.storage == initialization.storage; });
        if (allocation == commands.persistent_constants.allocations.end() ||
            impl.persistent_constant_buffer == nullptr) {
            result.errors_.push_back("constant initialization does not resolve to persistent storage");
            return false;
        }
        const std::string upload_error = transfers.upload(initialization.data.data(), impl.persistent_constant_buffer,
                                                          allocation->arena_offset, initialization.data.size());
        if (!upload_error.empty()) {
            result.errors_.push_back("upload " + initialization.label + ": " + upload_error);
            return false;
        }
    }
    return true;
}

bool executable_program_preparer::compile_artifacts() {
    ggml_hrx_loom_jit_amdgpu_options jit_options;
    jit_options.processor  = options.target.c_str();
    jit_options.identifier = options.target.c_str();
    jit_options.sanitizer  = options.sanitizer.empty() ? nullptr : options.sanitizer.c_str();
    jit_options.sanitizer_reporting =
        options.sanitizer_reporting.empty() ? nullptr : options.sanitizer_reporting.c_str();
    ggml_hrx_loom_jit_amdgpu * jit = nullptr;
    error                          = take_status(ggml_hrx_loom_jit_amdgpu_create(&jit_options, &jit));
    if (error) {
        result.errors_.push_back("create Loom JIT: " + *error);
        return false;
    }
    // Serialize compilation and publication per device; steady execution only queries retained artifacts.
    std::unique_lock<std::mutex> artifact_lock(artifact_repository.impl_->mutex);
    auto &                       artifact_cache = artifact_repository.impl_->artifacts;
    std::set<std::string>        program_artifact_keys;
    for (const Command & command : commands.commands) {
        if (command.ordinal >= record_command_count) {
            break;
        }
        if (command.kind != CommandKind::Kernel) {
            continue;
        }
        const kernel_resolve_result resolved   = resolve_kernel_definition(corpus, options.target, command.kernel);
        const kernel_definition *   definition = resolved.definition;
        if (!resolved.found()) {
            result.errors_.push_back(format_kernel_resolve_error(resolved, command.kernel));
            break;
        }
        const packed_kernel_constants constants = pack_kernel_constants(*definition, command);
        if (!constants.valid()) {
            for (const std::string & item : constants.errors) {
                result.errors_.push_back(kernel_specialization_name(command.kernel) + ": " + item);
            }
            break;
        }
        command_constants[command.ordinal] = constants.bytes;
        std::string key                    = kernel_artifact_key(*definition, command, options.target);
        if (!options.sanitizer.empty()) {
            key += "|sanitizer=" + options.sanitizer;
        }
        if (!options.sanitizer_reporting.empty()) {
            key += "|sanitizer_reporting=" + options.sanitizer_reporting;
        }
        auto found = artifact_cache.find(key);
        if (found != artifact_cache.end()) {
            command_artifacts[command.ordinal] = found->second;
            if (program_artifact_keys.insert(key).second) {
                impl.retained_artifacts.push_back(found->second);
                result.artifacts_.push_back(found->second->diagnostic);
            }
            continue;
        }

        if (definition->compile_recipe.primary_sources.empty()) {
            result.errors_.push_back("kernel " + kernel_specialization_name(command.kernel) +
                                     " has no embedded primary source");
            break;
        }
        const kernel_source_ref & primary_source  = definition->compile_recipe.primary_sources.front();
        const kernel_source *     source_contents = primary_source.contents;
        if (source_contents == nullptr) {
            result.errors_.push_back("missing embedded Loom source " +
                                     std::string(primary_source.path != nullptr ? primary_source.path : ""));
            break;
        }
        if (source_contents->dependency_count != definition->compile_recipe.library_sources.size()) {
            result.errors_.push_back("embedded Loom source dependency count does not match " +
                                     std::string(primary_source.path != nullptr ? primary_source.path : ""));
            break;
        }
        std::vector<ggml_hrx_loom_jit_source> dependencies;
        dependencies.reserve(definition->compile_recipe.library_sources.size());
        for (size_t dependency_index = 0; dependency_index < definition->compile_recipe.library_sources.size();
             ++dependency_index) {
            const kernel_source_ref & dependency_ref = definition->compile_recipe.library_sources[dependency_index];
            const kernel_source *     dependency_contents = dependency_ref.contents;
            if (dependency_contents == nullptr) {
                result.errors_.push_back("missing embedded Loom dependency " +
                                         std::string(dependency_ref.path != nullptr ? dependency_ref.path : ""));
                break;
            }
            const kernel_source_span & dependency = dependency_contents->source;
            dependencies.push_back(
                { dependency.data, dependency.length, to_jit_source_format(dependency.format), dependency_ref.path });
        }
        if (!result.errors_.empty()) {
            break;
        }
        std::vector<std::string>                      config_keys;
        std::vector<std::string>                      config_values;
        std::vector<ggml_hrx_loom_jit_config_binding> configs;
        for (const auto & item : command.kernel.compile_parameters) {
            config_keys.push_back(item.first);
            config_values.push_back(item.second);
        }
        for (size_t i = 0; i < config_keys.size(); ++i) {
            configs.push_back({ config_keys[i].c_str(), config_values[i].c_str() });
        }
        std::vector<int64_t> workload;
        for (const kernel_scalar_definition & parameter : definition->workload_parameters) {
            const auto value = command.kernel.integer_parameters.find(parameter.name != nullptr ? parameter.name : "");
            if (value == command.kernel.integer_parameters.end() ||
                std::strcmp(parameter.type != nullptr ? parameter.type : "", "index") != 0) {
                result.errors_.push_back("invalid workload ABI for " + kernel_specialization_name(command.kernel) +
                                         " parameter " + std::string(parameter.name != nullptr ? parameter.name : ""));
                break;
            }
            workload.push_back(value->second);
        }
        if (!result.errors_.empty()) {
            break;
        }
        ggml_hrx_loom_jit_compile_options compile_options = {};
        compile_options.source_data                       = source_contents->source.data;
        compile_options.source_size                       = source_contents->source.length;
        compile_options.source_format                     = to_jit_source_format(source_contents->source.format);
        compile_options.source_identifier                 = primary_source.path;
        compile_options.root_symbol                       = definition->symbol;
        compile_options.module_name                       = definition->symbol;
        compile_options.artifact_identifier               = definition->symbol;
        compile_options.dependencies                      = dependencies.data();
        compile_options.dependency_count                  = dependencies.size();
        compile_options.config_bindings                   = configs.data();
        compile_options.config_binding_count              = configs.size();
        compile_options.workload_arguments                = workload.data();
        compile_options.workload_argument_count           = workload.size();
        compile_options.evaluate_launch_config            = true;
        ggml_hrx_loom_jit_compile_result compiled;
        error = take_status(ggml_hrx_loom_jit_amdgpu_compile(jit, &compile_options, &compiled));
        if (error) {
            result.errors_.push_back("compile " + key + ": " + *error);
            break;
        }

        auto compiled_artifact    = std::make_shared<artifact_record>();
        compiled_artifact->launch = compiled.launch_config;
        if (compiled.manifest_json != nullptr) {
            compiled_artifact->diagnostic.manifest_json.assign(compiled.manifest_json, compiled.manifest_json_size);
        }
        if (compiled.compile_report_json != nullptr) {
            compiled_artifact->diagnostic.compile_report_json.assign(compiled.compile_report_json,
                                                                     compiled.compile_report_json_size);
        }
        if (compiled.final_module_text != nullptr) {
            compiled_artifact->diagnostic.final_module_text.assign(compiled.final_module_text,
                                                                   compiled.final_module_text_size);
        }
        error = take_status(hrx_executable_load_data(device, compiled.hsaco_data, compiled.hsaco_size, "amdgpu",
                                                     options.target.c_str(), &compiled_artifact->executable));
        if (error) {
            result.errors_.push_back("load " + key + ": " + *error);
            break;
        }
        error = take_status(hrx_executable_lookup_export_by_name(compiled_artifact->executable, definition->symbol,
                                                                 &compiled_artifact->export_ordinal));
        if (!error) {
            error = take_status(hrx_executable_export_info(
                compiled_artifact->executable, compiled_artifact->export_ordinal, &compiled_artifact->export_info));
        }
        if (error) {
            result.errors_.push_back("inspect " + key + ": " + *error);
            break;
        }
        if (compiled_artifact->export_info.binding_count != definition->bindings.size() ||
            compiled_artifact->export_info.constant_byte_length != constants.bytes.size() ||
            compiled_artifact->export_info.parameter_count !=
                definition->bindings.size() + definition->launch_parameters.size()) {
            result.errors_.push_back(
                "compiled ABI does not match manifest for " + key +
                ": bindings=" + std::to_string(compiled_artifact->export_info.binding_count) + "/" +
                std::to_string(definition->bindings.size()) +
                " constants=" + std::to_string(compiled_artifact->export_info.constant_byte_length) + "/" +
                std::to_string(constants.bytes.size()) +
                " parameters=" + std::to_string(compiled_artifact->export_info.parameter_count) + "/" +
                std::to_string(definition->bindings.size() + definition->launch_parameters.size()));
            break;
        }
        if (compiled_artifact->launch.workgroup_count[0] == 0 || compiled_artifact->launch.workgroup_count[1] == 0 ||
            compiled_artifact->launch.workgroup_count[2] == 0 || compiled_artifact->launch.workgroup_size[0] == 0 ||
            compiled_artifact->launch.workgroup_size[1] == 0 || compiled_artifact->launch.workgroup_size[2] == 0) {
            result.errors_.push_back("compiled launch geometry is empty for " + key);
            break;
        }
        if (!std::equal(compiled_artifact->launch.workgroup_size.begin(),
                        compiled_artifact->launch.workgroup_size.end(),
                        std::begin(compiled_artifact->export_info.workgroup_size))) {
            result.errors_.push_back("compiled launch workgroup size does not match executable metadata for " + key);
            break;
        }
        // launch.workgroup_storage_bytes is the kernel's total static
        // requirement; the code object from the same compile already carries
        // it, so dispatch needs no extra storage argument.
        compiled_artifact->diagnostic.key       = key;
        compiled_artifact->diagnostic.kernel_id = definition->name != nullptr ? definition->name : "";
        for (size_t i = 0; i < 3; ++i) {
            compiled_artifact->diagnostic.workgroup_count[i] = compiled_artifact->launch.workgroup_count[i];
            compiled_artifact->diagnostic.workgroup_size[i]  = compiled_artifact->launch.workgroup_size[i];
        }
        compiled_artifact->diagnostic.subgroup_size  = compiled_artifact->launch.subgroup_size;
        compiled_artifact->diagnostic.constant_bytes = constants.bytes.size();
        compiled_artifact->diagnostic.binding_count  = definition->bindings.size();
        artifact_cache.emplace(key, compiled_artifact);
        if (program_artifact_keys.insert(key).second) {
            impl.retained_artifacts.push_back(compiled_artifact);
            result.artifacts_.push_back(compiled_artifact->diagnostic);
        }
        command_artifacts[command.ordinal] = std::move(compiled_artifact);
    }
    ggml_hrx_loom_jit_amdgpu_release(jit);
    artifact_lock.unlock();
    return result.errors_.empty();
}

bool executable_program_preparer::bind_storage() {
    std::set<hrx_buffer_t> retained;
    for (const executable_buffer_binding & binding : bindings.storages) {
        const auto snapshot_it =
            std::find_if(bindings.snapshot.bindings.begin(), bindings.snapshot.bindings.end(),
                         [&](const ConcreteBinding & item) { return item.storage == binding.storage; });
        if (snapshot_it == bindings.snapshot.bindings.end() ||
            snapshot_it->buffer_identity != binding.buffer_identity || snapshot_it->generation != binding.generation ||
            snapshot_it->capacity != binding.capacity || snapshot_it->offset != binding.offset ||
            snapshot_it->length != binding.length || binding.length == 0 || binding.offset > binding.capacity ||
            binding.length > binding.capacity - binding.offset ||
            !storage_refs.emplace(binding.storage, hrx_buffer_ref_t{}).second) {
            result.errors_.push_back("concrete binding table disagrees with snapshot for storage " +
                                     std::to_string(binding.storage));
            return false;
        }
        hrx_buffer_t             concrete_buffer = binding.buffer;
        size_t                   concrete_offset = binding.offset;
        const ResourceContract & resource        = plan.resources.resources[binding.storage];
        if (concrete_buffer == nullptr) {
            if (binding.host_data == nullptr) {
                result.errors_.push_back("storage " + std::to_string(binding.storage) +
                                         " has no device or host allocation");
                return false;
            }
            if (resource.weight) {
                weight_source source;
                source.host_data                 = binding.host_data;
                source.buffer_identity           = binding.buffer_identity;
                source.generation                = binding.generation;
                source.capacity                  = binding.capacity;
                source.offset                    = binding.offset;
                source.length                    = binding.length;
                source.layout                    = binding.layout;
                source.transform                 = binding.transform;
                weight_residency_result resident = weights.acquire(stream, transfers, source);
                if (!resident.valid()) {
                    result.errors_.push_back("materialize host weight storage " + std::to_string(binding.storage) +
                                             ": " + *resident.error);
                    return false;
                }
                concrete_buffer = resident.lease.buffer();
                concrete_offset = 0;
                result.resident_host_weight_bytes_ += binding.length;
                impl.resident_weights.push_back(std::move(resident.lease));
            } else {
                prepared_executable_program::impl::host_staging_binding staging;
                staging.storage       = binding.storage;
                staging.host_data     = static_cast<uint8_t *>(binding.host_data) + binding.offset;
                staging.length        = binding.length;
                staging.upload        = binding.upload_before_launch;
                staging.initialize    = binding.initialize_from_host;
                staging.download      = binding.download_after_completion;
                staging.direct_mapped = staging.upload && !staging.initialize && !staging.download &&
                                        staging.length <= direct_host_input_max_bytes;
                if (staging.direct_mapped) {
                    const hrx_buffer_params_t params = {
                        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                        HRX_MEMORY_ACCESS_ALL,
                        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED |
                            HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                        0,
                    };
                    error = take_status(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params,
                                                                      binding.length, &staging.buffer));
                    if (!error) {
                        error = take_status(
                            hrx_buffer_map(staging.buffer, HRX_MAP_WRITE, 0, staging.length, &staging.mapped_data));
                    }
                } else {
                    error = take_status(hrx_buffer_allocate(stream, binding.length, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                                            HRX_BUFFER_USAGE_DEFAULT, &staging.buffer));
                }
                if (error) {
                    if (staging.buffer != nullptr) {
                        hrx_buffer_release(staging.buffer);
                    }
                    result.errors_.push_back("allocate host staging for storage " + std::to_string(binding.storage) +
                                             ": " + *error);
                    return false;
                }
                concrete_buffer = staging.buffer;
                concrete_offset = 0;
                result.host_staging_bytes_ += binding.length;
                result.retained_bytes_ += binding.length;
                impl.host_staging.push_back(staging);
            }
        } else if (retained.insert(concrete_buffer).second) {
            hrx_buffer_retain(concrete_buffer);
            impl.retained_buffers.push_back(concrete_buffer);
        }
        if (concrete_buffer != nullptr && binding.buffer != nullptr && resource.weight) {
            result.borrowed_device_weight_bytes_ += binding.length;
        }
        storage_refs[binding.storage] = { concrete_buffer, concrete_offset, binding.length };
    }
    result.retained_bytes_ += commands.transients.arena_size + commands.persistent_constants.arena_size;
    for (const prepared_executable_program::impl::host_staging_binding & staging : impl.host_staging) {
        if (!staging.initialize) {
            continue;
        }
        const std::string upload_error = transfers.upload(staging.host_data, staging.buffer, 0, staging.length);
        if (!upload_error.empty()) {
            result.errors_.push_back("initialize retained host staging: " + upload_error);
            return false;
        }
    }
    const std::string join_error = transfers.join(stream);
    if (!join_error.empty()) {
        result.errors_.push_back("join executable initialization transfers: " + join_error);
        return false;
    }
    return true;
}

bool executable_program_preparer::resolve_binding(const CommandBinding & binding, hrx_buffer_ref_t & result_ref) {
    if (binding.origin == BindingOrigin::Transient) {
        const TransientAllocation * allocation = find_transient(commands, binding.transient);
        if (allocation == nullptr || impl.transient_buffer == nullptr || binding.offset > allocation->size ||
            binding.length > allocation->size - binding.offset) {
            result.errors_.push_back("invalid concrete transient binding");
            return false;
        }
        result_ref = { impl.transient_buffer, allocation->arena_offset + binding.offset, binding.length };
    } else if (binding.origin == BindingOrigin::PersistentConstant) {
        const PersistentConstantAllocation * allocation =
            find_persistent_constant(commands, binding.persistent_constant);
        if (allocation == nullptr || impl.persistent_constant_buffer == nullptr || binding.offset > allocation->size ||
            binding.length > allocation->size - binding.offset) {
            result.errors_.push_back("invalid concrete persistent constant binding");
            return false;
        }
        result_ref = { impl.persistent_constant_buffer, allocation->arena_offset + binding.offset, binding.length };
    } else {
        const auto concrete = storage_refs.find(binding.storage);
        if (concrete == storage_refs.end() || binding.offset > concrete->second.length ||
            binding.length > concrete->second.length - binding.offset) {
            result.errors_.push_back("graph storage " + std::to_string(binding.storage) +
                                     " has no valid concrete range");
            return false;
        }
        result_ref = { concrete->second.buffer, concrete->second.offset + binding.offset, binding.length };
    }
    return true;
}

bool executable_program_preparer::record_graph() {
    error = take_status(hrx_graph_create(device, 0, &impl.graph));
    if (error) {
        result.errors_.push_back("create HRX graph: " + *error);
        return false;
    }

    std::vector<hrx_graph_node_t> nodes(record_command_count, nullptr);
    size_t                        expected_node_count = 0;
    for (const Command & command : commands.commands) {
        if (command.ordinal >= record_command_count) {
            break;
        }
        std::vector<hrx_graph_node_t> deps;
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= nodes.size() || nodes[dependency] == nullptr) {
                result.errors_.push_back("command dependency was not recorded");
                return false;
            }
            deps.push_back(nodes[dependency]);
        }
        // Chain non-recordable partition boundaries so independent commands cannot enter the preceding command buffer.
        if ((options.serialize_commands || options.split_commands) && command.ordinal != 0) {
            const hrx_graph_node_t predecessor = nodes[command.ordinal - 1];
            if (std::find(deps.begin(), deps.end(), predecessor) == deps.end()) {
                deps.push_back(predecessor);
            }
        }
        prepared_command_diagnostic diagnostic;
        diagnostic.ordinal       = command.ordinal;
        diagnostic.kind          = command.kind;
        diagnostic.label         = command.label;
        diagnostic.binding_count = command.bindings.size();
        if (command.kind == CommandKind::Kernel) {
            const std::shared_ptr<artifact_record> & compiled_artifact = command_artifacts[command.ordinal];
            std::vector<hrx_buffer_ref_t>            bindings;
            for (const CommandBinding & binding : command.bindings) {
                hrx_buffer_ref_t concrete = {};
                if (!resolve_binding(binding, concrete)) {
                    return false;
                }
                bindings.push_back(concrete);
                if (command.ordinal + 1 == record_command_count && binding.access != ResourceAccess::Read) {
                    impl.last_command_outputs.push_back({ binding.name, binding.access, concrete });
                }
            }
            const auto &                        constants = command_constants[command.ordinal];
            const hrx_graph_kernel_node_attrs_t attrs     = {
                compiled_artifact->executable,
                compiled_artifact->export_ordinal,
                { { compiled_artifact->launch.workgroup_count[0], compiled_artifact->launch.workgroup_count[1],
                        compiled_artifact->launch.workgroup_count[2] },
                                  {},
                                  compiled_artifact->launch.subgroup_size },
                constants.data(),
                constants.size(),
                bindings.data(),
                bindings.size(),
                0,
            };
            error = take_status(
                hrx_graph_add_kernel_node(impl.graph, deps.data(), deps.size(), &attrs, &nodes[command.ordinal]));
            diagnostic.artifact_key   = compiled_artifact->diagnostic.key;
            diagnostic.constant_bytes = constants.size();
        } else if (command.kind == CommandKind::Copy) {
            hrx_graph_copy_buffer_node_attrs_t attrs = {};
            if (!resolve_binding(command.bindings[0], attrs.src) || !resolve_binding(command.bindings[1], attrs.dst)) {
                return false;
            }
            if (command.ordinal + 1 == record_command_count) {
                impl.last_command_outputs.push_back(
                    { command.bindings[1].name, command.bindings[1].access, attrs.dst });
            }
            error = take_status(
                hrx_graph_add_copy_buffer_node(impl.graph, deps.data(), deps.size(), &attrs, &nodes[command.ordinal]));
        } else if (command.kind == CommandKind::Fill) {
            const uint32_t fill_byte = static_cast<uint32_t>(command.kernel.integer_parameters.at("fill_byte")) & 0xffu;
            hrx_graph_fill_buffer_node_attrs_t attrs = {};
            if (!resolve_binding(command.bindings[0], attrs.dst)) {
                return false;
            }
            if (command.ordinal + 1 == record_command_count) {
                impl.last_command_outputs.push_back(
                    { command.bindings[0].name, command.bindings[0].access, attrs.dst });
            }
            attrs.pattern      = fill_byte;
            attrs.pattern_size = 1;
            error              = take_status(
                hrx_graph_add_fill_buffer_node(impl.graph, deps.data(), deps.size(), &attrs, &nodes[command.ordinal]));
        } else {
            error =
                take_status(hrx_graph_add_empty_node(impl.graph, deps.data(), deps.size(), &nodes[command.ordinal]));
        }
        if (error) {
            result.errors_.push_back("record command " + std::to_string(command.ordinal) + ": " + *error);
            return false;
        }
        ++expected_node_count;
        if (options.split_commands) {
            const hrx_graph_node_t command_node = nodes[command.ordinal];
            error = take_status(hrx_graph_add_empty_node(impl.graph, &command_node, 1, &nodes[command.ordinal]));
            if (error) {
                result.errors_.push_back("record split after command " + std::to_string(command.ordinal) + ": " +
                                         *error);
                return false;
            }
            ++expected_node_count;
        }
        result.commands_.push_back(std::move(diagnostic));
    }
    error = take_status(hrx_graph_size(impl.graph, &result.node_count_));
    if (error) {
        result.errors_.push_back("query HRX graph size: " + *error);
        return false;
    }
    if (result.node_count_ != expected_node_count) {
        result.errors_.push_back("recorded HRX graph node count does not match command program");
        return false;
    }
    error = take_status(hrx_graph_instantiate(impl.graph, 0, &impl.graph_exec));
    if (error) {
        result.errors_.push_back("instantiate HRX graph: " + *error);
    }
    result.prepared_ = result.errors_.empty() && impl.graph_exec != nullptr;
    return result.prepared_;
}

prepared_executable_program executable_program_preparer::run() {
    if (!validate_and_initialize()) {
        return std::move(result);
    }
    if (!allocate_program_buffers()) {
        return std::move(result);
    }
    if (!compile_artifacts()) {
        return std::move(result);
    }
    if (!bind_storage()) {
        return std::move(result);
    }
    record_graph();
    return std::move(result);
}

prepared_executable_program prepare_executable_program(hrx_device_t                           device,
                                                       hrx_stream_t                           stream,
                                                       transfer_manager &                     transfers,
                                                       weight_residency_cache &               weights,
                                                       executable_artifact_repository &       artifact_repository,
                                                       const ProgramPlan &                    plan,
                                                       const kernel_corpus &                  corpus,
                                                       const CommandProgram &                 commands,
                                                       const executable_bindings &            bindings,
                                                       const executable_preparation_options & options) {
    return executable_program_preparer(device, stream, transfers, weights, artifact_repository, plan, corpus, commands,
                                       bindings, options)
        .run();
}

std::string prepared_executable_program::format() const {
    const prepared_executable_program & program = *this;
    std::ostringstream                  out;
    out << "prepared executable program\n"
        << "valid=" << (program.valid() ? "true" : "false") << '\n'
        << "artifacts=" << program.artifact_count() << '\n'
        << "nodes=" << program.node_count() << '\n'
        << "source_commands=" << program.source_command_count() << '\n'
        << "command_prefix=" << (program.command_prefix() ? "true" : "false") << '\n'
        << "split_commands=" << (program.split_commands() ? "true" : "false") << '\n'
        << "serialized_commands=" << (program.serialized_commands() ? "true" : "false") << '\n'
        << "retained_bytes=" << program.retained_bytes() << '\n'
        << "borrowed_device_weight_bytes=" << program.borrowed_device_weight_bytes() << '\n'
        << "resident_host_weight_bytes=" << program.resident_host_weight_bytes() << '\n'
        << "host_staging_bytes=" << program.host_staging_bytes() << '\n'
        << "transient_bytes=" << program.transient_bytes() << '\n'
        << "persistent_constant_bytes=" << program.persistent_constant_bytes() << '\n'
        << "allocation_fingerprint=" << program.allocation_fingerprint().value << '\n';
    for (const prepared_artifact_diagnostic & compiled_artifact : program.artifacts()) {
        out << "artifact " << compiled_artifact.kernel_id << " key=" << compiled_artifact.key
            << " workgroups=" << compiled_artifact.workgroup_count[0] << ',' << compiled_artifact.workgroup_count[1]
            << ',' << compiled_artifact.workgroup_count[2] << " workgroup_size=" << compiled_artifact.workgroup_size[0]
            << ',' << compiled_artifact.workgroup_size[1] << ',' << compiled_artifact.workgroup_size[2]
            << " subgroup=" << compiled_artifact.subgroup_size << " constants=" << compiled_artifact.constant_bytes
            << " bindings=" << compiled_artifact.binding_count << '\n';
    }
    for (const prepared_command_diagnostic & command : program.commands()) {
        out << "command " << command.ordinal << ' ' << command_kind_name(command.kind) << " label=" << command.label
            << " constants=" << command.constant_bytes << " bindings=" << command.binding_count;
        if (!command.artifact_key.empty()) {
            out << " artifact=" << command.artifact_key;
        }
        out << '\n';
    }
    for (const std::string & error : program.errors()) {
        out << "error: " << error << '\n';
    }
    return out.str();
}

std::string prepared_executable_program::serialize_json() const {
    const prepared_executable_program & program = *this;
    nlohmann::json                      root    = {
        { "schema",                       "ggml-hrx-prepared-executable-v1"      },
        { "valid",                        program.valid()                        },
        { "node_count",                   program.node_count()                   },
        { "source_command_count",         program.source_command_count()         },
        { "command_prefix",               program.command_prefix()               },
        { "split_commands",               program.split_commands()               },
        { "serialized_commands",          program.serialized_commands()          },
        { "artifact_count",               program.artifact_count()               },
        { "retained_bytes",               program.retained_bytes()               },
        { "borrowed_device_weight_bytes", program.borrowed_device_weight_bytes() },
        { "resident_host_weight_bytes",   program.resident_host_weight_bytes()   },
        { "host_staging_bytes",           program.host_staging_bytes()           },
        { "transient_bytes",              program.transient_bytes()              },
        { "persistent_constant_bytes",    program.persistent_constant_bytes()    },
        { "allocation_fingerprint",       program.allocation_fingerprint().value },
        { "errors",                       program.errors()                       },
    };
    root["artifacts"] = nlohmann::json::array();
    for (const prepared_artifact_diagnostic & compiled_artifact : program.artifacts()) {
        root["artifacts"].push_back({
            { "key",             compiled_artifact.key             },
            { "kernel",          compiled_artifact.kernel_id       },
            { "workgroup_count", compiled_artifact.workgroup_count },
            { "workgroup_size",  compiled_artifact.workgroup_size  },
            { "subgroup_size",   compiled_artifact.subgroup_size   },
            { "constant_bytes",  compiled_artifact.constant_bytes  },
            { "binding_count",   compiled_artifact.binding_count   },
        });
    }
    root["commands"] = nlohmann::json::array();
    for (const prepared_command_diagnostic & command : program.commands()) {
        root["commands"].push_back({
            { "ordinal",        command.ordinal                 },
            { "kind",           command_kind_name(command.kind) },
            { "label",          command.label                   },
            { "artifact",       command.artifact_key            },
            { "constant_bytes", command.constant_bytes          },
            { "binding_count",  command.binding_count           },
        });
    }
    return root.dump(2);
}

std::string executable_bindings::format() const {
    const executable_bindings & bindings = *this;
    std::ostringstream          out;
    out << "executable bindings count=" << bindings.storages.size() << '\n';
    for (const executable_buffer_binding & binding : bindings.storages) {
        out << "  storage " << binding.storage << " class=" << binding_class_name(binding)
            << " capacity=" << binding.capacity << " range=" << binding.offset << '+' << binding.length;
        if (binding.weight) {
            out << " layout=" << binding.layout;
        }
        out << '\n';
    }
    return out.str();
}

std::string executable_bindings::serialize_json() const {
    const executable_bindings & bindings = *this;
    nlohmann::ordered_json      root     = {
        { "schema",   "ggml-hrx-executable-bindings-v1" },
        { "bindings", nlohmann::ordered_json::array()   },
    };
    for (const executable_buffer_binding & binding : bindings.storages) {
        root["bindings"].push_back({
            { "storage",  binding.storage                                 },
            { "class",    binding_class_name(binding)                     },
            { "capacity", binding.capacity                                },
            { "offset",   binding.offset                                  },
            { "length",   binding.length                                  },
            { "layout",   binding.weight ? binding.layout : std::string() },
        });
    }
    return root.dump(2);
}

}  // namespace ggml::hrx
