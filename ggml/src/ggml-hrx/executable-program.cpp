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

void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

bool same_segment_contract(const StreamedExpertCommand & lhs, const StreamedExpertCommand & rhs) {
    if (lhs.valid() != rhs.valid()) return false;
    if (!lhs.valid()) return true;
    return lhs.layer == rhs.layer && lhs.weight == rhs.weight && lhs.expert_ids == rhs.expert_ids &&
           lhs.missing_suffix == rhs.missing_suffix && lhs.expert_begin == rhs.expert_begin &&
           lhs.expert_end == rhs.expert_end && lhs.load_chunk_size == rhs.load_chunk_size;
}

std::vector<prepared_execution_segment> build_execution_segments(const CommandProgram & commands,
                                                                 size_t command_count) {
    std::vector<prepared_execution_segment> result;
    for (size_t ordinal = 0; ordinal < command_count; ++ordinal) {
        const Command & command = commands.commands[ordinal];
        if (result.empty() || !same_segment_contract(result.back().streamed, command.streamed)) {
            prepared_execution_segment segment;
            segment.command_begin = static_cast<uint32_t>(ordinal);
            segment.command_end   = static_cast<uint32_t>(ordinal + 1);
            segment.streamed      = command.streamed;
            result.push_back(std::move(segment));
        } else {
            result.back().command_end = static_cast<uint32_t>(ordinal + 1);
        }
    }
    return result;
}

std::vector<prepared_graph_segment> build_graph_segments(const CommandProgram & commands, size_t command_count) {
    std::vector<prepared_graph_segment> result;
    size_t command_begin = 0;
    for (size_t ordinal = 0; ordinal < command_count; ++ordinal) {
        const StreamedExpertCommand & streamed = commands.commands[ordinal].streamed;
        const bool begins_missing_suffix = streamed.valid() && streamed.missing_suffix &&
                                           (ordinal == 0 || !same_segment_contract(
                                                               commands.commands[ordinal - 1].streamed, streamed));
        if (!begins_missing_suffix) continue;
        if (command_begin < ordinal) {
            prepared_graph_segment segment;
            segment.command_begin = static_cast<uint32_t>(command_begin);
            segment.command_end   = static_cast<uint32_t>(ordinal);
            segment.wait_after    = streamed;
            result.push_back(std::move(segment));
        }
        command_begin = ordinal;
    }
    if (command_begin < command_count) {
        prepared_graph_segment segment;
        segment.command_begin = static_cast<uint32_t>(command_begin);
        segment.command_end   = static_cast<uint32_t>(command_count);
        result.push_back(std::move(segment));
    }
    return result;
}

bool value_layout_size(const Value & value, size_t & element_count, size_t & compact_byte_length,
                       size_t & span_byte_length) {
    const size_t element_size = ggml_type_size(value.type);
    if (element_size == 0) return false;
    element_count     = 1;
    span_byte_length  = element_size;
    for (size_t dimension = 0; dimension < value.access.shape.size(); ++dimension) {
        const int64_t extent = value.access.shape[dimension];
        if (extent <= 0 || static_cast<uint64_t>(extent) > std::numeric_limits<size_t>::max() / element_count) {
            return false;
        }
        element_count *= static_cast<size_t>(extent);
        const size_t extent_minus_one = static_cast<size_t>(extent - 1);
        if (extent_minus_one != 0 &&
            value.access.strides[dimension] >
                (std::numeric_limits<size_t>::max() - span_byte_length) / extent_minus_one) {
            return false;
        }
        span_byte_length += extent_minus_one * value.access.strides[dimension];
    }
    if (element_count > std::numeric_limits<size_t>::max() / element_size) return false;
    compact_byte_length = element_count * element_size;
    return true;
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
        const std::string type = parameter.type != nullptr ? parameter.type : "";
        if (type == "index") {
            if (value->second < 0 || static_cast<uint64_t>(value->second) > std::numeric_limits<uint32_t>::max()) {
                result.errors.push_back("launch scalar " +
                                        std::string(parameter.name != nullptr ? parameter.name : "") +
                                        " does not fit the index ABI");
                continue;
            }
            append_u32(result.bytes, static_cast<uint32_t>(value->second));
        } else if (type == "i32") {
            if (value->second < std::numeric_limits<int32_t>::min() ||
                value->second > std::numeric_limits<int32_t>::max()) {
                result.errors.push_back("launch scalar " +
                                        std::string(parameter.name != nullptr ? parameter.name : "") +
                                        " does not fit the i32 ABI");
                continue;
            }
            append_u32(result.bytes, static_cast<uint32_t>(static_cast<int32_t>(value->second)));
        } else if (type == "i64") {
            append_u64(result.bytes, static_cast<uint64_t>(value->second));
        } else if (type == "f32") {
            if (value->second < 0 || static_cast<uint64_t>(value->second) > std::numeric_limits<uint32_t>::max()) {
                result.errors.push_back("launch scalar " +
                                        std::string(parameter.name != nullptr ? parameter.name : "") +
                                        " does not contain an f32 bit pattern");
                continue;
            }
            append_u32(result.bytes, static_cast<uint32_t>(value->second));
        } else {
            result.errors.push_back("unsupported launch scalar type " + type + " for " +
                                    std::string(parameter.name != nullptr ? parameter.name : ""));
        }
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

struct streamed_graph_runtime {
    streamed_execution_controller * controller      = nullptr;
    hrx_buffer_t                    readback_buffer  = nullptr;
    size_t                          readback_capacity = 0;
    std::vector<uint32_t>           compact_ids;
    std::string                     error;
    bool                            finished = false;
};

struct streamed_graph_callback {
    enum class kind {
        acquire,
        wait,
        finish,
    };

    kind                       operation = kind::finish;
    streamed_graph_runtime *   runtime   = nullptr;
    prepared_execution_segment segment;
};

static hrx_status_t run_streamed_graph_callback(void * user_data) {
    auto * callback = static_cast<streamed_graph_callback *>(user_data);
    if (callback == nullptr || callback->runtime == nullptr || callback->runtime->controller == nullptr) {
        return hrx_make_status(HRX_STATUS_FAILED_PRECONDITION, "streamed graph callback has no active controller");
    }
    streamed_graph_runtime & runtime = *callback->runtime;
    error_result error;
    if (callback->operation == streamed_graph_callback::kind::acquire) {
        const prepared_execution_segment & segment = callback->segment;
        if (runtime.readback_buffer == nullptr || segment.expert_id_count == 0 ||
            segment.expert_id_span_bytes > runtime.readback_capacity) {
            error = "streamed graph readback contract exceeds its mapped staging buffer";
        } else {
            void * mapped = nullptr;
            error = take_status(hrx_buffer_map(runtime.readback_buffer, HRX_MAP_READ, 0, segment.expert_id_span_bytes,
                                               &mapped));
            if (!error) {
                const auto * readback = static_cast<const uint8_t *>(mapped);
                runtime.compact_ids.resize(segment.expert_id_count);
                size_t output_index = 0;
                for (int64_t i3 = 0; i3 < segment.expert_id_shape[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < segment.expert_id_shape[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < segment.expert_id_shape[1]; ++i1) {
                            for (int64_t i0 = 0; i0 < segment.expert_id_shape[0]; ++i0) {
                                const size_t offset = static_cast<size_t>(i0) * segment.expert_id_strides[0] +
                                                      static_cast<size_t>(i1) * segment.expert_id_strides[1] +
                                                      static_cast<size_t>(i2) * segment.expert_id_strides[2] +
                                                      static_cast<size_t>(i3) * segment.expert_id_strides[3];
                                if (output_index >= runtime.compact_ids.size() ||
                                    offset > segment.expert_id_span_bytes ||
                                    sizeof(uint32_t) > segment.expert_id_span_bytes - offset) {
                                    error = "streamed graph expert-ID view exceeds its readback span";
                                    break;
                                }
                                std::memcpy(&runtime.compact_ids[output_index++], readback + offset, sizeof(uint32_t));
                            }
                            if (error) break;
                        }
                        if (error) break;
                    }
                    if (error) break;
                }
                if (!error && output_index != runtime.compact_ids.size()) {
                    error = "streamed graph expert-ID view did not compact every element";
                }
                const error_result unmap_error = take_status(hrx_buffer_unmap(runtime.readback_buffer));
                if (!error && unmap_error) error = unmap_error;
                if (!error) {
                    error = runtime.controller->acquire(segment.streamed, runtime.compact_ids.data(),
                                                        runtime.compact_ids.size());
                }
            }
        }
    } else if (callback->operation == streamed_graph_callback::kind::wait) {
        error = runtime.controller->wait(callback->segment.streamed);
    } else {
        error = runtime.controller->finish();
        if (!error) runtime.finished = true;
    }
    if (!error) return hrx_ok_status();
    if (runtime.error.empty()) runtime.error = *error;
    return hrx_make_status(HRX_STATUS_INTERNAL, runtime.error.c_str());
}

struct prepared_executable_program::impl {
    struct graph_segment {
        hrx_graph_t      graph      = nullptr;
        hrx_graph_exec_t executable = nullptr;
    };
    struct host_staging_binding {
        StorageId    storage    = kInvalidId;
        hrx_buffer_t buffer     = nullptr;
        void *       host_data  = nullptr;
        size_t       length     = 0;
        bool         upload     = false;
        bool         initialize = false;
        bool         download   = false;
    };

    struct debug_binding {
        std::string      name;
        ResourceAccess   access = ResourceAccess::Read;
        hrx_buffer_ref_t ref    = {};
    };

    ~impl() {
        for (graph_segment & segment : graph_segments) {
            if (segment.executable != nullptr) hrx_graph_exec_release(segment.executable);
            if (segment.graph != nullptr) hrx_graph_release(segment.graph);
        }
        if (streamed_readback_buffer != nullptr) hrx_buffer_release(streamed_readback_buffer);
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
    std::vector<graph_segment>                    graph_segments;
    std::unordered_map<ValueId, hrx_buffer_ref_t> value_refs;
    hrx_buffer_t                                  streamed_readback_buffer = nullptr;
    std::unique_ptr<streamed_graph_runtime>        streamed_runtime;
    std::vector<std::unique_ptr<streamed_graph_callback>> streamed_callbacks;
    bool                                          launch_in_flight = false;
};

prepared_executable_program::prepared_executable_program() : impl_(new impl()) {}

prepared_executable_program::~prepared_executable_program()                                                   = default;
prepared_executable_program::prepared_executable_program(prepared_executable_program &&) noexcept             = default;
prepared_executable_program & prepared_executable_program::operator=(prepared_executable_program &&) noexcept = default;

error_result prepared_executable_program::rebind(const executable_bindings & bindings) {
    if (!valid()) {
        return "cannot rebind an invalid prepared executable";
    }
    if (fingerprint_bindings(bindings.snapshot) != allocation_fingerprint_) {
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

error_result prepared_executable_program::begin_launch(hrx_stream_t stream) {
    if (!valid() || stream == nullptr) {
        return "cannot launch an invalid prepared executable";
    }
    if (impl_->transfers == nullptr) {
        return "prepared executable has no transfer manager";
    }
    const bool has_uploads = std::any_of(impl_->host_staging.begin(), impl_->host_staging.end(),
                                         [](const impl::host_staging_binding & staging) { return staging.upload; });
    if (impl_->launch_in_flight && has_uploads) {
        std::string error = impl_->transfers->wait_for_producer(stream);
        if (!error.empty()) {
            return "order reusable launch bindings: " + error;
        }
    }
    for (const impl::host_staging_binding & staging : impl_->host_staging) {
        if (!staging.upload) {
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
    impl_->launch_in_flight = true;
    return {};
}

error_result prepared_executable_program::launch_segment(size_t segment, hrx_stream_t stream) {
    if (!valid() || stream == nullptr || segment >= graph_segments_.size()) {
        return "cannot launch an invalid executable segment";
    }
    if (!impl_->launch_in_flight) return "executable launch has not begun";
    hrx_graph_exec_t executable = impl_->graph_segments.empty() ? (segment == 0 ? impl_->graph_exec : nullptr) :
                                                                  impl_->graph_segments[segment].executable;
    if (executable == nullptr) return "executable segment has no frozen graph";
    return take_status(hrx_graph_exec_launch(executable, stream));
}

error_result prepared_executable_program::readback_value(hrx_stream_t producer, ValueId value, void * destination,
                                                         size_t size) {
    if (!valid() || impl_->transfers == nullptr || producer == nullptr || destination == nullptr || size == 0) {
        return "invalid executable value readback";
    }
    const auto found = impl_->value_refs.find(value);
    if (found == impl_->value_refs.end() || size > found->second.length) {
        return "executable value has no valid readback range";
    }
    const std::string error =
        impl_->transfers->download(producer, found->second.buffer, found->second.offset, destination, size);
    if (!error.empty()) return "read back executable value: " + error;
    return {};
}

error_result prepared_executable_program::launch(hrx_stream_t stream) {
    if (has_streamed_execution_) return "streamed executable requires route-driven segmented launch";
    if (error_result error = begin_launch(stream)) return error;
    for (size_t segment = 0; segment < segment_count(); ++segment) {
        if (error_result error = launch_segment(segment, stream)) return error;
    }
    return {};
}

error_result prepared_executable_program::launch_streamed(hrx_stream_t stream,
                                                          streamed_execution_controller & controller) {
    if (!valid() || !has_streamed_execution_ || stream == nullptr || impl_->streamed_runtime == nullptr ||
        impl_->graph_segments.size() != graph_segments_.size()) {
        return "cannot launch an invalid streamed executable";
    }
    if (error_result error = begin_launch(stream)) return error;
    streamed_graph_runtime & runtime = *impl_->streamed_runtime;
    runtime.controller = &controller;
    runtime.error.clear();
    runtime.finished = false;
    error_result launch_error;
    for (size_t segment_index = 0; segment_index < graph_segments_.size(); ++segment_index) {
        const prepared_graph_segment & segment  = graph_segments_[segment_index];
        const impl::graph_segment &    recorded = impl_->graph_segments[segment_index];
        if (recorded.executable == nullptr) {
            launch_error = "streamed executable segment has no frozen graph";
            break;
        }
        launch_error = take_status(hrx_graph_exec_launch(recorded.executable, stream));
        if (launch_error) break;
        const StreamedExpertCommand & wait_after = segment.wait_after;
        if (!wait_after.valid()) continue;
        launch_error = take_status(hrx_stream_flush(stream));
        if (launch_error) break;
        launch_error = controller.wait(wait_after);
        if (launch_error) break;
        if (!runtime.error.empty()) {
            launch_error = runtime.error;
            break;
        }
    }
    const error_result synchronize_error = take_status(hrx_stream_synchronize(stream));
    if (!launch_error && synchronize_error) launch_error = synchronize_error;
    error_result cleanup_error = controller.finish();
    runtime.controller          = nullptr;
    if (!runtime.error.empty()) launch_error = runtime.error;
    if (!launch_error && cleanup_error) launch_error = cleanup_error;
    if (launch_error) {
        impl_->launch_in_flight = false;
        return launch_error;
    }
    return complete_after_synchronize();
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
    bool resolve_deferred_compile_parameters();
    bool allocate_program_buffers();
    bool compile_artifacts();
    bool bind_storage();
    bool bind_streamed_values();
    bool record_graph();
    bool record_segmented_graph();
    bool resolve_binding(const CommandBinding & binding, hrx_buffer_ref_t & result_ref);
    bool resolve_value(ValueId value, size_t length, hrx_buffer_ref_t & result_ref);

    prepared_executable_program                     result;
    prepared_executable_program::impl &             impl;
    hrx_device_t                                    device;
    hrx_stream_t                                    stream;
    transfer_manager &                              transfers;
    weight_residency_cache &                        weights;
    executable_artifact_repository &                artifact_repository;
    const ProgramPlan &                             plan;
    const kernel_corpus &                           corpus;
    CommandProgram                                  commands;
    const executable_bindings &                     bindings;
    const executable_preparation_options &          options;
    size_t                                          record_command_count;
    error_result                                    error;
    std::vector<std::shared_ptr<artifact_record>>   command_artifacts;
    std::vector<std::vector<uint8_t>>               command_constants;
    std::unordered_map<StorageId, hrx_buffer_ref_t> storage_refs;
    std::map<std::pair<StorageId, std::string>, hrx_buffer_ref_t> storage_view_refs;
};

bool executable_program_preparer::resolve_deferred_compile_parameters() {
    for (Command & command : commands.commands) {
        for (const auto & parameter : command.kernel.deferred_compile_parameters) {
            if (parameter.second.value >= plan.graph.values.size()) {
                result.errors_.push_back("deferred compile parameter references an invalid graph value");
                continue;
            }
            const StorageId storage = plan.graph.values[parameter.second.value].access.storage;
            const auto binding =
                std::find_if(bindings.storages.begin(), bindings.storages.end(),
                             [&](const executable_buffer_binding & item) { return item.storage == storage; });
            if (binding == bindings.storages.end()) {
                result.errors_.push_back("deferred compile parameter has no concrete storage binding");
                continue;
            }
            const auto property = binding->integer_properties.find(parameter.second.property);
            if (property == binding->integer_properties.end()) {
                result.errors_.push_back("storage " + std::to_string(storage) + " has no property " +
                                         parameter.second.property);
                continue;
            }
            command.kernel.compile_parameters[parameter.first] = std::to_string(property->second);
        }
        command.kernel.deferred_compile_parameters.clear();
    }
    return result.errors_.empty();
}

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
    if (result.errors_.empty()) resolve_deferred_compile_parameters();
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
    result.segments_                = build_execution_segments(commands, record_command_count);
    result.graph_segments_          = build_graph_segments(commands, record_command_count);
    result.has_streamed_execution_  = std::any_of(
        result.segments_.begin(), result.segments_.end(),
        [](const prepared_execution_segment & segment) { return segment.streamed.valid(); });
    if (record_command_count != 0 && (result.segments_.empty() || result.graph_segments_.empty())) {
        result.errors_.push_back("command program produced no executable segments");
        return false;
    }
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
        if (compiled_artifact->launch.workgroup_storage_bytes != 0) {
            result.errors_.push_back("HRX graph ABI cannot encode dynamic workgroup storage for " + key);
            break;
        }
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
                staging.storage    = binding.storage;
                staging.host_data  = static_cast<uint8_t *>(binding.host_data) + binding.offset;
                staging.length     = binding.length;
                staging.upload     = binding.upload_before_launch;
                staging.initialize = binding.initialize_from_host;
                staging.download   = binding.download_after_completion;
                error = take_status(hrx_buffer_allocate(stream, binding.length, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                                        HRX_BUFFER_USAGE_DEFAULT, &staging.buffer));
                if (error) {
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
        for (const executable_buffer_view & view : binding.views) {
            if (view.name.empty() || view.buffer == nullptr || view.length == 0 ||
                !storage_view_refs
                     .emplace(std::make_pair(binding.storage, view.name),
                              hrx_buffer_ref_t{ view.buffer, view.offset, view.length })
                     .second) {
                result.errors_.push_back("invalid physical view for storage " + std::to_string(binding.storage));
                return false;
            }
            if (retained.insert(view.buffer).second) {
                hrx_buffer_retain(view.buffer);
                impl.retained_buffers.push_back(view.buffer);
            }
        }
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
        if (!binding.storage_binding.empty()) {
            const auto view = storage_view_refs.find({ binding.storage, binding.storage_binding });
            if (view == storage_view_refs.end()) {
                result.errors_.push_back("graph storage " + std::to_string(binding.storage) +
                                         " has no physical view " + binding.storage_binding);
                return false;
            }
            result_ref = view->second;
            return true;
        }
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

bool executable_program_preparer::resolve_value(ValueId value_id, size_t length, hrx_buffer_ref_t & result_ref) {
    if (value_id >= plan.graph.values.size() || length == 0) {
        result.errors_.push_back("invalid graph value readback contract");
        return false;
    }
    const Value & value = plan.graph.values[value_id];
    CommandBinding binding;
    binding.value   = value_id;
    binding.storage = value.access.storage;
    binding.offset  = value.access.offset;
    binding.length  = length;
    const auto persistent =
        std::find_if(commands.persistent_constants.allocations.begin(), commands.persistent_constants.allocations.end(),
                     [&](const PersistentConstantAllocation & allocation) {
                         return allocation.storage == binding.storage;
                     });
    if (persistent != commands.persistent_constants.allocations.end()) {
        binding.origin              = BindingOrigin::PersistentConstant;
        binding.persistent_constant = persistent->id;
    } else {
        const auto transient =
            std::find_if(commands.transients.allocations.begin(), commands.transients.allocations.end(),
                         [&](const TransientAllocation & allocation) {
                             return allocation.storage == binding.storage;
                         });
        if (transient != commands.transients.allocations.end()) {
            binding.origin    = BindingOrigin::Transient;
            binding.transient = transient->id;
        }
    }
    return resolve_binding(binding, result_ref);
}

bool executable_program_preparer::bind_streamed_values() {
    size_t maximum_span_bytes   = 0;
    size_t maximum_expert_count = 0;
    for (prepared_execution_segment & segment : result.segments_) {
        if (!segment.streamed.valid()) {
            continue;
        }
        if (segment.streamed.expert_ids >= plan.graph.values.size()) {
            result.errors_.push_back("streamed segment references an invalid expert-ID value");
            return false;
        }
        const Value & ids = plan.graph.values[segment.streamed.expert_ids];
        size_t element_count       = 0;
        size_t compact_byte_length = 0;
        size_t span_byte_length    = 0;
        if (ids.type != GGML_TYPE_I32 ||
            !value_layout_size(ids, element_count, compact_byte_length, span_byte_length) ||
            compact_byte_length == 0 || span_byte_length == 0) {
            result.errors_.push_back("streamed expert IDs must use a nonempty I32 value");
            return false;
        }
        segment.expert_id_count      = element_count;
        segment.expert_id_bytes      = compact_byte_length;
        segment.expert_id_span_bytes = span_byte_length;
        segment.expert_id_shape      = ids.access.shape;
        segment.expert_id_strides    = ids.access.strides;
        maximum_span_bytes           = std::max(maximum_span_bytes, span_byte_length);
        maximum_expert_count         = std::max(maximum_expert_count, element_count);
        if (impl.value_refs.count(segment.streamed.expert_ids) == 0) {
            hrx_buffer_ref_t ref = {};
            if (!resolve_value(segment.streamed.expert_ids, span_byte_length, ref)) {
                return false;
            }
            impl.value_refs.emplace(segment.streamed.expert_ids, ref);
        }
    }
    if (maximum_span_bytes != 0) {
        hrx_buffer_params_t params = {
            HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
            HRX_MEMORY_ACCESS_ALL,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED,
            0,
        };
        error = take_status(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, maximum_span_bytes,
                                                          &impl.streamed_readback_buffer));
        if (error) {
            result.errors_.push_back("allocate streamed route readback: " + *error);
            return false;
        }
        impl.streamed_runtime                    = std::make_unique<streamed_graph_runtime>();
        impl.streamed_runtime->readback_buffer   = impl.streamed_readback_buffer;
        impl.streamed_runtime->readback_capacity = maximum_span_bytes;
        impl.streamed_runtime->compact_ids.reserve(maximum_expert_count);
        result.host_staging_bytes_ += maximum_span_bytes;
        result.retained_bytes_ += maximum_span_bytes;
    }
    return true;
}

bool executable_program_preparer::record_graph() {
    if (result.has_streamed_execution_) {
        return record_segmented_graph();
    }

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

bool executable_program_preparer::record_segmented_graph() {
    std::vector<const prepared_execution_segment *> execution_segment_starts(record_command_count, nullptr);
    for (const prepared_execution_segment & segment : result.segments_) {
        if (segment.command_begin < execution_segment_starts.size()) {
            execution_segment_starts[segment.command_begin] = &segment;
        }
    }

    std::set<std::pair<int32_t, ValueId>> acquired_scopes;
    impl.graph_segments.resize(result.graph_segments_.size());
    for (size_t segment_index = 0; segment_index < result.graph_segments_.size(); ++segment_index) {
        const prepared_graph_segment &                   segment  = result.graph_segments_[segment_index];
        prepared_executable_program::impl::graph_segment & recorded = impl.graph_segments[segment_index];
        error = take_status(hrx_graph_create(device, 0, &recorded.graph));
        if (error) {
            result.errors_.push_back("create HRX graph segment: " + *error);
            return false;
        }

        std::vector<hrx_graph_node_t>         nodes(segment.command_end - segment.command_begin, nullptr);
        std::unordered_map<StorageId, uint32_t> last_writer;
        hrx_graph_node_t                       previous_acquire    = nullptr;
        size_t                                 expected_node_count = 0;
        for (uint32_t ordinal = segment.command_begin; ordinal < segment.command_end; ++ordinal) {
            const Command & command       = commands.commands[ordinal];
            const size_t    local_ordinal = ordinal - segment.command_begin;
            std::vector<hrx_graph_node_t> deps;
            for (uint32_t dependency : command.dependencies) {
                // The compute stream supplies every cross-segment dependency.
                if (dependency < segment.command_begin) {
                    continue;
                }
                if (dependency >= ordinal || dependency >= segment.command_end ||
                    nodes[dependency - segment.command_begin] == nullptr) {
                    result.errors_.push_back("command dependency was not recorded in its segment");
                    return false;
                }
                deps.push_back(nodes[dependency - segment.command_begin]);
            }
            if ((options.serialize_commands || options.split_commands) && ordinal != segment.command_begin) {
                const hrx_graph_node_t predecessor = nodes[local_ordinal - 1];
                if (std::find(deps.begin(), deps.end(), predecessor) == deps.end()) {
                    deps.push_back(predecessor);
                }
            }
            const prepared_execution_segment * execution_segment = execution_segment_starts[ordinal];
            if (execution_segment != nullptr && execution_segment->streamed.valid() &&
                !execution_segment->streamed.missing_suffix &&
                acquired_scopes
                    .emplace(execution_segment->streamed.layer, execution_segment->streamed.expert_ids)
                    .second) {
                if (impl.streamed_runtime == nullptr || impl.streamed_readback_buffer == nullptr) {
                    result.errors_.push_back("streamed graph has no mapped route readback");
                    return false;
                }
                const auto source = impl.value_refs.find(execution_segment->streamed.expert_ids);
                if (source == impl.value_refs.end() || execution_segment->expert_id_span_bytes > source->second.length) {
                    result.errors_.push_back("streamed graph has no valid route readback source");
                    return false;
                }
                std::vector<hrx_graph_node_t> copy_deps;
                const StorageId expert_ids_storage =
                    plan.graph.values[execution_segment->streamed.expert_ids].access.storage;
                const auto writer = last_writer.find(expert_ids_storage);
                if (writer != last_writer.end()) {
                    copy_deps.push_back(nodes[writer->second - segment.command_begin]);
                }
                if (previous_acquire != nullptr &&
                    std::find(copy_deps.begin(), copy_deps.end(), previous_acquire) == copy_deps.end()) {
                    copy_deps.push_back(previous_acquire);
                }
                const hrx_graph_copy_buffer_node_attrs_t copy = {
                    { source->second.buffer, source->second.offset, execution_segment->expert_id_span_bytes },
                    { impl.streamed_readback_buffer, 0, execution_segment->expert_id_span_bytes },
                };
                hrx_graph_node_t copy_node = nullptr;
                error = take_status(hrx_graph_add_copy_buffer_node(recorded.graph, copy_deps.data(), copy_deps.size(),
                                                                   &copy, &copy_node));
                if (error) {
                    result.errors_.push_back("record streamed route readback: " + *error);
                    return false;
                }
                auto callback       = std::make_unique<streamed_graph_callback>();
                callback->operation = streamed_graph_callback::kind::acquire;
                callback->runtime   = impl.streamed_runtime.get();
                callback->segment   = *execution_segment;
                const hrx_graph_host_call_node_attrs_t callback_attrs = {
                    run_streamed_graph_callback,
                    callback.get(),
                };
                hrx_graph_node_t callback_node = nullptr;
                error = take_status(
                    hrx_graph_add_host_call_node(recorded.graph, &copy_node, 1, &callback_attrs, &callback_node));
                if (error) {
                    result.errors_.push_back("record streamed route callback: " + *error);
                    return false;
                }
                impl.streamed_callbacks.push_back(std::move(callback));
                previous_acquire = callback_node;
                deps.push_back(callback_node);
                expected_node_count += 2;
            }

            prepared_command_diagnostic diagnostic;
            diagnostic.ordinal       = command.ordinal;
            diagnostic.kind          = command.kind;
            diagnostic.label         = command.label;
            diagnostic.binding_count = command.bindings.size();
            if (command.kind == CommandKind::Kernel) {
                const std::shared_ptr<artifact_record> & compiled_artifact = command_artifacts[command.ordinal];
                std::vector<hrx_buffer_ref_t>            concrete_bindings;
                for (const CommandBinding & binding : command.bindings) {
                    hrx_buffer_ref_t concrete = {};
                    if (!resolve_binding(binding, concrete)) {
                        return false;
                    }
                    concrete_bindings.push_back(concrete);
                    if (command.ordinal + 1 == record_command_count && binding.access != ResourceAccess::Read) {
                        impl.last_command_outputs.push_back({ binding.name, binding.access, concrete });
                    }
                }
                const auto & constants = command_constants[command.ordinal];
                const hrx_graph_kernel_node_attrs_t attrs = {
                    compiled_artifact->executable,
                    compiled_artifact->export_ordinal,
                    { { compiled_artifact->launch.workgroup_count[0], compiled_artifact->launch.workgroup_count[1],
                        compiled_artifact->launch.workgroup_count[2] },
                      {},
                      compiled_artifact->launch.subgroup_size },
                    constants.data(),
                    constants.size(),
                    concrete_bindings.data(),
                    concrete_bindings.size(),
                    0,
                };
                error = take_status(hrx_graph_add_kernel_node(recorded.graph, deps.data(), deps.size(), &attrs,
                                                              &nodes[local_ordinal]));
                diagnostic.artifact_key   = compiled_artifact->diagnostic.key;
                diagnostic.constant_bytes = constants.size();
            } else if (command.kind == CommandKind::Copy) {
                hrx_graph_copy_buffer_node_attrs_t attrs = {};
                if (!resolve_binding(command.bindings[0], attrs.src) ||
                    !resolve_binding(command.bindings[1], attrs.dst)) {
                    return false;
                }
                if (command.ordinal + 1 == record_command_count) {
                    impl.last_command_outputs.push_back(
                        { command.bindings[1].name, command.bindings[1].access, attrs.dst });
                }
                error = take_status(hrx_graph_add_copy_buffer_node(recorded.graph, deps.data(), deps.size(), &attrs,
                                                                   &nodes[local_ordinal]));
            } else if (command.kind == CommandKind::Fill) {
                const uint32_t fill_byte =
                    static_cast<uint32_t>(command.kernel.integer_parameters.at("fill_byte")) & 0xffu;
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
                error = take_status(hrx_graph_add_fill_buffer_node(recorded.graph, deps.data(), deps.size(), &attrs,
                                                                   &nodes[local_ordinal]));
            } else {
                error = take_status(
                    hrx_graph_add_empty_node(recorded.graph, deps.data(), deps.size(), &nodes[local_ordinal]));
            }
            if (error) {
                result.errors_.push_back("record command " + std::to_string(command.ordinal) + ": " + *error);
                return false;
            }
            ++expected_node_count;
            if (options.split_commands) {
                const hrx_graph_node_t command_node = nodes[local_ordinal];
                error = take_status(
                    hrx_graph_add_empty_node(recorded.graph, &command_node, 1, &nodes[local_ordinal]));
                if (error) {
                    result.errors_.push_back("record split after command " + std::to_string(command.ordinal) + ": " +
                                             *error);
                    return false;
                }
                ++expected_node_count;
            }
            for (const CommandBinding & binding : command.bindings) {
                if (binding.storage != kInvalidId && binding.access != ResourceAccess::Read) {
                    last_writer[binding.storage] = command.ordinal;
                }
            }
            result.commands_.push_back(std::move(diagnostic));
        }

        size_t actual_node_count = 0;
        error = take_status(hrx_graph_size(recorded.graph, &actual_node_count));
        if (error) {
            result.errors_.push_back("query HRX graph segment size: " + *error);
            return false;
        }
        if (actual_node_count != expected_node_count) {
            result.errors_.push_back("recorded HRX graph segment node count does not match command program");
            return false;
        }
        result.node_count_ += actual_node_count;
        error = take_status(hrx_graph_instantiate(recorded.graph, 0, &recorded.executable));
        if (error) {
            result.errors_.push_back("instantiate HRX graph segment: " + *error);
            return false;
        }
    }
    result.prepared_ = result.errors_.empty() && impl.graph_segments.size() == result.graph_segments_.size();
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
    if (!bind_streamed_values()) {
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
