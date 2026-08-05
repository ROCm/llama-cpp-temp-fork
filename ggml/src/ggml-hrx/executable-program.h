#pragma once

#include "graph/command-program.h"
#include "hrx-interop-utils.h"
#include "weight-residency.h"

#include "hrx_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml::hrx {

class TransferManager;
class ExecutableProgramPreparer;

struct PackedKernelConstants {
    std::vector<uint8_t> bytes;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

PackedKernelConstants pack_kernel_constants(const KernelDefinition & definition,
                                             const Command & command);
std::string kernel_artifact_key(const KernelDefinition & definition,
                                const Command & command);
std::string kernel_artifact_key(const KernelDefinition & definition,
                                const Command & command,
                                const std::string & target);

struct PreparedArtifactDiagnostic {
    std::string key;
    std::string kernel_id;
    std::array<uint32_t, 3> workgroup_count = { 0, 0, 0 };
    std::array<uint32_t, 3> workgroup_size = { 0, 0, 0 };
    uint32_t subgroup_size = 0;
    size_t constant_bytes = 0;
    size_t binding_count = 0;
    std::string manifest_json;
    std::string compile_report_json;
    std::string final_module_text;
};

struct PreparedCommandDiagnostic {
    uint32_t ordinal = 0;
    CommandKind kind = CommandKind::Kernel;
    std::string label;
    std::string artifact_key;
    size_t constant_bytes = 0;
    size_t binding_count = 0;
};

struct PreparedBindingSnapshot {
    std::string name;
    ResourceAccess access = ResourceAccess::Read;
    size_t length = 0;
    std::vector<uint8_t> bytes;
};

struct ExecutablePreparationOptions {
    std::string target;
    size_t recorder_buffer_limit = 256ull * 1024ull * 1024ull;
    size_t command_limit = SIZE_MAX;
    bool serialize_commands = false;
    bool split_commands = false;
    std::string sanitizer;
    std::string sanitizer_reporting;
};

struct ExecutableBufferBinding {
    StorageId storage = kInvalidId;
    hrx_buffer_t buffer = nullptr;
    void * host_data = nullptr;
    uint64_t buffer_identity = 0;
    uint64_t generation = 0;
    size_t capacity = 0;
    size_t offset = 0;
    size_t length = 0;
    bool upload_before_launch = false;
    bool initialize_from_host = false;
    bool download_after_completion = false;
    bool weight = false;
    bool mutable_state = false;
    bool exported = false;
    std::string layout = "ggml-native";
};

struct ExecutableBindings {
    BindingSnapshot snapshot;
    std::vector<ExecutableBufferBinding> storages;

    std::string format() const;
    std::string serialize_json() const;
};

class PreparedExecutableProgram;

class ExecutableArtifactRepository {
public:
    ExecutableArtifactRepository();
    ~ExecutableArtifactRepository();
    ExecutableArtifactRepository(const ExecutableArtifactRepository &) = delete;
    ExecutableArtifactRepository & operator=(const ExecutableArtifactRepository &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class ExecutableProgramPreparer;
    friend PreparedExecutableProgram prepare_executable_program(
        hrx_device_t, hrx_stream_t, TransferManager &, WeightResidencyCache &,
        ExecutableArtifactRepository &, const ProgramPlan &,
        const KernelCorpus &, const CommandProgram &, const ExecutableBindings &,
        const ExecutablePreparationOptions &);
};

class PreparedExecutableProgram {
public:
    PreparedExecutableProgram();
    ~PreparedExecutableProgram();
    PreparedExecutableProgram(PreparedExecutableProgram &&) noexcept;
    PreparedExecutableProgram & operator=(PreparedExecutableProgram &&) noexcept;
    PreparedExecutableProgram(const PreparedExecutableProgram &) = delete;
    PreparedExecutableProgram & operator=(const PreparedExecutableProgram &) = delete;

    bool valid() const { return prepared_ && errors_.empty(); }
    size_t node_count() const { return node_count_; }
    size_t artifact_count() const { return artifacts_.size(); }
    size_t retained_bytes() const { return retained_bytes_; }
    size_t borrowed_device_weight_bytes() const { return borrowed_device_weight_bytes_; }
    size_t resident_host_weight_bytes() const { return resident_host_weight_bytes_; }
    size_t host_staging_bytes() const { return host_staging_bytes_; }
    size_t transient_bytes() const { return transient_bytes_; }
    size_t source_command_count() const { return source_command_count_; }
    bool command_prefix() const { return command_prefix_; }
    bool split_commands() const { return split_commands_; }
    bool serialized_commands() const { return serialized_commands_; }
    const AllocationFingerprint & allocation_fingerprint() const { return allocation_fingerprint_; }
    ErrorResult rebind(const ExecutableBindings & bindings);
    ErrorResult launch(hrx_stream_t stream);
    ErrorResult complete_after_synchronize();
    ErrorResult snapshot_transients(std::vector<uint8_t> & bytes);
    ErrorResult snapshot_last_command_outputs(
        std::vector<PreparedBindingSnapshot> & snapshots, size_t maximum_binding_bytes);
    void abandon_after_synchronize();
    const std::vector<std::string> & errors() const { return errors_; }
    const std::vector<PreparedArtifactDiagnostic> & artifacts() const { return artifacts_; }
    const std::vector<PreparedCommandDiagnostic> & commands() const { return commands_; }
    std::string format() const;
    std::string serialize_json() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool prepared_ = false;
    size_t node_count_ = 0;
    size_t retained_bytes_ = 0;
    size_t borrowed_device_weight_bytes_ = 0;
    size_t resident_host_weight_bytes_ = 0;
    size_t host_staging_bytes_ = 0;
    size_t transient_bytes_ = 0;
    size_t source_command_count_ = 0;
    bool command_prefix_ = false;
    bool split_commands_ = false;
    bool serialized_commands_ = false;
    AllocationFingerprint allocation_fingerprint_;
    std::vector<PreparedArtifactDiagnostic> artifacts_;
    std::vector<PreparedCommandDiagnostic> commands_;
    std::vector<std::string> errors_;
    friend class ExecutableProgramPreparer;
    friend PreparedExecutableProgram prepare_executable_program(
        hrx_device_t, hrx_stream_t, TransferManager &, WeightResidencyCache &,
        ExecutableArtifactRepository &, const ProgramPlan &,
        const KernelCorpus &, const CommandProgram &, const ExecutableBindings &,
        const ExecutablePreparationOptions &);
};

PreparedExecutableProgram prepare_executable_program(
    hrx_device_t device, hrx_stream_t stream, TransferManager & transfers,
    WeightResidencyCache & weights,
    ExecutableArtifactRepository & artifacts,
    const ProgramPlan & plan, const KernelCorpus & corpus, const CommandProgram & commands,
    const ExecutableBindings & bindings,
    const ExecutablePreparationOptions & options);

} // namespace ggml::hrx
