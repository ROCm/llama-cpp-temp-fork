#pragma once

#include "graph/command-program.h"
#include "hrx-interop-utils.h"
#include "hrx_runtime.h"
#include "weight-residency.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml::hrx {

class transfer_manager;
class executable_program_preparer;

struct packed_kernel_constants {
    std::vector<uint8_t>     bytes;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

packed_kernel_constants pack_kernel_constants(const kernel_definition & definition, const Command & command);
std::string             kernel_artifact_key(const kernel_definition & definition,
                                            const Command &           command,
                                            const std::string &       target);

struct prepared_artifact_diagnostic {
    std::string             key;
    std::string             kernel_id;
    std::array<uint32_t, 3> workgroup_count = { 0, 0, 0 };
    std::array<uint32_t, 3> workgroup_size  = { 0, 0, 0 };
    uint32_t                subgroup_size   = 0;
    size_t                  constant_bytes  = 0;
    size_t                  binding_count   = 0;
    std::string             manifest_json;
    std::string             compile_report_json;
    std::string             final_module_text;
};

struct prepared_command_diagnostic {
    uint32_t    ordinal = 0;
    CommandKind kind    = CommandKind::Kernel;
    std::string label;
    std::string artifact_key;
    size_t      constant_bytes = 0;
    size_t      binding_count  = 0;
};

struct prepared_binding_snapshot {
    std::string          name;
    ResourceAccess       access = ResourceAccess::Read;
    size_t               length = 0;
    std::vector<uint8_t> bytes;
};

struct executable_preparation_options {
    std::string target;
    size_t      recorder_buffer_limit = 256ull * 1024ull * 1024ull;
    size_t      command_limit         = SIZE_MAX;
    bool        serialize_commands    = false;
    bool        split_commands        = false;
    std::string sanitizer;
    std::string sanitizer_reporting;
};

struct executable_buffer_binding {
    StorageId    storage                   = kInvalidId;
    hrx_buffer_t buffer                    = nullptr;
    void *       host_data                 = nullptr;
    uint64_t     buffer_identity           = 0;
    uint64_t     generation                = 0;
    size_t       capacity                  = 0;
    size_t       offset                    = 0;
    size_t       length                    = 0;
    bool         upload_before_launch      = false;
    bool         initialize_from_host      = false;
    bool         download_after_completion = false;
    bool         weight                    = false;
    bool         mutable_state             = false;
    bool         exported                  = false;
    std::string  layout                    = "ggml-native";
    const kernel_storage_transform * transform = nullptr;
};

struct executable_bindings {
    BindingSnapshot                        snapshot;
    std::vector<executable_buffer_binding> storages;

    std::string format() const;
    std::string serialize_json() const;
};

class prepared_executable_program;

class executable_artifact_repository {
  public:
    executable_artifact_repository();
    ~executable_artifact_repository();
    executable_artifact_repository(const executable_artifact_repository &)             = delete;
    executable_artifact_repository & operator=(const executable_artifact_repository &) = delete;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class executable_program_preparer;
    friend prepared_executable_program prepare_executable_program(hrx_device_t,
                                                                  hrx_stream_t,
                                                                  transfer_manager &,
                                                                  weight_residency_cache &,
                                                                  executable_artifact_repository &,
                                                                  const ProgramPlan &,
                                                                  const kernel_corpus &,
                                                                  const CommandProgram &,
                                                                  const executable_bindings &,
                                                                  const executable_preparation_options &);
};

class prepared_executable_program {
  public:
    prepared_executable_program();
    ~prepared_executable_program();
    prepared_executable_program(prepared_executable_program &&) noexcept;
    prepared_executable_program & operator=(prepared_executable_program &&) noexcept;
    prepared_executable_program(const prepared_executable_program &)             = delete;
    prepared_executable_program & operator=(const prepared_executable_program &) = delete;

    bool valid() const { return prepared_ && errors_.empty(); }

    size_t node_count() const { return node_count_; }

    size_t artifact_count() const { return artifacts_.size(); }

    size_t retained_bytes() const { return retained_bytes_; }

    size_t borrowed_device_weight_bytes() const { return borrowed_device_weight_bytes_; }

    size_t resident_host_weight_bytes() const { return resident_host_weight_bytes_; }

    size_t host_staging_bytes() const { return host_staging_bytes_; }

    size_t transient_bytes() const { return transient_bytes_; }

    size_t persistent_constant_bytes() const { return persistent_constant_bytes_; }

    size_t source_command_count() const { return source_command_count_; }

    bool command_prefix() const { return command_prefix_; }

    bool split_commands() const { return split_commands_; }

    bool serialized_commands() const { return serialized_commands_; }

    const AllocationFingerprint & allocation_fingerprint() const { return allocation_fingerprint_; }

    error_result rebind(const executable_bindings & bindings, const AllocationFingerprint & fingerprint);
    error_result launch(hrx_stream_t stream);
    error_result complete_after_synchronize();
    error_result snapshot_transients(std::vector<uint8_t> & bytes);
    error_result snapshot_last_command_outputs(std::vector<prepared_binding_snapshot> & snapshots,
                                               size_t                                   maximum_binding_bytes);
    void         abandon_after_synchronize();

    const std::vector<std::string> & errors() const { return errors_; }

    const std::vector<prepared_artifact_diagnostic> & artifacts() const { return artifacts_; }

    const std::vector<prepared_command_diagnostic> & commands() const { return commands_; }

    std::string format() const;
    std::string serialize_json() const;

  private:
    struct impl;
    std::unique_ptr<impl>                     impl_;
    bool                                      prepared_                     = false;
    size_t                                    node_count_                   = 0;
    size_t                                    retained_bytes_               = 0;
    size_t                                    borrowed_device_weight_bytes_ = 0;
    size_t                                    resident_host_weight_bytes_   = 0;
    size_t                                    host_staging_bytes_           = 0;
    size_t                                    transient_bytes_              = 0;
    size_t                                    persistent_constant_bytes_    = 0;
    size_t                                    source_command_count_         = 0;
    bool                                      command_prefix_               = false;
    bool                                      split_commands_               = false;
    bool                                      serialized_commands_          = false;
    AllocationFingerprint                     allocation_fingerprint_;
    std::vector<prepared_artifact_diagnostic> artifacts_;
    std::vector<prepared_command_diagnostic>  commands_;
    std::vector<std::string>                  errors_;
    friend class executable_program_preparer;
    friend prepared_executable_program prepare_executable_program(hrx_device_t,
                                                                  hrx_stream_t,
                                                                  transfer_manager &,
                                                                  weight_residency_cache &,
                                                                  executable_artifact_repository &,
                                                                  const ProgramPlan &,
                                                                  const kernel_corpus &,
                                                                  const CommandProgram &,
                                                                  const executable_bindings &,
                                                                  const executable_preparation_options &);
};

prepared_executable_program prepare_executable_program(hrx_device_t                           device,
                                                       hrx_stream_t                           stream,
                                                       transfer_manager &                     transfers,
                                                       weight_residency_cache &               weights,
                                                       executable_artifact_repository &       artifacts,
                                                       const ProgramPlan &                    plan,
                                                       const kernel_corpus &                  corpus,
                                                       const CommandProgram &                 commands,
                                                       const executable_bindings &            bindings,
                                                       const executable_preparation_options & options);

}  // namespace ggml::hrx
