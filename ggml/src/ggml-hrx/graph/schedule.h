#pragma once

#include "graph-ir.h"
#include "kernel-corpus-catalog.h"

#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

struct DeferredCompileParameter {
    ValueId value = kInvalidId;
    std::string property;

    bool operator==(const DeferredCompileParameter & other) const {
        return value == other.value && property == other.property;
    }
};

struct KernelSpecialization {
    enum class ExecutionKind : uint8_t {
        Native,
        NativeGap,
        NativeEager,
        CpuFallback,
    };

    std::string family;
    std::string variant;
    std::map<std::string, int64_t> integer_parameters;
    ExecutionKind execution_kind = ExecutionKind::Native;
    std::map<std::string, std::string> compile_parameters;
    std::map<std::string, DeferredCompileParameter> deferred_compile_parameters;
    uint64_t kernel_id = GGML_HRX_KERNEL_ID_UNCATALOGED;
};

struct TensorBinding {
    std::string role;
    ValueId value = kInvalidId;
    size_t offset = 0;
    size_t length = 0;
    std::string storage_binding;

    bool operator==(const TensorBinding & other) const {
        return role == other.role && value == other.value && offset == other.offset && length == other.length &&
               storage_binding == other.storage_binding;
    }
    bool operator!=(const TensorBinding & other) const { return !(*this == other); }
};

struct Dispatch {
    KernelSpecialization kernel;
    std::vector<TensorBinding> bindings;
    std::vector<uint32_t> dependencies;
    ValueId streamed_weight = kInvalidId;
    ValueId streamed_expert_ids = kInvalidId;
    bool streamed_missing_suffix = false;
    uint32_t streamed_expert_begin = 0;
    uint32_t streamed_expert_end = 0xffffffffu;
    uint32_t streamed_load_chunk_size = 0;

    bool streamed() const { return streamed_weight != kInvalidId || streamed_expert_ids != kInvalidId; }
};

enum class RootDisposition : uint8_t {
    Materialized,
    OwnedEndpointReplacement,
    Unresolved,
};

struct RootContract {
    ValueId value = kInvalidId;
    RootDisposition disposition = RootDisposition::Unresolved;
    std::string replacement;
};

struct Invocation {
    KernelSpecialization kernel;
    std::vector<OperationId> covered_operations;
    std::vector<TensorBinding> inputs;
    std::vector<TensorBinding> outputs;
    std::vector<Dispatch> dispatches;
    std::string stage;
    int32_t layer = -1;
    std::string recipe;
    std::vector<uint32_t> logical_components;
};

struct Schedule {
    std::string graph_fingerprint;
    std::string workload;
    std::string oracle_revision;
    size_t expected_dispatch_count = 0;
    std::vector<Invocation> invocations;
    std::vector<RootContract> roots;
};

struct VerificationResult {
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

VerificationResult verify_schedule(const Graph & graph, const Schedule & schedule);
Schedule deserialize_schedule_json(const std::string & json, std::vector<std::string> & errors);
std::string serialize_schedule_json(const Schedule & schedule);
size_t schedule_dispatch_count(const Schedule & schedule);
size_t schedule_execution_kind_count(const Schedule & schedule, KernelSpecialization::ExecutionKind kind);
std::string kernel_specialization_name(const KernelSpecialization & kernel);
const char * execution_kind_name(KernelSpecialization::ExecutionKind kind);
const char * root_disposition_name(RootDisposition disposition);

} // namespace ggml::hrx
