#pragma once

#include "kernel-corpus-catalog.h"
#include "transitional-resource-access.h"
#include "transitional-schedule.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ggml::hrx {

template <typename T> struct kernel_span {
    const T * items = nullptr;
    size_t    count = 0;

    const T * begin() const { return items; }

    const T * end() const { return items == nullptr ? nullptr : items + count; }

    const T * data() const { return items; }

    size_t size() const { return count; }

    bool empty() const { return count == 0; }

    const T & operator[](size_t index) const { return items[index]; }

    const T & front() const { return items[0]; }
};

struct kernel_compile_config {
    const char * key   = "";
    const char * value = "";
};

struct kernel_binding_definition {
    const char *   name   = "";
    ResourceAccess access = ResourceAccess::Read;
};

struct kernel_scalar_definition {
    const char * name = "";
    const char * type = "";
};

enum kernel_source_format {
    KERNEL_SOURCE_FORMAT_TEXT,
    KERNEL_SOURCE_FORMAT_BINARY,
};

struct kernel_source_span {
    const char *         data;
    size_t               length;
    kernel_source_format format;
};

struct kernel_source {
    kernel_source_span         source;
    const kernel_source_span * dependencies;
    size_t                     dependency_count;
};

struct kernel_source_ref {
    const char *          path     = "";
    const kernel_source * contents = nullptr;
};

struct kernel_compile_recipe {
    const char *                   mode        = "";
    const char *                   link_module = "";
    kernel_span<kernel_source_ref> primary_sources;
    kernel_span<kernel_source_ref> library_sources;
};

struct kernel_definition {
    const char *                           family = "";
    const char *                           name   = "";
    uint64_t                               id     = GGML_HRX_KERNEL_ID_UNCATALOGED;
    const char *                           source = "";
    kernel_span<const char *>              dependencies;
    const char *                           symbol          = "";
    const char *                           backend         = "";
    const char *                           target_selector = "";
    kernel_span<kernel_compile_config>     compile_config;
    kernel_span<const char *>              scalar_parameters;
    kernel_span<kernel_binding_definition> bindings;
    kernel_span<kernel_scalar_definition>  workload_parameters;
    kernel_span<kernel_scalar_definition>  launch_parameters;
    kernel_compile_recipe                  compile_recipe;
};

struct kernel_corpus {
    const char *                   schema            = "ggml-hrx-kernel-corpus-v2";
    const char *                   upstream_revision = "";
    size_t                         plan_case_count   = 0;
    kernel_span<kernel_definition> kernels;
};

enum class kernel_resolve_status : uint8_t {
    KERNEL_RESOLVE_STATUS_FOUND,
    KERNEL_RESOLVE_STATUS_NATIVE_GAP,
    KERNEL_RESOLVE_STATUS_UNCATALOGED_NATIVE,
    KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY,
    KERNEL_RESOLVE_STATUS_HASH_COLLISION,
    KERNEL_RESOLVE_STATUS_INVALID_NATIVE_GAP,
    KERNEL_RESOLVE_STATUS_UNSUPPORTED_TARGET,
};

struct kernel_resolve_result {
    kernel_resolve_status     status     = kernel_resolve_status::KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY;
    const kernel_definition * definition = nullptr;

    bool found() const { return status == kernel_resolve_status::KERNEL_RESOLVE_STATUS_FOUND && definition != nullptr; }
};

const kernel_source * get_kernel_source(const char * source_path);
const kernel_corpus & get_qwen_kernel_corpus();
kernel_resolve_result resolve_kernel_definition(const kernel_corpus &               corpus,
                                                const std::string &                 target,
                                                const std::string &                 family,
                                                const std::string &                 name,
                                                uint64_t                            id,
                                                KernelSpecialization::ExecutionKind execution_kind);
kernel_resolve_result resolve_kernel_definition(const kernel_corpus &        corpus,
                                                const std::string &          target,
                                                const KernelSpecialization & kernel);
const char *          kernel_resolve_status_name(kernel_resolve_status status);
std::string           format_kernel_resolve_error(const kernel_resolve_result & result,
                                                  const std::string &           family,
                                                  const std::string &           name);
std::string format_kernel_resolve_error(const kernel_resolve_result & result, const KernelSpecialization & kernel);
VerificationResult verify_kernel_corpus(const kernel_corpus & corpus);
std::string        format_kernel_corpus(const kernel_corpus & corpus);

}  // namespace ggml::hrx
