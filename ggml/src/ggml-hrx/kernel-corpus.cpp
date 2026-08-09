#include "kernel-corpus.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace ggml::hrx {
namespace {

const char * kernel_resource_access_name(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read:
            return "read";
        case ResourceAccess::Write:
            return "write";
        case ResourceAccess::ReadWrite:
            return "read_write";
    }
    return "unknown";
}

struct kernel_source_record_entry {
    const char *          source_path;
    const kernel_source * source;
};

static bool string_equal(const char * lhs, const char * rhs) {
    return std::strcmp(lhs != nullptr ? lhs : "", rhs != nullptr ? rhs : "") == 0;
}

static bool string_empty(const char * value) {
    return value == nullptr || value[0] == 0;
}

static bool contains_source_ref(kernel_span<kernel_source_ref> values, const char * path) {
    return std::find_if(values.begin(), values.end(),
                        [&](const kernel_source_ref & item) { return string_equal(item.path, path); }) != values.end();
}

static bool string_span_equal(kernel_span<const char *> lhs, kernel_span<const char *> rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                                                  [](const char * a, const char * b) { return string_equal(a, b); });
}

static bool scalar_span_equal(kernel_span<kernel_scalar_definition> lhs, kernel_span<kernel_scalar_definition> rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](const kernel_scalar_definition & a, const kernel_scalar_definition & b) {
                          return string_equal(a.name, b.name) && string_equal(a.type, b.type);
                      });
}

static bool binding_span_equal(kernel_span<kernel_binding_definition> lhs, kernel_span<kernel_binding_definition> rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](const kernel_binding_definition & a, const kernel_binding_definition & b) {
                          return string_equal(a.name, b.name) && a.access == b.access;
                      });
}

static bool kernel_variant_contract_equal(const kernel_definition & lhs, const kernel_definition & rhs) {
    return string_equal(lhs.backend, rhs.backend) && string_span_equal(lhs.scalar_parameters, rhs.scalar_parameters) &&
           scalar_span_equal(lhs.workload_parameters, rhs.workload_parameters) &&
           scalar_span_equal(lhs.launch_parameters, rhs.launch_parameters) &&
           binding_span_equal(lhs.bindings, rhs.bindings);
}

// The corpus records refer to source records from the first include.
// clang-format off
#include "kernel-corpus-sources.inc"
#include "kernel-corpus.inc"
// clang-format on

}  // namespace

const kernel_source * get_kernel_source(const char * source_path) {
    if (source_path == nullptr) {
        return nullptr;
    }
    for (const kernel_source_record_entry & entry : kernel_source_records) {
        if (std::strcmp(source_path, entry.source_path) == 0) {
            return entry.source;
        }
    }
    return nullptr;
}

const kernel_corpus & get_kernel_corpus() {
    return embedded_kernel_corpus;
}

kernel_resolve_result resolve_kernel_definition(const kernel_corpus &               corpus,
                                                const std::string &                 target,
                                                const std::string &                 family,
                                                const std::string &                 name,
                                                uint64_t                            id,
                                                KernelSpecialization::ExecutionKind execution_kind) {
    if (execution_kind == KernelSpecialization::ExecutionKind::NativeGap) {
        return {
            id == GGML_HRX_KERNEL_ID_UNCATALOGED ? kernel_resolve_status::KERNEL_RESOLVE_STATUS_NATIVE_GAP :
                                                   kernel_resolve_status::KERNEL_RESOLVE_STATUS_INVALID_NATIVE_GAP,
            nullptr,
        };
    }
    if (execution_kind != KernelSpecialization::ExecutionKind::Native &&
        execution_kind != KernelSpecialization::ExecutionKind::NativeEager) {
        return { kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNCATALOGED_NATIVE, nullptr };
    }
    if (id == GGML_HRX_KERNEL_ID_UNCATALOGED) {
        return { kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNCATALOGED_NATIVE, nullptr };
    }
    bool                      id_match        = false;
    bool                      name_match      = false;
    const kernel_definition * default_variant = nullptr;
    for (const kernel_definition & kernel : corpus.kernels) {
        if (kernel.id != id) {
            continue;
        }
        id_match = true;
        if (string_equal(kernel.family, family.c_str()) && string_equal(kernel.name, name.c_str())) {
            name_match = true;
            if (string_equal(kernel.target_selector, target.c_str())) {
                return { kernel_resolve_status::KERNEL_RESOLVE_STATUS_FOUND, &kernel };
            }
            if (string_empty(kernel.target_selector)) {
                default_variant = &kernel;
            }
        }
    }
    if (default_variant != nullptr) {
        return { kernel_resolve_status::KERNEL_RESOLVE_STATUS_FOUND, default_variant };
    }
    if (name_match) {
        return { kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNSUPPORTED_TARGET, nullptr };
    }
    return { id_match ? kernel_resolve_status::KERNEL_RESOLVE_STATUS_HASH_COLLISION :
                        kernel_resolve_status::KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY,
             nullptr };
}

kernel_resolve_result resolve_kernel_definition(const kernel_corpus &        corpus,
                                                const std::string &          target,
                                                const KernelSpecialization & kernel) {
    return resolve_kernel_definition(corpus, target, kernel.family, kernel.variant, kernel.kernel_id,
                                     kernel.execution_kind);
}

const char * kernel_resolve_status_name(kernel_resolve_status status) {
    switch (status) {
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_FOUND:
            return "found";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_NATIVE_GAP:
            return "native_gap";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNCATALOGED_NATIVE:
            return "uncataloged_native";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY:
            return "missing_active_corpus_entry";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_HASH_COLLISION:
            return "hash_collision";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_INVALID_NATIVE_GAP:
            return "invalid_native_gap";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNSUPPORTED_TARGET:
            return "unsupported_target";
    }
    return "unknown";
}

std::string format_kernel_resolve_error(const kernel_resolve_result & result,
                                        const std::string &           family,
                                        const std::string &           name) {
    KernelSpecialization kernel;
    kernel.family           = family;
    kernel.variant          = name;
    const std::string label = kernel_specialization_name(kernel);
    switch (result.status) {
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_FOUND:
            return "";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_NATIVE_GAP:
            return "native gap for " + label;
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNCATALOGED_NATIVE:
            return "uncataloged native kernel " + label;
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY:
            return "cataloged kernel " + label + " is not available in the active corpus";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_HASH_COLLISION:
            return "kernel catalog id collision while resolving " + label;
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_INVALID_NATIVE_GAP:
            return "native gap " + label + " unexpectedly has a catalog id";
        case kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNSUPPORTED_TARGET:
            return "cataloged kernel " + label + " has no implementation for the requested target";
    }
    return "unknown kernel resolution failure for " + label;
}

std::string format_kernel_resolve_error(const kernel_resolve_result & result, const KernelSpecialization & kernel) {
    return format_kernel_resolve_error(result, kernel.family, kernel.variant);
}

VerificationResult verify_kernel_corpus(const kernel_corpus & corpus) {
    VerificationResult result;
    if (!string_equal(corpus.schema, "ggml-hrx-kernel-corpus-v2")) {
        result.errors.push_back("unsupported kernel corpus schema");
    }
    if (string_empty(corpus.upstream_revision)) {
        result.errors.push_back("kernel corpus has no upstream revision");
    }
    if (corpus.plan_case_count == 0) {
        result.errors.push_back("kernel corpus has no compile plan cases");
    }
    std::set<std::string>                            variants;
    std::map<std::string, const kernel_definition *> contracts;
    for (const kernel_definition & kernel : corpus.kernels) {
        if (string_empty(kernel.family) || string_empty(kernel.name) || string_empty(kernel.source) ||
            string_empty(kernel.symbol) || string_empty(kernel.backend)) {
            result.errors.push_back("kernel definition is incomplete");
        }
        if (kernel.id != kernel_catalog_id(kernel.family != nullptr ? kernel.family : "",
                                           kernel.name != nullptr ? kernel.name : "")) {
            result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                    " has an invalid catalog id");
        }
        const bool source_is_primary = contains_source_ref(kernel.compile_recipe.primary_sources, kernel.source);
        const bool source_is_library = contains_source_ref(kernel.compile_recipe.library_sources, kernel.source);
        if ((!string_equal(kernel.compile_recipe.mode, "direct") &&
             !string_equal(kernel.compile_recipe.mode, "archive")) ||
            kernel.compile_recipe.primary_sources.empty() || (!source_is_primary && !source_is_library) ||
            (string_equal(kernel.compile_recipe.mode, "archive") && string_empty(kernel.compile_recipe.link_module))) {
            result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                    " has an invalid BUILD compile recipe");
        }
        for (const kernel_source_ref & source : kernel.compile_recipe.primary_sources) {
            if (string_empty(source.path) || source.contents == nullptr) {
                result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                        " has an invalid embedded primary source reference");
            }
        }
        for (const kernel_source_ref & source : kernel.compile_recipe.library_sources) {
            if (string_empty(source.path) || source.contents == nullptr) {
                result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                        " has an invalid embedded library source reference");
            }
        }
        const std::string full_name = std::string(kernel.family != nullptr ? kernel.family : "") + ":" +
                                      std::string(kernel.name != nullptr ? kernel.name : "");
        const std::string target_selector = kernel.target_selector != nullptr ? kernel.target_selector : "";
        if (!variants.insert(full_name + "@" + target_selector).second) {
            result.errors.push_back("kernel corpus repeats target variant " + full_name + "@" +
                                    (target_selector.empty() ? "default" : target_selector));
        }
        const auto contract = contracts.emplace(full_name, &kernel);
        if (!contract.second && !kernel_variant_contract_equal(*contract.first->second, kernel)) {
            result.errors.push_back("kernel target variants disagree on ABI for " + full_name);
        }
        std::set<std::string> binding_names;
        for (const kernel_binding_definition & binding : kernel.bindings) {
            if (string_empty(binding.name) ||
                !binding_names.insert(binding.name != nullptr ? binding.name : "").second) {
                result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                        " has invalid binding names");
            }
        }
        if (kernel.bindings.size() == 0) {
            result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                    " has no binding ABI");
        }
    }
    return result;
}

std::string format_kernel_corpus(const kernel_corpus & corpus) {
    std::ostringstream out;
    out << "kernel-corpus " << corpus.schema << " revision=" << corpus.upstream_revision
        << " kernels=" << corpus.kernels.size() << " plan_cases=" << corpus.plan_case_count << '\n';
    for (const kernel_definition & kernel : corpus.kernels) {
        out << "  kernel " << kernel.family << ':' << kernel.name << " id=0x" << std::hex << kernel.id << std::dec
            << " backend=" << kernel.backend
            << " target=" << (string_empty(kernel.target_selector) ? "default" : kernel.target_selector) << " symbol=@"
            << kernel.symbol << " source=" << kernel.source << '\n';
        out << "    recipe " << kernel.compile_recipe.mode;
        if (!string_empty(kernel.compile_recipe.link_module)) {
            out << " module=" << kernel.compile_recipe.link_module;
        }
        out << " primary=";
        for (size_t i = 0; i < kernel.compile_recipe.primary_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.primary_sources[i].path;
        }
        out << " libraries=";
        for (size_t i = 0; i < kernel.compile_recipe.library_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.library_sources[i].path;
        }
        out << '\n';
        for (size_t i = 0; i < kernel.bindings.size(); ++i) {
            out << "    binding[" << i << "] " << kernel.bindings[i].name << ' '
                << kernel_resource_access_name(kernel.bindings[i].access) << '\n';
        }
        for (const kernel_scalar_definition & parameter : kernel.workload_parameters) {
            out << "    workload " << parameter.name << ' ' << parameter.type << '\n';
        }
        for (const kernel_scalar_definition & parameter : kernel.launch_parameters) {
            out << "    launch " << parameter.name << ' ' << parameter.type << '\n';
        }
    }
    return out.str();
}

}  // namespace ggml::hrx
