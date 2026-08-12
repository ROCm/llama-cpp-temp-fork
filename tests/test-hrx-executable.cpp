#include "executable-program.h"
#include "kernel-corpus.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

namespace {

ggml::hrx::kernel_definition definition() {
    static const char                          source_text[] = "test kernel";
    static ggml::hrx::kernel_source            source;
    static ggml::hrx::kernel_source_ref        primary_sources[1];
    static ggml::hrx::kernel_scalar_definition workload_parameters[1];
    static ggml::hrx::kernel_scalar_definition launch_parameters[2];
    source.source.data          = source_text;
    source.source.length        = sizeof(source_text) - 1;
    source.source.format        = ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT;
    source.dependencies         = nullptr;
    source.dependency_count     = 0;
    primary_sources[0].path     = "test.loom";
    primary_sources[0].contents = &source;
    workload_parameters[0].name = "rows";
    workload_parameters[0].type = "index";
    launch_parameters[0].name   = "rows";
    launch_parameters[0].type   = "index";
    launch_parameters[1].name   = "columns";
    launch_parameters[1].type   = "index";

    ggml::hrx::kernel_definition result;
    result.family                         = "test_family";
    result.name                           = "test_kernel";
    result.id                             = ggml::hrx::kernel_catalog_id(result.family, result.name);
    result.source                         = "test.loom";
    result.symbol                         = "test_kernel";
    result.backend                        = "amdgpu";
    result.target_selector                = "gfx1151";
    result.workload_parameters            = { workload_parameters, 1 };
    result.launch_parameters              = { launch_parameters, 2 };
    result.compile_recipe.mode            = "direct";
    result.compile_recipe.primary_sources = { primary_sources, 1 };
    return result;
}

ggml::hrx::kernel_definition unsupported_definition() {
    static const char                          source_text[] = "test kernel";
    static ggml::hrx::kernel_source            source;
    static ggml::hrx::kernel_source_ref        primary_sources[1];
    static ggml::hrx::kernel_scalar_definition workload_parameters[1];
    static ggml::hrx::kernel_scalar_definition launch_parameters[2];
    source.source.data          = source_text;
    source.source.length        = sizeof(source_text) - 1;
    source.source.format        = ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT;
    source.dependencies         = nullptr;
    source.dependency_count     = 0;
    primary_sources[0].path     = "test.loom";
    primary_sources[0].contents = &source;
    workload_parameters[0].name = "rows";
    workload_parameters[0].type = "index";
    launch_parameters[0].name   = "rows";
    launch_parameters[0].type   = "i64";
    launch_parameters[1].name   = "columns";
    launch_parameters[1].type   = "index";

    ggml::hrx::kernel_definition result;
    result.family                         = "test_family";
    result.name                           = "test_kernel";
    result.id                             = ggml::hrx::kernel_catalog_id(result.family, result.name);
    result.source                         = "test.loom";
    result.symbol                         = "test_kernel";
    result.backend                        = "amdgpu";
    result.target_selector                = "gfx1151";
    result.workload_parameters            = { workload_parameters, 1 };
    result.launch_parameters              = { launch_parameters, 2 };
    result.compile_recipe.mode            = "direct";
    result.compile_recipe.primary_sources = { primary_sources, 1 };
    return result;
}

ggml::hrx::kernel_definition floating_definition() {
    static ggml::hrx::kernel_scalar_definition launch_parameters[2];
    launch_parameters[0].name = "scale";
    launch_parameters[0].type = "f32";
    launch_parameters[1].name = "bias";
    launch_parameters[1].type = "f32";

    ggml::hrx::kernel_definition result = definition();
    result.launch_parameters            = { launch_parameters, 2 };
    return result;
}

void test_constant_packing() {
    const ggml::hrx::kernel_definition kernel = definition();
    ggml::hrx::Command                 command;
    command.kernel.integer_parameters = {
        { "rows",    7                                    },
        { "columns", std::numeric_limits<uint32_t>::max() },
    };
    const ggml::hrx::packed_kernel_constants packed = ggml::hrx::pack_kernel_constants(kernel, command);
    REQUIRE(packed.valid());
    REQUIRE(packed.bytes.size() == 8);
    uint32_t rows    = 0;
    uint32_t columns = 0;
    std::memcpy(&rows, packed.bytes.data(), sizeof(rows));
    std::memcpy(&columns, packed.bytes.data() + sizeof(rows), sizeof(columns));
    REQUIRE(rows == 7);
    REQUIRE(columns == std::numeric_limits<uint32_t>::max());

    command.kernel.integer_parameters.erase("columns");
    REQUIRE(!ggml::hrx::pack_kernel_constants(kernel, command).valid());
    command.kernel.integer_parameters["columns"] = -1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(kernel, command).valid());
    command.kernel.integer_parameters["columns"] = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(kernel, command).valid());

    command.kernel.integer_parameters["columns"] = 1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(unsupported_definition(), command).valid());

    command.kernel.integer_parameters = {
        { "scale", 0x3f000000 },
        { "bias",  0xbf800000 },
    };
    const ggml::hrx::packed_kernel_constants floating =
        ggml::hrx::pack_kernel_constants(floating_definition(), command);
    REQUIRE(floating.valid());
    REQUIRE(floating.bytes.size() == 8);
    float scale = 0.0f;
    float bias  = 0.0f;
    std::memcpy(&scale, floating.bytes.data(), sizeof(scale));
    std::memcpy(&bias, floating.bytes.data() + sizeof(scale), sizeof(bias));
    REQUIRE(scale == 0.5f);
    REQUIRE(bias == -1.0f);

    command.kernel.integer_parameters["scale"] = -1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(floating_definition(), command).valid());
}

void test_artifact_key_uses_only_compilation_facts() {
    const ggml::hrx::kernel_definition kernel = definition();
    ggml::hrx::Command                 command;
    command.kernel.integer_parameters = {
        { "rows",    7  },
        { "columns", 32 },
        { "layer",   4  }
    };
    command.kernel.compile_parameters = {
        { "mode", "fast" }
    };
    const std::string first                    = ggml::hrx::kernel_artifact_key(kernel, command, "gfx1151");
    command.kernel.integer_parameters["layer"] = 47;
    REQUIRE(ggml::hrx::kernel_artifact_key(kernel, command, "gfx1151") == first);
    command.kernel.integer_parameters["rows"] = 8;
    REQUIRE(ggml::hrx::kernel_artifact_key(kernel, command, "gfx1151") != first);
    command.kernel.integer_parameters["rows"] = 7;
    command.kernel.compile_parameters["mode"] = "precise";
    REQUIRE(ggml::hrx::kernel_artifact_key(kernel, command, "gfx1151") != first);
}

void test_binding_diagnostics_are_explicit() {
    ggml::hrx::executable_bindings       bindings;
    ggml::hrx::executable_buffer_binding device_weight;
    device_weight.storage  = 3;
    device_weight.buffer   = reinterpret_cast<hrx_buffer_t>(uintptr_t{ 1 });
    device_weight.capacity = 4096;
    device_weight.length   = 2048;
    device_weight.weight   = true;
    bindings.storages.push_back(device_weight);
    ggml::hrx::executable_buffer_binding host_weight;
    host_weight.storage   = 4;
    host_weight.host_data = reinterpret_cast<void *>(uintptr_t{ 1 });
    host_weight.capacity  = 4096;
    host_weight.length    = 4096;
    host_weight.weight    = true;
    bindings.storages.push_back(host_weight);
    const std::string text = bindings.format();
    REQUIRE(text.find("borrowed_device_weight") != std::string::npos);
    REQUIRE(text.find("resident_host_weight") != std::string::npos);
    REQUIRE(text.find("layout=ggml-native") != std::string::npos);
    const std::string json = bindings.serialize_json();
    REQUIRE(json.find("borrowed_device_weight") != std::string::npos);
    REQUIRE(json.find("resident_host_weight") != std::string::npos);
}

void test_kernel_source_lookup() {
    const ggml::hrx::kernel_source * source =
        ggml::hrx::get_kernel_source("qwen_moe/ggml/linear_q6k_f32.loom");
    REQUIRE(source != nullptr);
    REQUIRE(source->source.data != nullptr);
    REQUIRE(source->source.length != 0);
    REQUIRE(source->source.format == ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT);
    REQUIRE(source->dependency_count == 2);
    REQUIRE(source->dependencies != nullptr);
    REQUIRE(source->dependencies[0].data != nullptr);
    REQUIRE(source->dependencies[0].length != 0);
    REQUIRE(source->dependencies[0].format == ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT);

    const ggml::hrx::kernel_source * dependency_only =
        ggml::hrx::get_kernel_source("qwen_moe/qwen3_moe/model_config.loom");
    REQUIRE(dependency_only != nullptr);
    REQUIRE(dependency_only->dependency_count == 0);
    REQUIRE(ggml::hrx::get_kernel_source("missing.loom") == nullptr);
}

void test_kernel_resolution_classifies_catalog_misses() {
    const ggml::hrx::kernel_definition kernel = definition();
    ggml::hrx::kernel_corpus           corpus;
    corpus.kernels = { &kernel, 1 };

    ggml::hrx::kernel_resolve_result resolved =
        ggml::hrx::resolve_kernel_definition(corpus, "gfx1151", "test_family", "test_kernel", kernel.id,
                                             ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.status == ggml::hrx::kernel_resolve_status::KERNEL_RESOLVE_STATUS_FOUND);
    REQUIRE(resolved.definition == &kernel);

    const uint64_t other_family_id = ggml::hrx::kernel_catalog_id("other_family", "test_kernel");
    REQUIRE(other_family_id != kernel.id);
    resolved = ggml::hrx::resolve_kernel_definition(corpus, "gfx1151", "other_family", "test_kernel", other_family_id,
                                                    ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.status == ggml::hrx::kernel_resolve_status::KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY);

    resolved = ggml::hrx::resolve_kernel_definition(corpus, "gfx1151", "test_family", "test_kernel",
                                                    ggml::hrx::GGML_HRX_KERNEL_ID_UNCATALOGED,
                                                    ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.status == ggml::hrx::kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNCATALOGED_NATIVE);

    resolved = ggml::hrx::resolve_kernel_definition(corpus, "gfx1151", "test_family", "missing_kernel",
                                                    ggml::hrx::kernel_catalog_id("test_family", "missing_kernel"),
                                                    ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.status == ggml::hrx::kernel_resolve_status::KERNEL_RESOLVE_STATUS_MISSING_ACTIVE_CORPUS_ENTRY);

    resolved = ggml::hrx::resolve_kernel_definition(corpus, "gfx1151", "test_family", "missing_kernel",
                                                    ggml::hrx::GGML_HRX_KERNEL_ID_UNCATALOGED,
                                                    ggml::hrx::KernelSpecialization::ExecutionKind::NativeGap);
    REQUIRE(resolved.status == ggml::hrx::kernel_resolve_status::KERNEL_RESOLVE_STATUS_NATIVE_GAP);

    ggml::hrx::kernel_definition default_variant  = kernel;
    default_variant.symbol                        = "test_kernel_default";
    default_variant.target_selector               = "";
    const ggml::hrx::kernel_definition variants[] = { default_variant, kernel };
    corpus.kernels                                = { variants, 2 };
    resolved = ggml::hrx::resolve_kernel_definition(corpus, "gfx1151", "test_family", "test_kernel", kernel.id,
                                                    ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.definition == &variants[1]);
    resolved = ggml::hrx::resolve_kernel_definition(corpus, "gfx1100", "test_family", "test_kernel", kernel.id,
                                                    ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.definition == &variants[0]);
    corpus.kernels = { &kernel, 1 };
    resolved       = ggml::hrx::resolve_kernel_definition(corpus, "gfx1100", "test_family", "test_kernel", kernel.id,
                                                          ggml::hrx::KernelSpecialization::ExecutionKind::Native);
    REQUIRE(resolved.status == ggml::hrx::kernel_resolve_status::KERNEL_RESOLVE_STATUS_UNSUPPORTED_TARGET);
}

}  // namespace

int main() {
    test_constant_packing();
    test_artifact_key_uses_only_compilation_facts();
    test_binding_diagnostics_are_explicit();
    test_kernel_source_lookup();
    test_kernel_resolution_classifies_catalog_misses();
    return 0;
}
