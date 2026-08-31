#pragma once

#include "graph/value-map.h"
#include "kernel-corpus/kernel-corpus-catalog.h"
#include "kernel-corpus/kernel-types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

inline constexpr const char kNativeWeightLayout[]       = "ggml-native";
inline constexpr const char kQ4KI4K32Row64Layout[]      = "q4k-i4-k32-row64";
inline constexpr const char kQ6KI8K32Row64Layout[]      = "q6k-i8-k32-row64";
inline constexpr const char kQ6KPackedK256Row64Layout[] = "q6k-packed-k256-row64";
inline constexpr const char kQ6KPackedScaleRowLayout[]  = "q6k-packed-k256-row64-scalerow";
inline constexpr const char kQ6KSymmetricI2PackedScaleRowLayout[] =
    "q6k-symi2-k32-eightgroups-shared4-plus-packed-scalerow";
inline constexpr const char kQ5KSymmetricI8K256Row64Layout[]          = "q5k-symi8-k256-row64";
inline constexpr const char kQ5KSymmetricI5K32Layout[]                = "q5k-symi5-k32-native-footprint";
inline constexpr const char kSymmetricI4K64Row64Layout[]              = "symi4-k64-row64";
inline constexpr const char kSymmetricI4K32Row64Layout[]              = "symi4-k32-row64";
inline constexpr const char kSymmetricI4K32EightGroupsShared4Layout[] = "symi4-k32-eightgroups-shared4-payload-first";

struct KernelSpecialization {
    uint64_t                           kernel_id = kUncatalogedKernelId;
    std::map<std::string, int64_t>     integer_parameters;
    std::map<std::string, std::string> compile_parameters;
};

inline KernelSpecialization make_kernel_specialization(KernelCatalogRef ref) {
    KernelSpecialization kernel;
    kernel.kernel_id = ref.id;
    return kernel;
}

struct DispatchBinding {
    ValueId value;
    size_t  offset = 0;
    size_t  length = 0;
    // Recipes may materialize a backend-local resident layout from canonical GGML weights.
    std::string layout        = kNativeWeightLayout;
    ggml_type   source_type   = GGML_TYPE_COUNT;
    int64_t     input_size    = 0;
    int64_t     output_size   = 0;
    size_t      source_length = 0;
};

struct Dispatch {
    KernelSpecialization         kernel;
    std::vector<DispatchBinding> bindings;
};

}  // namespace ggml::hrx
