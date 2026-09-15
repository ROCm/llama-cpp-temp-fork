#pragma once

#include <cstdint>

namespace ggml::hrx {

// Qualified Q6 endpoint schedule. Other shapes use the general matmul paths.
constexpr int64_t kQuantizedEndpointInputSize = 2048;
constexpr int64_t kQuantizedEndpointOutputSize = 151936;

constexpr bool is_quantized_endpoint_shape(int64_t input_size, int64_t output_size) {
    return input_size == kQuantizedEndpointInputSize && output_size == kQuantizedEndpointOutputSize;
}

}  // namespace ggml::hrx
