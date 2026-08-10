#pragma once

#include <cstdint>

namespace ggml::hrx {

enum class ResourceAccess : uint8_t {
    Read,
    Write,
    ReadWrite,
};

}  // namespace ggml::hrx
