#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ggml::hrx {

enum class kernel_storage_transform_kind : uint8_t {
    RowGroupFieldInterleave,
    RowGroupBlockGroupHeaderPayload,
};

struct kernel_storage_transform_field_order {
    const uint32_t * items = nullptr;
    size_t           count = 0;

    const uint32_t & operator[](size_t index) const { return items[index]; }
};

// Capacity-preserving physical weight layout declared by a kernel corpus.
struct kernel_storage_transform {
    const char *                       name            = "";
    const char *                       target_selector = "";
    int32_t                            type            = -1;
    std::array<int64_t, 4>             shape           = {};
    bool                               contiguous      = false;
    const char *                       name_prefix     = "";
    const char *                       name_suffix     = "";
    bool                               decimal_middle  = false;
    kernel_storage_transform_kind kind          = kernel_storage_transform_kind::RowGroupFieldInterleave;
    size_t                        outer_count   = 0;
    size_t                        row_count     = 0;
    size_t                        block_count   = 0;
    size_t                        field_count   = 0;
    size_t                        unit_bytes    = 0;
    size_t                        row_group     = 0;
    size_t                        block_group   = 0;
    size_t                        header_fields = 0;
    kernel_storage_transform_field_order field_order;
};

size_t kernel_storage_transform_size(const kernel_storage_transform & transform);
bool kernel_storage_transform_pack(const kernel_storage_transform & transform,
                                   const void *                     source,
                                   size_t                           source_size,
                                   void *                           destination,
                                   size_t                           destination_size);

}  // namespace ggml::hrx
