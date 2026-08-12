#include "storage-transform.h"

#include <cstring>
#include <limits>

namespace ggml::hrx {
namespace {

bool checked_multiply(size_t lhs, size_t rhs, size_t & result) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

}  // namespace

size_t kernel_storage_transform_size(const kernel_storage_transform & transform) {
    size_t result = transform.outer_count;
    if (!checked_multiply(result, transform.row_count, result) ||
        !checked_multiply(result, transform.block_count, result) ||
        !checked_multiply(result, transform.field_count, result) ||
        !checked_multiply(result, transform.unit_bytes, result)) {
        return 0;
    }
    return result;
}

bool kernel_storage_transform_pack(const kernel_storage_transform & transform,
                                   const void *                     source,
                                   size_t                           source_size,
                                   void *                           destination,
                                   size_t                           destination_size) {
    const size_t expected_size = kernel_storage_transform_size(transform);
    if (source == nullptr || destination == nullptr || source == destination || expected_size == 0 ||
        source_size != expected_size || destination_size != expected_size || transform.row_group == 0 ||
        transform.row_count % transform.row_group != 0) {
        return false;
    }
    const auto * source_bytes      = static_cast<const uint8_t *>(source);
    auto *       destination_bytes = static_cast<uint8_t *>(destination);
    size_t       packed_offset     = 0;

    if (transform.kind == kernel_storage_transform_kind::RowGroupFieldInterleave) {
        if (transform.field_order.count != transform.field_count || transform.field_order.items == nullptr) {
            return false;
        }
        for (size_t outer = 0; outer < transform.outer_count; ++outer) {
            for (size_t group = 0; group < transform.row_count / transform.row_group; ++group) {
                for (size_t block = 0; block < transform.block_count; ++block) {
                    for (size_t component = 0; component < transform.field_count; ++component) {
                        const size_t field = transform.field_order[component];
                        if (field >= transform.field_count) {
                            return false;
                        }
                        for (size_t lane = 0; lane < transform.row_group; ++lane) {
                            const size_t row = group * transform.row_group + lane;
                            const size_t canonical_offset =
                                ((((outer * transform.row_count + row) * transform.block_count + block) *
                                  transform.field_count + field) * transform.unit_bytes);
                            std::memcpy(destination_bytes + packed_offset, source_bytes + canonical_offset,
                                        transform.unit_bytes);
                            packed_offset += transform.unit_bytes;
                        }
                    }
                }
            }
        }
    } else if (transform.kind == kernel_storage_transform_kind::RowGroupBlockGroupHeaderPayload) {
        if (transform.block_group == 0 || transform.block_count % transform.block_group != 0 ||
            transform.header_fields == 0 || transform.header_fields >= transform.field_count) {
            return false;
        }
        const size_t header_bytes = transform.header_fields * transform.unit_bytes;
        const size_t payload_bytes = (transform.field_count - transform.header_fields) * transform.unit_bytes;
        for (size_t outer = 0; outer < transform.outer_count; ++outer) {
            for (size_t row_group = 0; row_group < transform.row_count / transform.row_group; ++row_group) {
                for (size_t block_group = 0; block_group < transform.block_count / transform.block_group;
                     ++block_group) {
                    for (size_t lane = 0; lane < transform.row_group; ++lane) {
                        const size_t row = row_group * transform.row_group + lane;
                        for (size_t block_lane = 0; block_lane < transform.block_group; ++block_lane) {
                            const size_t block = block_group * transform.block_group + block_lane;
                            const size_t canonical_offset =
                                (((outer * transform.row_count + row) * transform.block_count + block) *
                                 transform.field_count * transform.unit_bytes);
                            std::memcpy(destination_bytes + packed_offset, source_bytes + canonical_offset, header_bytes);
                            packed_offset += header_bytes;
                        }
                    }
                    for (size_t lane = 0; lane < transform.row_group; ++lane) {
                        const size_t row = row_group * transform.row_group + lane;
                        for (size_t block_lane = 0; block_lane < transform.block_group; ++block_lane) {
                            const size_t block = block_group * transform.block_group + block_lane;
                            const size_t canonical_offset =
                                ((((outer * transform.row_count + row) * transform.block_count + block) *
                                  transform.field_count + transform.header_fields) * transform.unit_bytes);
                            std::memcpy(destination_bytes + packed_offset, source_bytes + canonical_offset, payload_bytes);
                            packed_offset += payload_bytes;
                        }
                    }
                }
            }
        }
    } else {
        return false;
    }
    return packed_offset == expected_size;
}

}  // namespace ggml::hrx
