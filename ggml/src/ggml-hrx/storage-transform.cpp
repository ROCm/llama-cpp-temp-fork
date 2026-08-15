#include "storage-transform.h"

#include "../ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace ggml::hrx {
namespace {

bool checked_multiply(size_t lhs, size_t rhs, size_t & result) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

float fp16_to_fp32(uint16_t bits) {
    _Float16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

uint16_t fp32_to_fp16(float value) {
    const _Float16 half = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

bool q4_tensor_payload_header8_shape_valid(const kernel_storage_transform & transform) {
    return transform.outer_count == 1 && transform.row_group == 8 &&
           transform.row_count % transform.row_group == 0;
}

void pack_q4_tensor_payload_header8_row(const kernel_storage_transform & transform,
                                        size_t                           row,
                                        const uint8_t *                  canonical_row,
                                        uint8_t *                        destination) {
    static_assert(sizeof(block_q4_K) == 144);
    constexpr size_t header_bytes  = 16;
    constexpr size_t payload_bytes = sizeof(block_q4_K) - header_bytes;
    const size_t payload_row_bytes = transform.block_count * payload_bytes;
    const size_t payload_bytes_all = transform.row_count * payload_row_bytes;

    for (size_t block = 0; block < transform.block_count; ++block) {
        const uint8_t * canonical_block = canonical_row + block * sizeof(block_q4_K);
        std::memcpy(destination + row * payload_row_bytes + block * payload_bytes,
                    canonical_block + header_bytes, payload_bytes);

        const size_t row_group = row / transform.row_group;
        const size_t row_lane  = row % transform.row_group;
        const size_t header_index =
            ((row_group * transform.block_count + block) * transform.row_group + row_lane) * header_bytes;
        std::memcpy(destination + payload_bytes_all + header_index, canonical_block, header_bytes);
    }
}

struct q4_k_symmetric_i4_k64_block {
    std::array<uint16_t, 4>                scales;
    std::array<std::array<uint8_t, 16>, 8> payloads;
};

void decode_q4_k_scale_min(const uint8_t * header, size_t logical_group,
                           uint8_t & scale, uint8_t & minimum) {
    if (logical_group < 4) {
        scale   = header[4 + logical_group] & 0x3F;
        minimum = header[8 + logical_group] & 0x3F;
    } else {
        const size_t lane = logical_group - 4;
        scale = (header[12 + lane] & 0x0F) | ((header[4 + lane] >> 6) << 4);
        minimum = (header[12 + lane] >> 4) | ((header[8 + lane] >> 6) << 4);
    }
}

uint8_t decode_q4_k_code(const uint8_t * block, size_t logical_group, size_t element) {
    const size_t field_pair   = logical_group / 2;
    const size_t nibble_shift = (logical_group % 2) * 4;
    const size_t field        = 1 + field_pair * 2 + element / 16;
    return static_cast<uint8_t>((block[field * 16 + element % 16] >> nibble_shift) & 0x0F);
}

q4_k_symmetric_i4_k64_block quantize_q4_k_symmetric_i4_k64(const uint8_t * canonical) {
    q4_k_symmetric_i4_k64_block output = {};
    const uint8_t * header = canonical;
    const float d = fp16_to_fp32(
        static_cast<uint16_t>(header[0] | (static_cast<uint16_t>(header[1]) << 8)));
    const float dmin = fp16_to_fp32(
        static_cast<uint16_t>(header[2] | (static_cast<uint16_t>(header[3]) << 8)));

    for (size_t pair = 0; pair < output.scales.size(); ++pair) {
        float values[64];
        float positive_max = 0.0f;
        float negative_max = 0.0f;
        for (size_t half = 0; half < 2; ++half) {
            const size_t logical_group = pair * 2 + half;
            uint8_t qscale = 0;
            uint8_t qminimum = 0;
            decode_q4_k_scale_min(header, logical_group, qscale, qminimum);
            for (size_t element = 0; element < 32; ++element) {
                const float value = d * static_cast<float>(qscale) *
                                        static_cast<float>(decode_q4_k_code(canonical, logical_group, element)) -
                                    dmin * static_cast<float>(qminimum);
                values[half * 32 + element] = value;
                positive_max = std::max(positive_max, value);
                negative_max = std::max(negative_max, -value);
            }
        }

        float scale = std::max({positive_max / 7.0f, negative_max / 8.0f,
                                std::numeric_limits<float>::min()});
        for (int iteration = 0; iteration < 2; ++iteration) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (float value : values) {
                const int quantized = std::clamp(
                    static_cast<int>(std::nearbyint(value / scale)), -8, 7);
                numerator += static_cast<double>(value) * quantized;
                denominator += static_cast<double>(quantized) * quantized;
            }
            if (denominator > 0.0) {
                scale = static_cast<float>(numerator / denominator);
            }
        }
        output.scales[pair] = fp32_to_fp16(scale);
        scale = fp16_to_fp32(output.scales[pair]);

        for (size_t half = 0; half < 2; ++half) {
            const size_t logical_group = pair * 2 + half;
            for (size_t element_pair = 0; element_pair < 16; ++element_pair) {
                const int low = std::clamp(static_cast<int>(std::nearbyint(
                    values[half * 32 + element_pair * 2] / scale)), -8, 7);
                const int high = std::clamp(static_cast<int>(std::nearbyint(
                    values[half * 32 + element_pair * 2 + 1] / scale)), -8, 7);
                output.payloads[logical_group][element_pair] =
                    static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
            }
        }
    }
    return output;
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
    } else if (transform.kind == kernel_storage_transform_kind::Q4KRowGroupI4Interleave) {
        // Q4_K stores eight logical K32 groups as low/high nibbles spread
        // across four pairs of 16-byte payload fields. Keep the 16-byte
        // header intact and publish each logical K32 group as one packed-I4
        // field, interleaved across the row group for coalesced WMMA loads.
        if (transform.field_count != 9 || transform.unit_bytes != 16) {
            return false;
        }
        for (size_t outer = 0; outer < transform.outer_count; ++outer) {
            for (size_t group = 0; group < transform.row_count / transform.row_group; ++group) {
                for (size_t block = 0; block < transform.block_count; ++block) {
                    for (size_t lane = 0; lane < transform.row_group; ++lane) {
                        const size_t row = group * transform.row_group + lane;
                        const size_t header_offset =
                            (((outer * transform.row_count + row) * transform.block_count + block) *
                             transform.field_count * transform.unit_bytes);
                        std::memcpy(destination_bytes + packed_offset, source_bytes + header_offset,
                                    transform.unit_bytes);
                        packed_offset += transform.unit_bytes;
                    }
                    for (size_t logical_group = 0; logical_group < 8; ++logical_group) {
                        const size_t field_pair = logical_group / 2;
                        const size_t nibble_shift = (logical_group % 2) * 4;
                        for (size_t lane = 0; lane < transform.row_group; ++lane) {
                            const size_t row = group * transform.row_group + lane;
                            const size_t field0 = 1 + field_pair * 2;
                            const size_t field1 = field0 + 1;
                            const size_t field0_offset =
                                ((((outer * transform.row_count + row) * transform.block_count + block) *
                                  transform.field_count + field0) * transform.unit_bytes);
                            const size_t field1_offset =
                                ((((outer * transform.row_count + row) * transform.block_count + block) *
                                  transform.field_count + field1) * transform.unit_bytes);
                            const uint8_t * field0_source = source_bytes + field0_offset;
                            const uint8_t * field1_source = source_bytes + field1_offset;
                            for (size_t pair = 0; pair < 8; ++pair) {
                                const uint8_t low = (field0_source[pair * 2] >> nibble_shift) & 0x0F;
                                const uint8_t high = (field0_source[pair * 2 + 1] >> nibble_shift) & 0x0F;
                                destination_bytes[packed_offset + pair] = low | (high << 4);
                            }
                            for (size_t pair = 0; pair < 8; ++pair) {
                                const uint8_t low = (field1_source[pair * 2] >> nibble_shift) & 0x0F;
                                const uint8_t high = (field1_source[pair * 2 + 1] >> nibble_shift) & 0x0F;
                                destination_bytes[packed_offset + 8 + pair] = low | (high << 4);
                            }
                            packed_offset += transform.unit_bytes;
                        }
                    }
                }
            }
        }
    } else if (transform.kind == kernel_storage_transform_kind::Q4KRowGroupSymmetricI4Interleave) {
        // Capacity-neutral signed-I4 layout. Each adjacent K32 pair shares one
        // K64 scale; repeated f16 scale slots preserve the 144-byte Q4_K size.
        if (transform.field_count != 9 || transform.unit_bytes != 16) {
            return false;
        }
        for (size_t outer = 0; outer < transform.outer_count; ++outer) {
            for (size_t group = 0; group < transform.row_count / transform.row_group; ++group) {
                for (size_t block = 0; block < transform.block_count; ++block) {
                    const size_t output_group_base = packed_offset;
                    for (size_t lane = 0; lane < transform.row_group; ++lane) {
                        const size_t row = group * transform.row_group + lane;
                        const size_t canonical_offset =
                            (((outer * transform.row_count + row) * transform.block_count + block) *
                             transform.field_count * transform.unit_bytes);
                        const uint8_t * canonical = source_bytes + canonical_offset;
                        const q4_k_symmetric_i4_k64_block converted =
                            quantize_q4_k_symmetric_i4_k64(canonical);
                        uint8_t * output_header = destination_bytes + output_group_base + lane * 16;
                        std::memset(output_header, 0, 16);
                        for (size_t pair = 0; pair < converted.scales.size(); ++pair) {
                            const uint16_t scale_f16 = converted.scales[pair];
                            for (size_t repeat = 0; repeat < 2; ++repeat) {
                                const size_t scale_byte = (pair * 2 + repeat) * sizeof(scale_f16);
                                output_header[scale_byte] = static_cast<uint8_t>(scale_f16 & 0xFF);
                                output_header[scale_byte + 1] = static_cast<uint8_t>(scale_f16 >> 8);
                            }
                        }
                        for (size_t logical_group = 0; logical_group < converted.payloads.size();
                             ++logical_group) {
                            uint8_t * output_payload = destination_bytes + output_group_base +
                                (1 + logical_group) * transform.row_group * 16 + lane * 16;
                            std::memcpy(output_payload, converted.payloads[logical_group].data(), 16);
                        }
                    }
                    packed_offset += transform.row_group * transform.field_count * transform.unit_bytes;
                }
            }
        }
    } else if (transform.kind == kernel_storage_transform_kind::Q4KTensorPayloadHeader8) {
        if (!q4_tensor_payload_header8_shape_valid(transform) ||
            transform.field_count != 9 || transform.unit_bytes != 16) {
            return false;
        }
        const size_t canonical_row_bytes = transform.block_count * sizeof(block_q4_K);
        for (size_t row = 0; row < transform.row_count; ++row) {
            pack_q4_tensor_payload_header8_row(
                transform, row, source_bytes + row * canonical_row_bytes, destination_bytes);
        }
        packed_offset = expected_size;
    } else if (transform.kind == kernel_storage_transform_kind::Q4KTensorSymmetricI4K64) {
        // Capacity-preserving allocation containing a compact tensor-plane
        // prefix: 128 payload bytes and four unique K64 scales per source
        // Q4_K block. The unused tail remains zero and is never read.
        if (!q4_tensor_payload_header8_shape_valid(transform) ||
            transform.field_count != 9 || transform.unit_bytes != 16) {
            return false;
        }
        constexpr size_t payload_bytes = 128;
        constexpr size_t header_bytes  = 8;
        const size_t payload_row_bytes = transform.block_count * payload_bytes;
        const size_t payload_bytes_all = transform.row_count * payload_row_bytes;
        std::memset(destination_bytes, 0, expected_size);
        for (size_t row = 0; row < transform.row_count; ++row) {
            for (size_t block = 0; block < transform.block_count; ++block) {
                const size_t canonical_offset =
                    (row * transform.block_count + block) * sizeof(block_q4_K);
                const q4_k_symmetric_i4_k64_block converted =
                    quantize_q4_k_symmetric_i4_k64(source_bytes + canonical_offset);
                uint8_t * payload = destination_bytes +
                    row * payload_row_bytes + block * payload_bytes;
                for (size_t group = 0; group < converted.payloads.size(); ++group) {
                    std::memcpy(payload + group * 16, converted.payloads[group].data(), 16);
                }

                const size_t row_group = row / transform.row_group;
                const size_t row_lane  = row % transform.row_group;
                const size_t header_index =
                    ((row_group * transform.block_count + block) * transform.row_group + row_lane) *
                    header_bytes;
                uint8_t * header = destination_bytes + payload_bytes_all + header_index;
                for (size_t pair = 0; pair < converted.scales.size(); ++pair) {
                    header[pair * 2] = static_cast<uint8_t>(converted.scales[pair] & 0xFF);
                    header[pair * 2 + 1] = static_cast<uint8_t>(converted.scales[pair] >> 8);
                }
            }
        }
        packed_offset = expected_size;
    } else if (transform.kind == kernel_storage_transform_kind::Q6KToQ4KTensorPayloadHeader8) {
        if (!q4_tensor_payload_header8_shape_valid(transform) ||
            transform.field_count != 1 || transform.unit_bytes != sizeof(block_q6_K)) {
            return false;
        }
        constexpr size_t elements_per_block = QK_K;
        const size_t row_elements = transform.block_count * elements_per_block;
        size_t q4_block_count = transform.row_count;
        if (!checked_multiply(q4_block_count, transform.block_count, q4_block_count)) {
            return false;
        }
        size_t q4_size = 0;
        if (!checked_multiply(q4_block_count, sizeof(block_q4_K), q4_size) || q4_size > expected_size) {
            return false;
        }

        const size_t worker_count = std::min<size_t>(
            transform.row_count, std::max(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (size_t worker = 0; worker < worker_count; ++worker) {
            const size_t begin = transform.row_count * worker / worker_count;
            const size_t end   = transform.row_count * (worker + 1) / worker_count;
            workers.emplace_back([=]() {
                std::vector<float> row_values(row_elements);
                std::vector<block_q4_K> q4_row(transform.block_count);
                for (size_t row = begin; row < end; ++row) {
                    const auto * source_row = reinterpret_cast<const block_q6_K *>(
                        source_bytes + row * transform.block_count * sizeof(block_q6_K));
                    dequantize_row_q6_K(source_row, row_values.data(), static_cast<int64_t>(row_elements));
                    quantize_row_q4_K_ref(row_values.data(), q4_row.data(), static_cast<int64_t>(row_elements));
                    pack_q4_tensor_payload_header8_row(
                        transform, row, reinterpret_cast<const uint8_t *>(q4_row.data()), destination_bytes);
                }
            });
        }
        for (std::thread & worker : workers) {
            worker.join();
        }
        std::memset(destination_bytes + q4_size, 0, expected_size - q4_size);
        packed_offset = expected_size;
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
