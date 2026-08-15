#include "hrx_runtime.h"
#include "../ggml/src/ggml-quants.h"
#include "storage-transform.h"
#include "transfer-manager.h"
#include "weight-residency.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static void check(hrx_status_t status, const char * operation) {
    if (hrx_status_is_ok(status)) {
        return;
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_to_string(status, &message, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", operation, static_cast<int>(length),
                 message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    std::abort();
}

static void test_kernel_storage_transforms() {
    const uint32_t field_order[] = { 2, 0, 1 };
    ggml::hrx::kernel_storage_transform interleave;
    interleave.kind        = ggml::hrx::kernel_storage_transform_kind::RowGroupFieldInterleave;
    interleave.outer_count = 1;
    interleave.row_count   = 4;
    interleave.block_count = 2;
    interleave.field_count = 3;
    interleave.unit_bytes  = 1;
    interleave.row_group   = 2;
    interleave.field_order = { field_order, 3 };
    const std::array<uint8_t, 24> canonical = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    };
    const std::array<uint8_t, 24> expected_interleave = {
        2, 8, 0, 6, 1, 7, 5, 11, 3, 9, 4, 10, 14, 20, 12, 18, 13, 19, 17, 23, 15, 21, 16, 22,
    };
    std::array<uint8_t, 24> packed = {};
    REQUIRE(ggml::hrx::kernel_storage_transform_size(interleave) == canonical.size());
    REQUIRE(ggml::hrx::kernel_storage_transform_pack(
        interleave, canonical.data(), canonical.size(), packed.data(), packed.size()));
    REQUIRE(packed == expected_interleave);

    ggml::hrx::kernel_storage_transform header_payload;
    header_payload.kind          = ggml::hrx::kernel_storage_transform_kind::RowGroupBlockGroupHeaderPayload;
    header_payload.outer_count   = 1;
    header_payload.row_count     = 2;
    header_payload.block_count   = 2;
    header_payload.field_count   = 3;
    header_payload.unit_bytes    = 1;
    header_payload.row_group     = 2;
    header_payload.block_group   = 2;
    header_payload.header_fields = 1;
    const std::array<uint8_t, 12> canonical_header_payload = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const std::array<uint8_t, 12> expected_header_payload  = { 0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11 };
    std::array<uint8_t, 12>       packed_header_payload    = {};
    REQUIRE(ggml::hrx::kernel_storage_transform_pack(
        header_payload, canonical_header_payload.data(), canonical_header_payload.size(),
        packed_header_payload.data(), packed_header_payload.size()));
    REQUIRE(packed_header_payload == expected_header_payload);

    ggml::hrx::kernel_storage_transform q4_i4;
    q4_i4.kind        = ggml::hrx::kernel_storage_transform_kind::Q4KRowGroupI4Interleave;
    q4_i4.outer_count = 1;
    q4_i4.row_count   = 2;
    q4_i4.block_count = 1;
    q4_i4.field_count = 9;
    q4_i4.unit_bytes  = 16;
    q4_i4.row_group   = 2;
    std::array<uint8_t, 288> canonical_q4 = {};
    for (size_t row = 0; row < 2; ++row) {
        for (size_t byte = 0; byte < 16; ++byte) {
            canonical_q4[(row * 9 + 0) * 16 + byte] = static_cast<uint8_t>(0x80 + row * 16 + byte);
            canonical_q4[(row * 9 + 1) * 16 + byte] = static_cast<uint8_t>((1 + row * 2) * 16 + byte);
            canonical_q4[(row * 9 + 2) * 16 + byte] = static_cast<uint8_t>((2 + row * 2) * 16 + byte);
        }
    }
    std::array<uint8_t, 288> packed_q4 = {};
    REQUIRE(ggml::hrx::kernel_storage_transform_pack(
        q4_i4, canonical_q4.data(), canonical_q4.size(), packed_q4.data(), packed_q4.size()));
    REQUIRE(std::equal(packed_q4.begin(), packed_q4.begin() + 16, canonical_q4.begin()));
    REQUIRE(std::equal(packed_q4.begin() + 16, packed_q4.begin() + 32, canonical_q4.begin() + 144));
    const std::array<uint8_t, 16> expected_low = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
    };
    REQUIRE(std::equal(packed_q4.begin() + 32, packed_q4.begin() + 48, expected_low.begin()));
    REQUIRE(std::equal(packed_q4.begin() + 48, packed_q4.begin() + 64, expected_low.begin()));
    REQUIRE(std::all_of(packed_q4.begin() + 64, packed_q4.begin() + 72,
                        [](uint8_t value) { return value == 0x11; }));
    REQUIRE(std::all_of(packed_q4.begin() + 72, packed_q4.begin() + 80,
                        [](uint8_t value) { return value == 0x22; }));
    REQUIRE(std::all_of(packed_q4.begin() + 80, packed_q4.begin() + 88,
                        [](uint8_t value) { return value == 0x33; }));
    REQUIRE(std::all_of(packed_q4.begin() + 88, packed_q4.begin() + 96,
                        [](uint8_t value) { return value == 0x44; }));

    ggml::hrx::kernel_storage_transform q4_tensor_plane;
    q4_tensor_plane.kind        = ggml::hrx::kernel_storage_transform_kind::Q4KTensorPayloadHeader8;
    q4_tensor_plane.outer_count = 1;
    q4_tensor_plane.row_count   = 8;
    q4_tensor_plane.block_count = 2;
    q4_tensor_plane.field_count = 9;
    q4_tensor_plane.unit_bytes  = 16;
    q4_tensor_plane.row_group   = 8;
    std::array<uint8_t, 8 * 2 * sizeof(block_q4_K)> canonical_q4_plane = {};
    for (size_t row = 0; row < q4_tensor_plane.row_count; ++row) {
        for (size_t block = 0; block < q4_tensor_plane.block_count; ++block) {
            const size_t base = (row * q4_tensor_plane.block_count + block) * sizeof(block_q4_K);
            for (size_t byte = 0; byte < sizeof(block_q4_K); ++byte) {
                canonical_q4_plane[base + byte] =
                    static_cast<uint8_t>((row * 37 + block * 19 + byte) & 0xff);
            }
        }
    }
    std::array<uint8_t, canonical_q4_plane.size()> expected_q4_plane = {};
    constexpr size_t q4_header_bytes  = 16;
    constexpr size_t q4_payload_bytes = sizeof(block_q4_K) - q4_header_bytes;
    const size_t payload_row_bytes = q4_tensor_plane.block_count * q4_payload_bytes;
    const size_t payload_bytes_all = q4_tensor_plane.row_count * payload_row_bytes;
    for (size_t row = 0; row < q4_tensor_plane.row_count; ++row) {
        for (size_t block = 0; block < q4_tensor_plane.block_count; ++block) {
            const size_t source = (row * q4_tensor_plane.block_count + block) * sizeof(block_q4_K);
            std::copy_n(canonical_q4_plane.begin() + source + q4_header_bytes, q4_payload_bytes,
                        expected_q4_plane.begin() + row * payload_row_bytes + block * q4_payload_bytes);
            std::copy_n(canonical_q4_plane.begin() + source, q4_header_bytes,
                        expected_q4_plane.begin() + payload_bytes_all +
                            (block * q4_tensor_plane.row_group + row) * q4_header_bytes);
        }
    }
    std::array<uint8_t, canonical_q4_plane.size()> packed_q4_plane = {};
    REQUIRE(ggml::hrx::kernel_storage_transform_pack(
        q4_tensor_plane, canonical_q4_plane.data(), canonical_q4_plane.size(),
        packed_q4_plane.data(), packed_q4_plane.size()));
    REQUIRE(packed_q4_plane == expected_q4_plane);

    constexpr size_t symmetric_rows   = 64;
    constexpr size_t symmetric_blocks = 2;
    std::array<float, symmetric_rows * symmetric_blocks * QK_K> symmetric_source_values = {};
    for (size_t i = 0; i < symmetric_source_values.size(); ++i) {
        symmetric_source_values[i] = static_cast<float>(
            (static_cast<int>((i * 13 + i / QK_K * 7) % 101) - 50) * 0.015625);
    }
    std::array<block_q4_K, symmetric_rows * symmetric_blocks> symmetric_source = {};
    quantize_row_q4_K_ref(symmetric_source_values.data(), symmetric_source.data(),
                          symmetric_source_values.size());

    ggml::hrx::kernel_storage_transform symmetric_row_group;
    symmetric_row_group.kind =
        ggml::hrx::kernel_storage_transform_kind::Q4KRowGroupSymmetricI4Interleave;
    symmetric_row_group.outer_count = 1;
    symmetric_row_group.row_count   = symmetric_rows;
    symmetric_row_group.block_count = symmetric_blocks;
    symmetric_row_group.field_count = 9;
    symmetric_row_group.unit_bytes  = 16;
    symmetric_row_group.row_group   = symmetric_rows;
    std::array<uint8_t, sizeof(symmetric_source)> packed_symmetric_row_group = {};
    REQUIRE(ggml::hrx::kernel_storage_transform_pack(
        symmetric_row_group, symmetric_source.data(), sizeof(symmetric_source),
        packed_symmetric_row_group.data(), packed_symmetric_row_group.size()));

    ggml::hrx::kernel_storage_transform symmetric_tensor_plane = symmetric_row_group;
    symmetric_tensor_plane.kind =
        ggml::hrx::kernel_storage_transform_kind::Q4KTensorSymmetricI4K64;
    symmetric_tensor_plane.row_group = 8;
    std::array<uint8_t, sizeof(symmetric_source)> packed_symmetric_tensor_plane = {};
    REQUIRE(ggml::hrx::kernel_storage_transform_pack(
        symmetric_tensor_plane, symmetric_source.data(), sizeof(symmetric_source),
        packed_symmetric_tensor_plane.data(), packed_symmetric_tensor_plane.size()));

    constexpr size_t symmetric_payload_bytes = 128;
    constexpr size_t symmetric_header_bytes  = 8;
    const size_t symmetric_payload_bytes_all =
        symmetric_rows * symmetric_blocks * symmetric_payload_bytes;
    for (size_t row = 0; row < symmetric_rows; ++row) {
        for (size_t block = 0; block < symmetric_blocks; ++block) {
            for (size_t group = 0; group < 8; ++group) {
                const size_t row_group_offset =
                    ((block * 9 + 1 + group) * symmetric_rows + row) * 16;
                const size_t tensor_plane_offset =
                    (row * symmetric_blocks + block) * symmetric_payload_bytes + group * 16;
                REQUIRE(std::equal(
                    packed_symmetric_row_group.begin() + row_group_offset,
                    packed_symmetric_row_group.begin() + row_group_offset + 16,
                    packed_symmetric_tensor_plane.begin() + tensor_plane_offset));
            }
            const size_t row_group_header = (block * 9 * symmetric_rows + row) * 16;
            const size_t tensor_plane_header = symmetric_payload_bytes_all +
                (((row / 8) * symmetric_blocks + block) * 8 + row % 8) * symmetric_header_bytes;
            for (size_t pair = 0; pair < 4; ++pair) {
                REQUIRE(std::equal(
                    packed_symmetric_row_group.begin() + row_group_header + pair * 4,
                    packed_symmetric_row_group.begin() + row_group_header + pair * 4 + 2,
                    packed_symmetric_tensor_plane.begin() + tensor_plane_header + pair * 2));
            }
        }
    }
    const size_t symmetric_useful_bytes = symmetric_payload_bytes_all +
        symmetric_rows * symmetric_blocks * symmetric_header_bytes;
    REQUIRE(std::all_of(
        packed_symmetric_tensor_plane.begin() + symmetric_useful_bytes,
        packed_symmetric_tensor_plane.end(), [](uint8_t value) { return value == 0; }));

    q4_tensor_plane.row_count = 7;
    REQUIRE(!ggml::hrx::kernel_storage_transform_pack(
        q4_tensor_plane, canonical_q4_plane.data(), canonical_q4_plane.size(),
        packed_q4_plane.data(), packed_q4_plane.size()));
}

int main() {
    test_kernel_storage_transforms();
    hrx_status_t initialize_status = hrx_gpu_initialize(0);
    if (!hrx_status_is_ok(initialize_status)) {
        REQUIRE(hrx_status_code(initialize_status) == HRX_STATUS_ALREADY_EXISTS);
        hrx_status_ignore(initialize_status);
    }

    hrx_device_t device = nullptr;
    check(hrx_gpu_device_get(0, &device), "get device");
    REQUIRE(device != nullptr);
    hrx_device_retain(device);

    hrx_stream_t consumer = nullptr;
    check(hrx_stream_create(device, 0, &consumer), "create consumer stream");
    hrx_buffer_t     first      = nullptr;
    hrx_buffer_t     second     = nullptr;
    constexpr size_t byte_count = 5 * 1024 + 37;
    check(hrx_buffer_allocate(consumer, byte_count, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &first),
          "allocate first buffer");
    check(hrx_buffer_allocate(consumer, byte_count, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &second),
          "allocate second buffer");

    {
        ggml::hrx::transfer_manager_options options;
        options.staging_page_size     = 1024;
        options.maximum_staging_bytes = 2 * 1024;
        ggml::hrx::transfer_manager transfers(device, options);
        REQUIRE(transfers.valid());

        std::vector<uint8_t> expected(byte_count);
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<uint8_t>((i * 29 + 7) & 0xff);
        }
        REQUIRE(transfers.upload(expected.data(), first, 0, expected.size()).empty());
        REQUIRE(transfers.join(consumer).empty());

        // A device copy is ordered after the consumer that consumed the upload,
        // then joined back into that consumer without a host synchronization.
        REQUIRE(transfers.copy(consumer, first, 0, second, 0, expected.size()).empty());
        REQUIRE(transfers.join(consumer).empty());

        std::vector<uint8_t> actual(byte_count, 0);
        REQUIRE(transfers.download(consumer, second, 0, actual.data(), actual.size()).empty());
        REQUIRE(actual == expected);

        // Reusing a binding on the transfer stream must be ordered after its
        // previous consumer. This is the repeated executable launch shape.
        const uint8_t old_pattern = 0x3c;
        check(hrx_stream_fill_buffer(consumer, first, 0, byte_count, &old_pattern, sizeof(old_pattern)),
              "record prior consumer write");
        REQUIRE(transfers.wait_for_producer(consumer).empty());
        for (uint8_t & value : expected) {
            value ^= 0x5a;
        }
        REQUIRE(transfers.upload(expected.data(), first, 0, expected.size()).empty());
        REQUIRE(transfers.join(consumer).empty());
        REQUIRE(transfers.download(consumer, first, 0, actual.data(), actual.size()).empty());
        REQUIRE(actual == expected);

        const uint8_t pattern = 0xa5;
        REQUIRE(transfers.fill(second, 0, byte_count, &pattern, sizeof(pattern)).empty());
        REQUIRE(transfers.join(consumer).empty());
        REQUIRE(transfers.download(consumer, second, 0, actual.data(), actual.size()).empty());
        REQUIRE(std::all_of(actual.begin(), actual.end(), [](uint8_t value) { return value == 0xa5; }));

        const ggml::hrx::transfer_manager_stats stats = transfers.stats();
        REQUIRE(stats.uploads == 2);
        REQUIRE(stats.downloads == 3);
        REQUIRE(stats.uploaded_bytes == 2 * byte_count);
        REQUIRE(stats.downloaded_bytes == 3 * byte_count);
        REQUIRE(stats.page_allocations == 2);
        REQUIRE(stats.page_reuses > 0);
        REQUIRE(stats.backpressure_waits > 0);
        REQUIRE(stats.consumer_waits >= 3);
        REQUIRE(stats.producer_waits >= 2);
        REQUIRE(stats.staging_bytes == 2 * 1024);

        ggml::hrx::weight_residency_cache weights(device);
        REQUIRE(weights.valid());
        ggml::hrx::weight_source source;
        source.host_data       = expected.data();
        source.buffer_identity = 17;
        source.generation      = 3;
        source.capacity        = expected.size();
        source.length          = expected.size();
        auto first_weight      = weights.acquire(consumer, transfers, source);
        REQUIRE(first_weight.valid());
        auto second_weight = weights.acquire(consumer, transfers, source);
        REQUIRE(second_weight.valid());
        REQUIRE(first_weight.lease.buffer() == second_weight.lease.buffer());
        ggml::hrx::weight_residency_stats weight_stats = weights.stats();
        REQUIRE(weight_stats.misses == 1);
        REQUIRE(weight_stats.hits == 1);
        REQUIRE(weight_stats.allocation_count == 1);
        REQUIRE(weight_stats.resident_bytes == expected.size());
        source.layout = "incompatible-layout";
        auto conflict = weights.acquire(consumer, transfers, source);
        REQUIRE(!conflict.valid());
        REQUIRE(weights.stats().layout_conflicts == 1);
        REQUIRE(transfers.synchronize().empty());
    }

    hrx_buffer_release(second);
    hrx_buffer_release(first);
    hrx_stream_release(consumer);
    hrx_device_release(device);
    hrx_status_t shutdown_status = hrx_gpu_shutdown();
    if (!hrx_status_is_ok(shutdown_status)) {
        hrx_status_ignore(shutdown_status);
    }
    return 0;
}
