#include "hrx_runtime.h"
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
