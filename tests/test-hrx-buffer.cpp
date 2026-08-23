#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "hrx-interop-utils.h"
#include "runtime/host-memory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static void require_hrx_status(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        std::abort();
    }
}

static ggml_backend_hrx_context * backend_context(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(context != nullptr);
    REQUIRE(context->device != nullptr);
    REQUIRE(context->device->device != nullptr);
    REQUIRE(context->stream != nullptr);
    return context;
}

static void run_backend_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * hrx = backend_context(backend);

    ggml_init_params params = {};
    params.mem_size         = 16 * 1024;
    params.no_alloc         = true;
    ggml_context * context  = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor *         tensor      = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_tensor *         copy        = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t buffer      = ggml_backend_alloc_buffer(backend, 4096);
    ggml_backend_buffer_t copy_buffer = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(buffer != nullptr);
    REQUIRE(copy_buffer != nullptr);
    tensor->buffer = buffer;
    tensor->data   = ggml_backend_buffer_get_base(buffer);
    copy->buffer   = copy_buffer;
    copy->data     = ggml_backend_buffer_get_base(copy_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, tensor) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_buffer_init_tensor(copy_buffer, copy) == GGML_STATUS_SUCCESS);

    std::array<uint32_t, 64> input = {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint32_t>(i * 17 + 3);
    }
    ggml_backend_tensor_set(tensor, input.data(), 0, sizeof(input));
    std::array<uint32_t, 64> output = {};
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    REQUIRE(output == input);
    ggml_backend_tensor_copy(tensor, copy);
    output.fill(0);
    ggml_backend_tensor_get(copy, output.data(), 0, sizeof(output));
    REQUIRE(output == input);

    input[0] = 0x12345678;
    ggml_backend_tensor_set_async(backend, tensor, input.data(), 0, sizeof(input));
    ggml_backend_synchronize(backend);
    output.fill(0);
    ggml_backend_tensor_get_async(backend, tensor, output.data(), 0, sizeof(output));
    ggml_backend_synchronize(backend);
    REQUIRE(output == input);
    REQUIRE(hrx->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed) == 1);
    REQUIRE(hrx->device->synchronous_download_fallbacks.load(std::memory_order_relaxed) == 1);

    ggml_backend_tensor_memset(tensor, 0x5a, 16, 32);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(output.data());
    for (size_t i = 16; i < 48; ++i) {
        REQUIRE(bytes[i] == 0x5a);
    }

    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    for (uint32_t value : output) {
        REQUIRE(value == 0);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(copy_buffer);
    ggml_free(context);
    ggml_backend_synchronize(backend);
    REQUIRE(hrx->device->device != nullptr);
}

static void run_host_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * context = backend_context(backend);
    ggml_backend_buffer_type_t buft    = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    REQUIRE(buft != nullptr);
    REQUIRE(ggml_backend_buft_is_host(buft));

    const bool original_direct_host_bindings = context->device->use_direct_host_bindings;
    context->device->use_direct_host_bindings = false;
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    context->device->use_direct_host_bindings = original_direct_host_bindings;
    REQUIRE(buffer != nullptr);
    REQUIRE(ggml_backend_buffer_is_host(buffer));
    auto * buffer_context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
    REQUIRE(buffer_context != nullptr);
    REQUIRE(buffer_context->buffer != nullptr);
    REQUIRE(buffer_context->base == ggml_backend_buffer_get_base(buffer));
    REQUIRE(!buffer_context->direct_host_binding);

    const uint32_t pattern = 0x12345678;
    require_hrx_status(
        hrx_stream_fill_buffer(context->stream, buffer_context->buffer, 0, 4096, &pattern, sizeof(pattern)));
    require_hrx_status(hrx_stream_synchronize(context->stream));
    const auto * words = static_cast<const uint32_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 4096 / sizeof(uint32_t); ++i) {
        REQUIRE(words[i] == pattern);
    }

    ggml_init_params params = {};
    params.mem_size         = 4096;
    params.no_alloc         = true;
    ggml_context * ggml     = ggml_init(params);
    REQUIRE(ggml != nullptr);
    ggml_tensor * host_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    host_tensor->buffer       = buffer;
    host_tensor->data         = ggml_backend_buffer_get_base(buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, host_tensor) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding staged_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(host_tensor, staged_binding));
    REQUIRE(staged_binding.buffer == nullptr);
    REQUIRE(staged_binding.host_data == ggml_backend_buffer_get_base(buffer));
    REQUIRE(staged_binding.offset == 0);
    REQUIRE(staged_binding.length == ggml_nbytes(host_tensor));

    context->device->use_direct_host_bindings = true;
    ggml_backend_buffer_t direct_buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    context->device->use_direct_host_bindings = original_direct_host_bindings;
    REQUIRE(direct_buffer != nullptr);
    auto * direct_buffer_context = ggml_backend_hrx_buffer_context_from_buffer(direct_buffer);
    REQUIRE(direct_buffer_context->direct_host_binding);
    ggml_tensor * direct_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    direct_tensor->buffer       = direct_buffer;
    direct_tensor->data         = ggml_backend_buffer_get_base(direct_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(direct_buffer, direct_tensor) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding direct_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(direct_tensor, direct_binding));
    REQUIRE(direct_binding.buffer == direct_buffer_context->buffer);
    REQUIRE(direct_binding.host_data == nullptr);
    REQUIRE(direct_binding.offset == 0);
    REQUIRE(direct_binding.length == ggml_nbytes(direct_tensor));

    ggml_tensor *         tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t local  = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(local != nullptr);
    tensor->buffer = local;
    tensor->data   = ggml_backend_buffer_get_base(local);
    REQUIRE(ggml_backend_buffer_init_tensor(local, tensor) == GGML_STATUS_SUCCESS);

    const uint64_t upload_fallbacks =
        context->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed);
    const uint64_t download_fallbacks =
        context->device->synchronous_download_fallbacks.load(std::memory_order_relaxed);
    auto * host_words = static_cast<uint32_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 64; ++i) {
        host_words[i] = static_cast<uint32_t>(i * 13 + 7);
    }
    ggml_backend_tensor_set_async(backend, tensor, host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_synchronize(backend);
    std::memset(host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_tensor_get_async(backend, tensor, host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_synchronize(backend);
    for (size_t i = 0; i < 64; ++i) {
        REQUIRE(host_words[i] == static_cast<uint32_t>(i * 13 + 7));
    }
    REQUIRE(context->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed) == upload_fallbacks);
    REQUIRE(context->device->synchronous_download_fallbacks.load(std::memory_order_relaxed) == download_fallbacks);

    ggml_backend_buffer_free(local);
    ggml_backend_buffer_free(direct_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
}

static void run_host_transfer_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostStagingBuffer   staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 64, staging).success());

    const std::array<uint8_t, 64> zero = {};
    require_hrx_status(hrx_synchronous_h2d(context->device->device, zero.data(), staging.buffer, 0, zero.size()));

    std::array<uint8_t, 64> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i + 1);
    }

    REQUIRE(transfers.upload_synchronous(context->stream, host.data(), staging.buffer, 0, 0).success());
    REQUIRE(transfers.upload_synchronous(context->stream, host.data() + 8, staging.buffer, 16, 24).success());
    ggml::hrx::HostTransferStats stats = transfers.stats();
    REQUIRE(stats.uploads == 1);
    REQUIRE(stats.upload_bytes == 24);
    require_hrx_status(hrx_stream_synchronize(context->stream));

    std::array<uint8_t, 64> upload_result = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, staging.buffer, 0, upload_result.data(), upload_result.size()));
    for (size_t i = 0; i < upload_result.size(); ++i) {
        const uint8_t expected = i >= 16 && i < 40 ? host[i - 8] : 0;
        REQUIRE(upload_result[i] == expected);
    }

    std::array<uint8_t, 64> device_values = {};
    for (size_t i = 0; i < device_values.size(); ++i) {
        device_values[i] = static_cast<uint8_t>(0xa0 + i);
    }
    require_hrx_status(
        hrx_synchronous_h2d(context->device->device, device_values.data(), staging.buffer, 0, device_values.size()));

    std::array<uint8_t, 48> download_result = {};
    REQUIRE(transfers.download_synchronous(
        context->stream, staging.buffer, 12, download_result.data() + 4, 20).success());
    stats = transfers.stats();
    REQUIRE(stats.downloads == 1);
    REQUIRE(stats.download_bytes == 20);
    require_hrx_status(hrx_stream_synchronize(context->stream));
    for (size_t i = 0; i < download_result.size(); ++i) {
        const uint8_t expected = i >= 4 && i < 24 ? device_values[i + 8] : 0;
        REQUIRE(download_result[i] == expected);
    }

    REQUIRE(!transfers.upload_synchronous(nullptr, host.data(), staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload_synchronous(context->stream, nullptr, staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload_synchronous(context->stream, host.data(), nullptr, 0, 4).success());
    REQUIRE(!transfers.download_synchronous(nullptr, staging.buffer, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download_synchronous(context->stream, nullptr, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download_synchronous(context->stream, staging.buffer, 0, nullptr, 4).success());

    transfers.clear();
    stats = transfers.stats();
    REQUIRE(stats.uploads == 0);
    REQUIRE(stats.downloads == 0);
    REQUIRE(stats.upload_bytes == 0);
    REQUIRE(stats.download_bytes == 0);
}

static void run_host_staging_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostStagingBuffer staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 32, staging).success());
    REQUIRE(staging.buffer != nullptr);
    REQUIRE(staging.length == 32);

    hrx_buffer_t                 original = staging.buffer;
    ggml::hrx::HostStagingBuffer moved(std::move(staging));
    REQUIRE(moved.buffer == original);
    REQUIRE(moved.length == 32);
    REQUIRE(staging.buffer == nullptr);
    REQUIRE(staging.length == 0);

    ggml::hrx::HostStagingBuffer assigned;
    assigned = std::move(moved);
    REQUIRE(assigned.buffer == original);
    REQUIRE(assigned.length == 32);
    REQUIRE(moved.buffer == nullptr);
    REQUIRE(moved.length == 0);

    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
    REQUIRE(assigned.length == 0);
    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
}

static void run_host_weight_cache_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache     weights;

    std::array<uint8_t, 128> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i);
    }

    ggml::hrx::HostWeightSource source;
    source.host_data  = host.data();
    source.identity   = 0x1234;
    source.generation = 1;
    source.capacity   = host.size();
    source.offset     = 16;
    source.length     = 32;
    source.layout     = "ggml-native";

    ggml::hrx::HostWeightAcquireResult first =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(first.valid());

    std::array<uint8_t, 32> first_bytes = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, first.lease.buffer(), 0, first_bytes.data(), first_bytes.size()));
    for (size_t i = 0; i < first_bytes.size(); ++i) {
        REQUIRE(first_bytes[i] == host[source.offset + i]);
    }

    ggml::hrx::HostWeightAcquireResult second =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(second.valid());
    REQUIRE(second.lease.buffer() == first.lease.buffer());

    source.offset = 32;
    ggml::hrx::HostWeightAcquireResult slice =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(slice.valid());
    REQUIRE(slice.lease.buffer() != first.lease.buffer());

    source.offset     = 16;
    source.generation = 2;
    ggml::hrx::HostWeightAcquireResult next_generation =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(next_generation.valid());
    REQUIRE(next_generation.lease.buffer() != first.lease.buffer());

    source.generation = 1;
    source.layout     = "alternate-layout";
    ggml::hrx::HostWeightAcquireResult conflict =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(!conflict.valid());

    ggml::hrx::HostWeightCacheStats weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 1);
    REQUIRE(weight_stats.misses == 3);
    REQUIRE(weight_stats.allocation_count == 3);
    REQUIRE(weight_stats.resident_bytes == 96);

    const ggml::hrx::HostTransferStats transfer_stats = transfers.stats();
    REQUIRE(transfer_stats.uploads == 3);
    REQUIRE(transfer_stats.upload_bytes == 96);

    weights.clear();
    weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 0);
    REQUIRE(weight_stats.misses == 0);
    REQUIRE(weight_stats.allocation_count == 0);
    REQUIRE(weight_stats.resident_bytes == 0);
}

static std::vector<uint8_t> materialize_test_weight(ggml_backend_hrx_context *   context,
                                                    ggml_type                    source_type,
                                                    const char *                 layout,
                                                    int64_t                      input_size,
                                                    int64_t                      output_size,
                                                    size_t                       materialized_length,
                                                    const std::vector<uint8_t> & canonical) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache     weights;
    ggml::hrx::HostWeightSource    source;
    source.host_data           = canonical.data();
    source.identity            = UINT64_C(0x12340000) + static_cast<uint64_t>(source_type);
    source.generation          = 1;
    source.capacity            = canonical.size();
    source.length              = canonical.size();
    source.materialized_length = materialized_length;
    source.layout              = layout;
    source.source_type         = source_type;
    source.input_size          = input_size;
    source.output_size         = output_size;

    ggml::hrx::HostWeightAcquireResult result =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(result.valid());
    REQUIRE(result.lease.length() == materialized_length);
    std::vector<uint8_t> materialized(materialized_length);
    require_hrx_status(hrx_synchronous_d2h(context->device->device, result.lease.buffer(), 0, materialized.data(),
                                           materialized.size()));
    return materialized;
}

static void set_first_block_scale(std::vector<uint8_t> & canonical, ggml_type type, float scale) {
    const ggml_fp16_t encoded_scale = ggml_fp32_to_fp16(scale);
    size_t            scale_offset  = 0;
    switch (type) {
        case GGML_TYPE_Q4_K:
            scale_offset = offsetof(block_q4_K, d);
            break;
        case GGML_TYPE_Q5_K:
            scale_offset = offsetof(block_q5_K, d);
            break;
        case GGML_TYPE_Q6_K:
            scale_offset = offsetof(block_q6_K, d);
            break;
        case GGML_TYPE_IQ4_XS:
            scale_offset = offsetof(block_iq4_xs, d);
            break;
        default:
            REQUIRE(false);
    }
    REQUIRE(scale_offset + sizeof(encoded_scale) <= canonical.size());
    std::memcpy(canonical.data() + scale_offset, &encoded_scale, sizeof(encoded_scale));
}

static void require_test_weight_rejected(ggml_backend_hrx_context *   context,
                                         ggml_type                    source_type,
                                         const char *                 layout,
                                         int64_t                      input_size,
                                         int64_t                      output_size,
                                         size_t                       materialized_length,
                                         const std::vector<uint8_t> & canonical) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache     weights;
    ggml::hrx::HostWeightSource    source;
    source.host_data           = canonical.data();
    source.identity            = UINT64_C(0x56780000) + static_cast<uint64_t>(source_type);
    source.generation          = 1;
    source.capacity            = canonical.size();
    source.length              = canonical.size();
    source.materialized_length = materialized_length;
    source.layout              = layout;
    source.source_type         = source_type;
    source.input_size          = input_size;
    source.output_size         = output_size;

    const ggml::hrx::HostWeightAcquireResult result =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(!result.valid());
    REQUIRE(!result.status.success());
    REQUIRE(weights.stats().allocation_count == 0);
    REQUIRE(transfers.stats().uploads == 0);
}

static void run_zero_symmetric_weight_materialization_checks(ggml_backend_hrx_context * context) {
    constexpr int64_t input_size = 256;
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS }) {
        constexpr int64_t          output_size = 64;
        const std::vector<uint8_t> canonical(ggml_row_size(type, input_size) * output_size, uint8_t{ 0 });
        const std::vector<uint8_t> materialized =
            materialize_test_weight(context, type, ggml::hrx::kSymmetricI4K64Row64Layout, input_size, output_size,
                                    output_size * 144, canonical);
        REQUIRE(std::all_of(materialized.begin(), materialized.end(), [](uint8_t value) { return value == 0; }));
    }

    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS }) {
        constexpr int64_t          output_size = 4;
        const std::vector<uint8_t> canonical(ggml_row_size(type, input_size) * output_size, uint8_t{ 0 });
        const std::vector<uint8_t> materialized =
            materialize_test_weight(context, type, ggml::hrx::kSymmetricI4K32EightGroupsShared4Layout, input_size,
                                    output_size, 32 * 132, canonical);
        REQUIRE(std::all_of(materialized.begin(), materialized.end(), [](uint8_t value) { return value == 0; }));
    }

    constexpr int64_t          output_size = 64;
    const std::vector<uint8_t> canonical(ggml_row_size(GGML_TYPE_Q6_K, input_size) * output_size, uint8_t{ 0 });
    const std::vector<uint8_t> materialized =
        materialize_test_weight(context, GGML_TYPE_Q6_K, ggml::hrx::kQ6KSymmetricI2PackedScaleRowLayout, input_size,
                                output_size, output_size * 68 + output_size * 210, canonical);
    REQUIRE(std::all_of(materialized.begin(), materialized.end(), [](uint8_t value) { return value == 0; }));
}

static void run_nonfinite_symmetric_weight_materialization_checks(ggml_backend_hrx_context * context) {
    constexpr int64_t input_size = 256;
    for (float invalid_scale : { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity() }) {
        for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS }) {
            constexpr int64_t    output_size = 64;
            std::vector<uint8_t> canonical(ggml_row_size(type, input_size) * output_size, uint8_t{ 0 });
            set_first_block_scale(canonical, type, invalid_scale);
            require_test_weight_rejected(context, type, ggml::hrx::kSymmetricI4K64Row64Layout, input_size, output_size,
                                         output_size * 144, canonical);
        }

        for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS }) {
            constexpr int64_t    output_size = 4;
            std::vector<uint8_t> canonical(ggml_row_size(type, input_size) * output_size, uint8_t{ 0 });
            set_first_block_scale(canonical, type, invalid_scale);
            require_test_weight_rejected(context, type, ggml::hrx::kSymmetricI4K32EightGroupsShared4Layout, input_size,
                                         output_size, 32 * 132, canonical);
        }

        constexpr int64_t    output_size = 64;
        std::vector<uint8_t> canonical(ggml_row_size(GGML_TYPE_Q6_K, input_size) * output_size, uint8_t{ 0 });
        set_first_block_scale(canonical, GGML_TYPE_Q6_K, invalid_scale);
        require_test_weight_rejected(context, GGML_TYPE_Q6_K, ggml::hrx::kQ6KSymmetricI2PackedScaleRowLayout,
                                     input_size, output_size, output_size * 68 + output_size * 210, canonical);
    }
}

static std::vector<uint8_t> make_iq4_xs_scale_case(int64_t output_size,
                                                   float   block_scale,
                                                   int     scale_code,
                                                   uint8_t quant_index) {
    constexpr int64_t input_size = 256;
    REQUIRE(scale_code >= 0 && scale_code <= 63);
    REQUIRE(quant_index <= 15);
    std::vector<uint8_t> canonical(ggml_row_size(GGML_TYPE_IQ4_XS, input_size) * output_size, uint8_t{ 0 });
    block_iq4_xs         block = {};
    block.d                    = ggml_fp32_to_fp16(block_scale);
    block.scales_h             = UINT16_C(0xAAAA);  // All eight group scale codes start at 32 (zero multiplier).
    block.scales_l[0]          = static_cast<uint8_t>(scale_code & 0x0F);
    block.scales_h =
        static_cast<uint16_t>((block.scales_h & ~UINT16_C(0x0003)) | static_cast<uint16_t>((scale_code >> 4) & 0x03));
    std::fill_n(block.qs, 16, static_cast<uint8_t>(quant_index | (quant_index << 4)));
    std::memcpy(canonical.data(), &block, sizeof(block));
    return canonical;
}

static std::vector<uint8_t> make_q6_k_scale_case(int64_t output_size,
                                                 float   block_scale,
                                                 int8_t  group_scale,
                                                 int     quantized_value) {
    constexpr int64_t input_size = 256;
    REQUIRE(quantized_value >= -32 && quantized_value <= 31);
    std::vector<uint8_t> canonical(ggml_row_size(GGML_TYPE_Q6_K, input_size) * output_size, uint8_t{ 0 });
    block_q6_K           block = {};
    block.d                    = ggml_fp32_to_fp16(block_scale);
    block.scales[0]            = group_scale;
    block.scales[1]            = group_scale;
    const uint8_t encoded      = static_cast<uint8_t>(quantized_value + 32);
    for (size_t element = 0; element < 32; ++element) {
        block.ql[element] = static_cast<uint8_t>(encoded & 0x0F);
        block.qh[element] = static_cast<uint8_t>((encoded >> 4) & 0x03);
    }
    std::memcpy(canonical.data(), &block, sizeof(block));
    return canonical;
}

static void run_out_of_range_symmetric_weight_materialization_checks(ggml_backend_hrx_context * context) {
    constexpr int64_t input_size = 256;
    constexpr float   fp16_min   = 0x1p-24f;
    constexpr float   fp16_max   = 65504.0f;

    constexpr int64_t k64_output_size = 64;
    require_test_weight_rejected(context, GGML_TYPE_IQ4_XS, ggml::hrx::kSymmetricI4K64Row64Layout, input_size,
                                 k64_output_size, k64_output_size * 144,
                                 make_iq4_xs_scale_case(k64_output_size, fp16_min, 33, 8));
    require_test_weight_rejected(context, GGML_TYPE_IQ4_XS, ggml::hrx::kSymmetricI4K64Row64Layout, input_size,
                                 k64_output_size, k64_output_size * 144,
                                 make_iq4_xs_scale_case(k64_output_size, fp16_max, 63, 15));

    constexpr int64_t shared4_output_size = 4;
    require_test_weight_rejected(context, GGML_TYPE_Q6_K, ggml::hrx::kSymmetricI4K32EightGroupsShared4Layout,
                                 input_size, shared4_output_size, 32 * 132,
                                 make_q6_k_scale_case(shared4_output_size, fp16_min, 1, 1));
    require_test_weight_rejected(context, GGML_TYPE_Q6_K, ggml::hrx::kSymmetricI4K32EightGroupsShared4Layout,
                                 input_size, shared4_output_size, 32 * 132,
                                 make_q6_k_scale_case(shared4_output_size, fp16_max, 127, -32));

    constexpr int64_t composite_output_size = 64;
    constexpr size_t  composite_size        = composite_output_size * 68 + composite_output_size * 210;
    require_test_weight_rejected(context, GGML_TYPE_Q6_K, ggml::hrx::kQ6KSymmetricI2PackedScaleRowLayout, input_size,
                                 composite_output_size, composite_size,
                                 make_q6_k_scale_case(composite_output_size, fp16_min, 1, -1));
    require_test_weight_rejected(context, GGML_TYPE_Q6_K, ggml::hrx::kQ6KSymmetricI2PackedScaleRowLayout, input_size,
                                 composite_output_size, composite_size,
                                 make_q6_k_scale_case(composite_output_size, fp16_max, 127, -32));
}

int main() {
    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_backend_hrx_context * context = backend_context(backend);

    run_backend_buffer_checks(backend);
    run_host_buffer_checks(backend);
    run_host_transfer_checks(context);
    run_host_staging_checks(context);
    run_host_weight_cache_checks(context);
    run_zero_symmetric_weight_materialization_checks(context);
    run_nonfinite_symmetric_weight_materialization_checks(context);
    run_out_of_range_symmetric_weight_materialization_checks(context);

    ggml_backend_free(backend);
    return 0;
}
