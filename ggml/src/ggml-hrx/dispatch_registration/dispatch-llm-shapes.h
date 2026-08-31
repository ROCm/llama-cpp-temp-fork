#pragma once

#include "dispatch-llm-profiles.h"

#include <cstddef>
#include <cstdint>

namespace ggml::hrx {

constexpr bool is_llm_prefill_512_query_length(const LlmMoeDispatchProfile & profile, int64_t query_length) {
    return is_llm_prefill_query_length(profile, query_length) && query_length == 512;
}

constexpr bool is_qwen_supported_query_length(int64_t query_length) {
    return is_llm_supported_query_length(kQwen30BMoeDispatchProfile, query_length);
}

constexpr bool is_qwen_decode_query_length(int64_t query_length) {
    return is_llm_decode_query_length(query_length);
}

constexpr bool is_qwen_prefill_query_length(int64_t query_length) {
    return is_llm_prefill_query_length(kQwen30BMoeDispatchProfile, query_length);
}

constexpr bool is_qwen_prefill_512_query_length(int64_t query_length) {
    return is_llm_prefill_512_query_length(kQwen30BMoeDispatchProfile, query_length);
}

// Private packed activation shared by low-row symmetric-I4 contractions and their semantic producers.
struct LlmSymmetricI4ActivationLayout {
    size_t payload_bytes  = 0;
    size_t scales_offset  = 0;
    size_t metadata_bytes = 0;
    size_t sums_offset    = 0;
    size_t total_bytes    = 0;
};

inline constexpr const char * kLlmSymmetricI4ActivationAlternateName = "llm.symmetric_i4.activation";
inline constexpr const char * kLlmQ8SwiGluPlaneAlternateName         = "qwen.prefill.q8_swiglu_plane";

constexpr size_t llm_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr LlmSymmetricI4ActivationLayout llm_symmetric_i4_activation_layout(int64_t input_size, int64_t token_count) {
    const size_t element_count  = static_cast<size_t>(input_size) * static_cast<size_t>(token_count);
    const size_t payload_bytes  = element_count / 2;
    const size_t metadata_bytes = element_count / 8;
    const size_t scales_offset  = llm_align_up(payload_bytes, 256);
    const size_t sums_offset    = llm_align_up(scales_offset + metadata_bytes, 256);
    return {
        payload_bytes, scales_offset, metadata_bytes, sums_offset, llm_align_up(sums_offset + metadata_bytes, 256),
    };
}

}  // namespace ggml::hrx
