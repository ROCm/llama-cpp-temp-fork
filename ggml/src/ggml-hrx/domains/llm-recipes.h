#pragma once

#include "graph-plan.h"

#include <cstring>

namespace ggml::hrx::llm_recipes {

inline schedule_capability capability(region_kind kind) {
    switch (kind) {
        case region_kind::REGION_KIND_EMBEDDING:
            return schedule_capability::SCHEDULE_CAPABILITY_EMBEDDING;
        case region_kind::REGION_KIND_ATTENTION_PREPARE:
            return schedule_capability::SCHEDULE_CAPABILITY_ATTENTION_PREPARE;
        case region_kind::REGION_KIND_ATTENTION_QKV:
            return schedule_capability::SCHEDULE_CAPABILITY_ATTENTION_QKV;
        case region_kind::REGION_KIND_ATTENTION:
            return schedule_capability::SCHEDULE_CAPABILITY_ATTENTION;
        case region_kind::REGION_KIND_ATTENTION_OUTPUT:
            return schedule_capability::SCHEDULE_CAPABILITY_ATTENTION_OUTPUT;
        case region_kind::REGION_KIND_ROUTER_SELECTION:
            return schedule_capability::SCHEDULE_CAPABILITY_ROUTER_SELECTION;
        case region_kind::REGION_KIND_EXPERT_GATE_UP:
            return schedule_capability::SCHEDULE_CAPABILITY_EXPERT_GATE_UP;
        case region_kind::REGION_KIND_EXPERT_DOWN:
            return schedule_capability::SCHEDULE_CAPABILITY_EXPERT_DOWN;
        case region_kind::REGION_KIND_ENDPOINT:
            return schedule_capability::SCHEDULE_CAPABILITY_ENDPOINT;
        case region_kind::REGION_KIND_ATOM:
        case region_kind::REGION_KIND_FUSION:
            return schedule_capability::SCHEDULE_CAPABILITY_PRIMITIVE;
    }
    return schedule_capability::SCHEDULE_CAPABILITY_PRIMITIVE;
}

inline bool supports_qwen3_30b_decode_576(const std::map<std::string, int64_t> & facts) {
    auto fact_is = [&](const char * name, int64_t value) {
        const auto position = facts.find(name);
        return position != facts.end() && position->second == value;
    };
    float    expected_epsilon      = 0.000001f;
    uint32_t expected_epsilon_bits = 0;
    std::memcpy(&expected_epsilon_bits, &expected_epsilon, sizeof(expected_epsilon_bits));
    return fact_is("query_token_count", 1) && fact_is("key_value_token_count", 576) &&
           fact_is("output_token_count", 1) && fact_is("layer_count", 48) && fact_is("hidden_size", 2048) &&
           fact_is("head_size", 128) && fact_is("query_size", 4096) && fact_is("key_value_size", 512) &&
           fact_is("query_head_count", 32) && fact_is("key_value_head_count", 4) && fact_is("expert_count", 128) &&
           fact_is("route_count", 8) && fact_is("expert_intermediate_size", 768) &&
           fact_is("vocabulary_count", 151936) && fact_is("rms_epsilon_bits", expected_epsilon_bits);
}

}  // namespace ggml::hrx::llm_recipes
