#pragma once

#include "fusion-search.h"

#include <set>
#include <string>
#include <vector>

namespace ggml::hrx {

struct RoutedTransformerComponent {
    std::string role;
    OperationId hero = kInvalidId;
    std::vector<OperationId> operations;
    SemanticBindings bindings;
};

struct RoutedTransformerBlock {
    size_t ordinal = 0;
    SemanticBindings bindings;
    std::vector<OperationId> operations;
    std::vector<RoutedTransformerComponent> components;
};

struct RoutedTransformerModel {
    std::string graph_fingerprint;
    SemanticBindings bindings;
    std::vector<OperationId> preamble_operations;
    std::vector<RoutedTransformerBlock> blocks;
    std::vector<OperationId> endpoint_operations;
    int64_t query_token_count = 0;
    int64_t output_token_count = 0;
    int64_t key_value_token_count = 0;
    int64_t hidden_size = 0;
    int64_t query_size = 0;
    int64_t key_value_size = 0;
    int64_t expert_count = 0;
    int64_t route_count = 0;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty() && !blocks.empty(); }
    static RoutedTransformerModel analyze(const GraphIndex & index);
};

// A schedule family is offered to the search only when its physical recipe is
// available. Keeping this as a recipe catalog (rather than a model mode) lets
// the same provider compare current and newly landed kernels incrementally.
struct RoutedTransformerRecipeCatalog {
    std::set<std::string> available;

    bool contains(const std::string & recipe) const { return available.count(recipe) != 0; }
};

namespace routed_transformer_recipes {
inline constexpr const char * kDecodeQkvPostprocess = "decode.attention.qkv_postprocess";
inline constexpr const char * kDecodeOutputNextQ8 = "decode.attention.output_next_q8";
inline constexpr const char * kDecodeRouterTopK = "decode.router.projection_topk";
inline constexpr const char * kDecodeGateUpNextQ8 = "decode.experts.gate_up_next_q8";
inline constexpr const char * kDecodeDownNextQ8 = "decode.experts.down_next_q8";
inline constexpr const char * kPrefillExpertPartition = "prefill.router.expert_partition";
inline constexpr const char * kPrefillDownNextNorm = "prefill.experts.down_next_norm";
} // namespace routed_transformer_recipes

// Structural provider used by the generic search proof. It exposes semantic
// components rather than a model-sized op; physical recipe materialization is
// layered on these candidates.
class RoutedTransformerProvider final : public FusionProvider {
public:
    explicit RoutedTransformerProvider(
        RoutedTransformerRecipeCatalog catalog = {},
        std::shared_ptr<const RoutedTransformerModel> supplied_model = {})
        : catalog_(std::move(catalog)), supplied_model_(std::move(supplied_model)) {}
    const char * id() const override { return "llm.routed_transformer"; }
    const char * revision() const override { return "2"; }
    Decision discover(const GraphIndex & index, FactDatabase & facts) const override;
    void seed(const GraphIndex & index, const FactDatabase & facts,
              std::vector<FusionCandidate> & candidates) const override;
    void expand(const GraphIndex & index, const FactDatabase & facts,
                const FusionCandidate & candidate,
                std::vector<FusionCandidate> & expansions) const override;
    static PlannerConfiguration make_planner(
        RoutedTransformerRecipeCatalog catalog = {},
        std::shared_ptr<const RoutedTransformerModel> supplied_model = {});

private:
    RoutedTransformerRecipeCatalog catalog_;
    std::shared_ptr<const RoutedTransformerModel> supplied_model_;
};

} // namespace ggml::hrx
