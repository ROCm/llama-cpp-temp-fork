#pragma once

#include "fusion-search.h"

#include <set>
#include <string>
#include <vector>

namespace ggml::hrx {

struct VerificationResult;

using LogicalComponentId = uint32_t;

enum class RoutedTransformerComponentKind : uint8_t {
    ProgramPreamble,
    AttentionPrepare,
    AttentionQkvPublication,
    Attention,
    AttentionOutputPrepare,
    RouterSelection,
    ExpertGateUp,
    ExpertDownPublication,
    ProgramEndpoint,
    Atom,
};

struct RoutedTransformerOperations {
    OperationId program_embedding = kInvalidId;
    OperationId endpoint_norm = kInvalidId;
    OperationId endpoint_prepared = kInvalidId;
    OperationId endpoint_projection = kInvalidId;

    OperationId attention_norm = kInvalidId;
    OperationId attention_prepared = kInvalidId;
    OperationId attention_query_projection = kInvalidId;
    OperationId attention_key_projection = kInvalidId;
    OperationId attention_value_projection = kInvalidId;
    OperationId attention_query_rope = kInvalidId;
    OperationId attention_key_cache_writer = kInvalidId;
    OperationId attention_value_cache_writer = kInvalidId;
    OperationId attention_flash = kInvalidId;
    OperationId attention_result_reshape = kInvalidId;
    OperationId attention_output_projection = kInvalidId;
    OperationId attention_output_selection = kInvalidId;
    OperationId hidden_state_selection = kInvalidId;
    OperationId attention_residual = kInvalidId;
    OperationId feed_forward_prepared = kInvalidId;
    OperationId router_projection = kInvalidId;
    OperationId router_route_ids = kInvalidId;
    OperationId router_route_weights = kInvalidId;
    OperationId experts_gate_projection = kInvalidId;
    OperationId experts_up_projection = kInvalidId;
    OperationId experts_gate_up = kInvalidId;
    OperationId experts_routed_down = kInvalidId;
    OperationId hidden_output = kInvalidId;
};

struct RoutedTransformerValues {
    ValueId program_hidden_state = kInvalidId;
    ValueId attention_prepared = kInvalidId;
    ValueId router_route_ids = kInvalidId;
    ValueId experts_activation = kInvalidId;
};

struct RoutedTransformerComponent {
    LogicalComponentId id = kInvalidId;
    RoutedTransformerComponentKind kind = RoutedTransformerComponentKind::Atom;
    OperationId hero = kInvalidId;
    std::vector<OperationId> operations;
    RegionBoundary boundary;
};

struct RoutedTransformerBlock {
    size_t ordinal = 0;
    RoutedTransformerOperations operations_by_role;
    RoutedTransformerValues values_by_role;
    std::vector<OperationId> operations;
    std::vector<RoutedTransformerComponent> components;
};

struct RoutedTransformerModel {
    std::string graph_fingerprint;
    RoutedTransformerOperations operations_by_role;
    RoutedTransformerValues values_by_role;
    RoutedTransformerComponent preamble;
    std::vector<OperationId> preamble_operations;
    std::vector<RoutedTransformerBlock> blocks;
    RoutedTransformerComponent endpoint;
    std::vector<OperationId> endpoint_operations;
    std::vector<RoutedTransformerComponent> fallback_components;
    std::vector<OperationId> unraised_operations;
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
    static VerificationResult verify(const GraphIndex & index, const RoutedTransformerModel & model);
    static const char * component_kind_name(RoutedTransformerComponentKind kind);
    static std::string format(const RoutedTransformerModel & model);
    static std::string serialize_json(const RoutedTransformerModel & model);
    static std::string dot(const RoutedTransformerModel & model);
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
