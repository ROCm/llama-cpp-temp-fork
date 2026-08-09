#pragma once

#include "fusion-search.h"
#include "schedule.h"

#include <memory>
#include <string>
#include <vector>

namespace ggml::hrx {

struct HybridCarrierRecipeDiscovery;

enum class HybridCarrierComponentKind : uint8_t {
    ProgramPreamble,
    AttentionControl,
    AttentionBody,
    FeedForwardControl,
    RouterSelection,
    ExpertGateUp,
    ExpertDownPublication,
    ProgramEndpoint,
    Atom,
};

struct HybridCarrierOperations {
    OperationId attention_pre = kInvalidId;
    OperationId attention_post = kInvalidId;
    OperationId feed_forward_pre = kInvalidId;
    OperationId feed_forward_post = kInvalidId;
    OperationId router_projection = kInvalidId;
    OperationId experts_gate_projection = kInvalidId;
    OperationId experts_up_projection = kInvalidId;
    OperationId experts_gate_up = kInvalidId;
    OperationId experts_down_projection = kInvalidId;
    OperationId experts_weighted = kInvalidId;
};

struct HybridCarrierValues {
    ValueId hidden_input = kInvalidId;
    ValueId attention_output = kInvalidId;
    ValueId feed_forward_input = kInvalidId;
    ValueId route_ids = kInvalidId;
    ValueId route_weights = kInvalidId;
    ValueId expert_activation = kInvalidId;
};

struct HybridCarrierComponent {
    uint32_t id = kInvalidId;
    HybridCarrierComponentKind kind = HybridCarrierComponentKind::Atom;
    OperationId hero = kInvalidId;
    std::vector<OperationId> operations;
    RegionBoundary boundary;
};

struct HybridCarrierBlock {
    size_t ordinal = 0;
    HybridCarrierOperations operations_by_role;
    HybridCarrierValues values_by_role;
    std::vector<OperationId> operations;
    std::vector<HybridCarrierComponent> components;
};

struct HybridCarrierTransformerModel {
    std::string graph_fingerprint;
    HybridCarrierComponent preamble;
    std::vector<OperationId> preamble_operations;
    std::vector<HybridCarrierBlock> blocks;
    HybridCarrierComponent endpoint;
    std::vector<OperationId> endpoint_operations;
    std::vector<HybridCarrierComponent> fallback_components;
    std::vector<OperationId> unraised_operations;
    int64_t token_count = 0;
    int64_t hidden_size = 0;
    int64_t carrier_count = 0;
    int64_t expert_count = 0;
    int64_t route_count = 0;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty() && !blocks.empty(); }
    static HybridCarrierTransformerModel analyze(const GraphIndex & index);
    static VerificationResult verify(const GraphIndex & index, const HybridCarrierTransformerModel & model);
    static const char * component_kind_name(HybridCarrierComponentKind kind);
    static std::string format(const HybridCarrierTransformerModel & model);
    static std::string serialize_json(const HybridCarrierTransformerModel & model);
    static std::string dot(const HybridCarrierTransformerModel & model);
};

class HybridCarrierTransformerProvider final : public FusionProvider {
public:
    explicit HybridCarrierTransformerProvider(
        std::shared_ptr<const HybridCarrierTransformerModel> supplied_model = {},
        std::shared_ptr<const HybridCarrierRecipeDiscovery> supplied_recipes = {})
        : supplied_model_(std::move(supplied_model)), supplied_recipes_(std::move(supplied_recipes)) {}

    const char * id() const override { return "llm.hybrid_carrier_transformer"; }
    const char * revision() const override { return "1"; }
    Decision discover(const GraphIndex & index, FactDatabase & facts) const override;
    void seed(const GraphIndex & index, const FactDatabase & facts,
              std::vector<FusionCandidate> & candidates) const override;

    static PlannerConfiguration make_planner(
        std::shared_ptr<const HybridCarrierTransformerModel> supplied_model = {},
        std::shared_ptr<const HybridCarrierRecipeDiscovery> supplied_recipes = {});

private:
    std::shared_ptr<const HybridCarrierTransformerModel> supplied_model_;
    std::shared_ptr<const HybridCarrierRecipeDiscovery> supplied_recipes_;
};

struct HybridCarrierTransformerProgramProof {
    bool structurally_recognized = false;
    std::shared_ptr<const HybridCarrierTransformerModel> logical_program;
    Schedule schedule;
    SearchResult search;
    std::vector<std::string> errors;

    bool valid() const {
        return structurally_recognized && errors.empty() && !schedule.invocations.empty() &&
            search.valid() && search.uncovered_operations.empty();
    }

    static HybridCarrierTransformerProgramProof recover(Graph & graph, const std::string & target);
};

} // namespace ggml::hrx
