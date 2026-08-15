#pragma once

#include "fusion-search.h"
#include "schedule.h"

#include <memory>
#include <string>
#include <vector>

namespace ggml::hrx {

struct QwenHybridRecipeDiscovery;

enum class QwenHybridComponentKind : uint8_t {
    ProgramPreamble,
    Block,
    ProgramEndpoint,
};

struct QwenHybridComponent {
    uint32_t                 id   = kInvalidId;
    QwenHybridComponentKind  kind = QwenHybridComponentKind::Block;
    OperationId              hero = kInvalidId;
    std::vector<OperationId> operations;
    RegionBoundary           boundary;
};

enum class QwenHybridFeedForwardKind : uint8_t {
    RoutedExperts,
    DenseSwiGLU,
};

struct QwenHybridBlock {
    size_t                           ordinal            = 0;
    QwenHybridFeedForwardKind        feed_forward_kind  = QwenHybridFeedForwardKind::RoutedExperts;
    OperationId                      feed_forward_hero  = kInvalidId;
    ValueId                          feed_forward_input = kInvalidId;
    OperationId                      router_projection  = kInvalidId;
    OperationId                      attention_residual = kInvalidId;
    OperationId                      final_residual     = kInvalidId;
    ValueId                          route_ids          = kInvalidId;
    std::vector<OperationId>         operations;
    std::vector<QwenHybridComponent> components;
};

struct QwenHybridTransformerModel {
    std::string                  graph_fingerprint;
    QwenHybridComponent          preamble;
    std::vector<QwenHybridBlock> blocks;
    QwenHybridComponent          endpoint;
    int64_t                      token_count  = 0;
    int64_t                      hidden_size  = 0;
    int64_t                      expert_count = 0;
    int64_t                      route_count  = 0;
    std::vector<std::string>     errors;

    bool valid() const { return errors.empty() && !blocks.empty(); }

    static QwenHybridTransformerModel analyze(const GraphIndex & index);
    static VerificationResult         verify(const GraphIndex & index, const QwenHybridTransformerModel & model);
    static const char *               component_kind_name(QwenHybridComponentKind kind);
    static std::string                format(const QwenHybridTransformerModel & model);
    static std::string                serialize_json(const QwenHybridTransformerModel & model);
    static std::string                dot(const QwenHybridTransformerModel & model);
};

class QwenHybridTransformerProvider final : public FusionProvider {
  public:
    explicit QwenHybridTransformerProvider(std::shared_ptr<const QwenHybridTransformerModel> supplied_model   = {},
                                           std::shared_ptr<const QwenHybridRecipeDiscovery>  supplied_recipes = {}) :
        supplied_model_(std::move(supplied_model)),
        supplied_recipes_(std::move(supplied_recipes)) {}

    const char * id() const override { return "llm.qwen_hybrid_transformer"; }

    const char * revision() const override { return "2"; }

    Decision discover(const GraphIndex & index, FactDatabase & facts) const override;
    void     seed(const GraphIndex &             index,
                  const FactDatabase &           facts,
                  std::vector<FusionCandidate> & candidates) const override;

    static PlannerConfiguration make_planner(std::shared_ptr<const QwenHybridTransformerModel> supplied_model   = {},
                                             std::shared_ptr<const QwenHybridRecipeDiscovery>  supplied_recipes = {});

  private:
    std::shared_ptr<const QwenHybridTransformerModel> supplied_model_;
    std::shared_ptr<const QwenHybridRecipeDiscovery>  supplied_recipes_;
};

struct QwenHybridTransformerProgramProof {
    bool                                              structurally_recognized = false;
    std::shared_ptr<const QwenHybridTransformerModel> logical_program;
    Schedule                                          schedule;
    SearchResult                                      search;
    std::vector<std::string>                          errors;

    bool valid() const {
        return structurally_recognized && errors.empty() && !schedule.invocations.empty() && search.valid() &&
               search.uncovered_operations.empty();
    }

    static QwenHybridTransformerProgramProof recover(Graph & graph, const std::string & target);
};

}  // namespace ggml::hrx
