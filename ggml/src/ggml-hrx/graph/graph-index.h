#pragma once

#include "graph-ir.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class DecisionReason : uint16_t {
    Allowed,
    EmptyRegion,
    InvalidOperation,
    InvalidValue,
    DuplicateOperation,
    DisconnectedRegion,
    ContractedCycle,
    MissingMaterialization,
    Overlap,
    UnsupportedRecipe,
    InconsistentFact,
    NoComparableCost,
    NoNativeCoverage,
    ProviderError,
};

struct Decision {
    bool                  allowed = false;
    DecisionReason        reason  = DecisionReason::ProviderError;
    std::string           detail;
    std::vector<uint32_t> implicated_ids;

    static Decision     allow();
    static Decision     reject(DecisionReason reason, std::string detail, std::vector<uint32_t> implicated_ids = {});
    static const char * reason_name(DecisionReason reason);
};

struct RegionBoundary {
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
};

// Immutable adjacency and storage-version index over the normalized graph.
// Search providers query this object instead of repeatedly scanning Graph or
// depending on the incidental operation vector layout.
class GraphIndex {
  public:
    explicit GraphIndex(const Graph & graph);

    const Graph & graph() const { return graph_; }

    bool valid() const { return errors_.empty(); }

    const std::vector<std::string> & errors() const { return errors_; }

    const std::vector<OperationId> & consumers(ValueId value) const;
    const std::vector<OperationId> & predecessors(OperationId operation) const;
    const std::vector<OperationId> & successors(OperationId operation) const;
    OperationId                      storage_writer(StorageId storage, uint32_t version) const;
    const std::string &              structural_key(OperationId operation) const;

    RegionBoundary boundary(const std::vector<OperationId> & operations) const;
    Decision       validate_region(const std::vector<OperationId> & operations,
                                   const std::vector<ValueId> &     materialized_outputs,
                                   bool                             allow_disconnected = false) const;
    Decision       topologically_order_regions(const std::vector<std::vector<OperationId>> & regions,
                                               std::vector<size_t> &                         order) const;

  private:
    const Graph &                                         graph_;
    std::vector<std::vector<OperationId>>                 consumers_;
    std::vector<std::vector<OperationId>>                 predecessors_;
    std::vector<std::vector<OperationId>>                 successors_;
    std::vector<uint8_t>                                  root_values_;
    std::map<std::pair<StorageId, uint32_t>, OperationId> writers_;
    std::vector<std::string>                              structural_keys_;
    std::vector<std::string>                              errors_;
};

}  // namespace ggml::hrx
