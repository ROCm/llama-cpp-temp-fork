#pragma once

#include "hybrid-carrier-transformer.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace ggml::hrx {

// Carries a verified physical recipe and normalized graph identities into lowering.
struct HybridCarrierRecipeMatch final : CandidatePayload {
    std::string recipe;
    std::shared_ptr<const nlohmann::json> definition;
    std::map<std::string, OperationId> operations_by_role;
    std::map<std::string, ValueId> values_by_role;
    std::vector<OperationId> operations;
    std::vector<uint32_t> logical_components;
    int32_t layer = -1;
    size_t dispatch_count = 0;
    size_t eliminated_materialization_bytes = 0;
    bool allow_disconnected = false;
};

struct HybridCarrierRecipeDiscovery {
    std::vector<std::shared_ptr<const HybridCarrierRecipeMatch>> matches;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }

    static HybridCarrierRecipeDiscovery discover(
        const GraphIndex & index, const HybridCarrierTransformerModel & model,
        const std::string & target);
};

} // namespace ggml::hrx
