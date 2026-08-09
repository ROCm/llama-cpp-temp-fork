#pragma once

#include "hybrid-carrier-recipes.h"

namespace ggml::hrx {

// Lowers one selected physical recipe to its kernel ABI.
VerificationResult materialize_hybrid_carrier_recipe(
    Graph & graph, Invocation & invocation,
    const HybridCarrierRecipeMatch & recipe, uint32_t & dispatch_ordinal);

} // namespace ggml::hrx
