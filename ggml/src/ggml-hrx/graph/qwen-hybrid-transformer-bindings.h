#pragma once

#include "qwen-hybrid-recipes.h"

namespace ggml::hrx {

// Lowers one selected physical recipe to its kernel ABI.
VerificationResult materialize_qwen_hybrid_recipe(Graph &                       graph,
                                                  Invocation &                  invocation,
                                                  const QwenHybridRecipeMatch & recipe,
                                                  uint32_t &                    dispatch_ordinal);

}  // namespace ggml::hrx
