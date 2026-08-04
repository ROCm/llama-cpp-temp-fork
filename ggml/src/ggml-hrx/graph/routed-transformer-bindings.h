#pragma once

#include "routed-transformer.h"
#include "schedule.h"

namespace ggml::hrx {

// Resolves a structurally recovered routed-transformer schedule to concrete
// kernel ABI bindings. This is independent of raw operation ordinals and the
// legacy Qwen whole-program recognizer.
VerificationResult materialize_routed_transformer_dispatch_bindings(Graph & graph, Schedule & schedule);

} // namespace ggml::hrx
