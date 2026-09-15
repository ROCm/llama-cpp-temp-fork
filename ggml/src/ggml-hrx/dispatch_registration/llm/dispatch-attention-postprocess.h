#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_llm_attention_postprocess_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
