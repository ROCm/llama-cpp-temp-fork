#include "transitional-graph.h"

namespace ggml::hrx {

transitional_graph transitional_graph::from_plan(const graph_plan & plan) {
    transitional_graph result;
    result.storages   = plan.storages();
    result.values     = plan.values();
    result.operations = plan.operations();
    result.roots      = plan.roots();
    result.errors     = plan.diagnostics().errors;
    return result;
}

}  // namespace ggml::hrx
