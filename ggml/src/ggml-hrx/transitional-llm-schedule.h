#pragma once

#include "transitional-llm-program.h"
#include "transitional-schedule.h"

namespace ggml::hrx {

struct RoutedTransformerProgramProof {
    bool                                          structurally_recognized = false;
    std::shared_ptr<const RoutedTransformerModel> logical_program;
    Schedule                                      schedule;
    SearchResult                                  search;
    std::vector<std::string>                      native_gaps;
    std::vector<std::string>                      errors;

    bool valid() const {
        return structurally_recognized && errors.empty() && !schedule.invocations.empty() && search.valid() &&
               search.uncovered_operations.empty();
    }

    static RoutedTransformerProgramProof recover(const Graph & graph);
    static VerificationResult            materialize_dispatch_bindings(Graph &                        graph,
                                                                       Schedule &                     schedule,
                                                                       const RoutedTransformerModel & logical_program);
};

// Materializes the currently executable Qwen-derived recipes from structural
// routed-transformer candidates. Kernel names retain their heritage; graph
// discovery and specialization facts do not depend on a Qwen model identity.
}  // namespace ggml::hrx
