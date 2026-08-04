#pragma once

#include "schedule.h"

#include <string>
#include <vector>

namespace ggml::hrx {

struct QwenProgramProof {
    Schedule schedule;
    std::vector<std::string> errors;
    std::vector<std::string> native_gaps;
    std::vector<std::string> root_seams;

    bool recognized() const { return errors.empty() && !schedule.invocations.empty(); }
    bool structurally_sufficient() const;
    bool natively_complete() const;
    static QwenProgramProof recover(const Graph & graph);
    static VerificationResult verify(const Graph & graph, const QwenProgramProof & proof);
    static std::string signature(const QwenProgramProof & proof);
    static VerificationResult materialize_dispatch_bindings(Graph & graph, Schedule & schedule);
};

} // namespace ggml::hrx
