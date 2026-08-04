#pragma once

#include "graph-index.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ggml::hrx {

using FactValue = std::variant<int64_t, double, bool, std::string>;

struct FactEvidence {
    std::string source;
    uint32_t graph_id = kInvalidId;
};

struct Fact {
    std::string key;
    FactValue value;
    std::vector<FactEvidence> evidence;
};

class FactDatabase {
public:
    Decision observe(std::string key, FactValue value, FactEvidence evidence);
    const Fact * find(const std::string & key) const;
    const std::map<std::string, Fact> & facts() const { return facts_; }

private:
    std::map<std::string, Fact> facts_;
};

struct SemanticBindings {
    std::map<std::string, OperationId> operations;
    std::map<std::string, ValueId> values;
};

enum class CostEvidenceKind : uint8_t {
    StructuralDominance,
    Measured,
};

struct CandidateEconomics {
    CostEvidenceKind evidence = CostEvidenceKind::StructuralDominance;
    int64_t measured_nanoseconds_saved = 0;
    int64_t eliminated_materialization_bytes = 0;
    int64_t additional_scratch_bytes = 0;
    int32_t reference_dispatches = 0;
    int32_t planned_dispatches = 0;
};

struct CandidatePayload {
    virtual ~CandidatePayload() = default;
};

struct FusionCandidate {
    std::string provider;
    std::string family;
    std::string key;
    OperationId hero = kInvalidId;
    std::vector<OperationId> operations;
    std::vector<ValueId> materialized_outputs;
    // Multi-dispatch recipes may intentionally own several roots connected by
    // a shared boundary input or schedule-only state. Providers must opt in;
    // ordinary single-cluster candidates remain connectivity checked.
    bool allow_disconnected = false;
    // A correctness baseline may remain selectable without a positive
    // optimization benefit. It ranks behind profitable native candidates and
    // makes fallback ownership explicit instead of gaming a cost estimate.
    bool correctness_baseline = false;
    SemanticBindings bindings;
    CandidateEconomics economics;
    std::shared_ptr<const CandidatePayload> payload;
};

struct CandidateScore {
    CostEvidenceKind evidence = CostEvidenceKind::StructuralDominance;
    int64_t primary_benefit = 0;
    int64_t secondary_benefit = 0;
    int64_t covered_operations = 0;

    bool positive() const;
};

struct SearchOptions {
    bool require_complete_coverage = false;
    bool record_trace = false;
    size_t maximum_candidates = 100000;
    size_t maximum_expansions = 100000;
};

enum class SearchEventKind : uint8_t {
    Seeded,
    Rejected,
    Popped,
    Expanded,
    Stale,
    Invalidated,
    Committed,
};

struct SearchEvent {
    SearchEventKind kind = SearchEventKind::Rejected;
    std::string candidate;
    CandidateScore score;
    Decision decision;
};

struct SearchReport {
    size_t seeded = 0;
    size_t rejected = 0;
    size_t popped = 0;
    size_t stale = 0;
    size_t expanded = 0;
    size_t invalidated = 0;
    size_t committed = 0;
    std::vector<SearchEvent> events;
};

class FusionProvider {
public:
    virtual ~FusionProvider() = default;
    virtual const char * id() const = 0;
    virtual const char * revision() const = 0;
    virtual Decision discover(const GraphIndex & index, FactDatabase & facts) const;
    virtual void seed(const GraphIndex & index, const FactDatabase & facts,
                      std::vector<FusionCandidate> & candidates) const = 0;
    virtual void expand(const GraphIndex & index, const FactDatabase & facts,
                        const FusionCandidate & candidate,
                        std::vector<FusionCandidate> & expansions) const;
};

class PlannerConfiguration {
public:
    void add_provider(std::shared_ptr<const FusionProvider> provider);
    const std::vector<std::shared_ptr<const FusionProvider>> & providers() const { return providers_; }
    std::string identity() const;

private:
    std::vector<std::shared_ptr<const FusionProvider>> providers_;
};

struct SearchResult {
    FactDatabase facts;
    std::vector<FusionCandidate> selected;
    std::vector<OperationId> uncovered_operations;
    SearchReport report;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

CandidateScore score_candidate(const FusionCandidate & candidate);
SearchResult search_fusions(const GraphIndex & index, const PlannerConfiguration & configuration,
                            const SearchOptions & options = {});
std::string format_search_report(const SearchResult & result);
std::string serialize_search_report_json(const SearchResult & result);
std::string fusion_region_dot(const GraphIndex & index, const SearchResult & result);

} // namespace ggml::hrx
