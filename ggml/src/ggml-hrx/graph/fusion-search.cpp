#include "fusion-search.h"

#include <algorithm>
#include <iomanip>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

static bool same_fact_value(const FactValue & lhs, const FactValue & rhs) {
    return lhs == rhs;
}

static std::string fact_value_text(const FactValue & value) {
    return std::visit([](const auto & item) {
        std::ostringstream out;
        out << std::boolalpha << item;
        return out.str();
    }, value);
}

static bool better_score(const CandidateScore & lhs, const CandidateScore & rhs) {
    const auto left = std::make_tuple(lhs.evidence == CostEvidenceKind::Measured,
                                      lhs.primary_benefit, lhs.secondary_benefit,
                                      lhs.covered_operations);
    const auto right = std::make_tuple(rhs.evidence == CostEvidenceKind::Measured,
                                       rhs.primary_benefit, rhs.secondary_benefit,
                                       rhs.covered_operations);
    return left > right;
}

static bool equal_score(const CandidateScore & lhs, const CandidateScore & rhs) {
    return lhs.evidence == rhs.evidence && lhs.primary_benefit == rhs.primary_benefit &&
        lhs.secondary_benefit == rhs.secondary_benefit && lhs.covered_operations == rhs.covered_operations;
}

struct QueueEntry {
    CandidateScore score;
    std::string key;
    size_t candidate = 0;
    uint64_t generation = 0;
};

struct WorseQueueEntry {
    bool operator()(const QueueEntry & lhs, const QueueEntry & rhs) const {
        if (!equal_score(lhs.score, rhs.score)) return better_score(rhs.score, lhs.score);
        return lhs.key > rhs.key;
    }
};

static void record(SearchResult & result, const SearchOptions & options, SearchEventKind kind,
                   const FusionCandidate & candidate, CandidateScore score, Decision decision) {
    if (!options.record_trace) return;
    result.report.events.push_back({ kind, candidate.key, score, std::move(decision) });
}

static std::string event_kind_name(SearchEventKind kind) {
    switch (kind) {
        case SearchEventKind::Seeded: return "seeded";
        case SearchEventKind::Rejected: return "rejected";
        case SearchEventKind::Popped: return "popped";
        case SearchEventKind::Expanded: return "expanded";
        case SearchEventKind::Stale: return "stale";
        case SearchEventKind::Invalidated: return "invalidated";
        case SearchEventKind::Committed: return "committed";
    }
    return "unknown";
}

} // namespace

Decision FactDatabase::observe(std::string key, FactValue value, FactEvidence evidence) {
    auto [position, inserted] = facts_.try_emplace(key);
    Fact & fact = position->second;
    if (inserted) {
        fact.key = std::move(key);
        fact.value = std::move(value);
        fact.evidence.push_back(std::move(evidence));
        return Decision::allow();
    }
    if (!same_fact_value(fact.value, value)) {
        std::vector<uint32_t> ids;
        for (const FactEvidence & prior : fact.evidence) if (prior.graph_id != kInvalidId) ids.push_back(prior.graph_id);
        if (evidence.graph_id != kInvalidId) ids.push_back(evidence.graph_id);
        return Decision::reject(DecisionReason::InconsistentFact,
                                "fact " + fact.key + " disagrees: " + fact_value_text(fact.value) +
                                " versus " + fact_value_text(value), std::move(ids));
    }
    fact.evidence.push_back(std::move(evidence));
    return Decision::allow();
}

const Fact * FactDatabase::find(const std::string & key) const {
    const auto position = facts_.find(key);
    return position == facts_.end() ? nullptr : &position->second;
}

bool CandidateScore::positive() const {
    return primary_benefit > 0 || (primary_benefit == 0 && secondary_benefit > 0);
}

Decision FusionProvider::discover(const GraphIndex &, FactDatabase &) const { return Decision::allow(); }
void FusionProvider::expand(const GraphIndex &, const FactDatabase &, const FusionCandidate &,
                            std::vector<FusionCandidate> &) const {}

void PlannerConfiguration::add_provider(std::shared_ptr<const FusionProvider> provider) {
    if (!provider) return;
    const std::string provider_id = provider->id();
    const auto duplicate = std::find_if(providers_.begin(), providers_.end(), [&](const auto & existing) {
        return existing->id() == provider_id;
    });
    if (duplicate != providers_.end()) return;
    providers_.push_back(std::move(provider));
    std::sort(providers_.begin(), providers_.end(), [](const auto & lhs, const auto & rhs) {
        return std::string(lhs->id()) < std::string(rhs->id());
    });
}

std::string PlannerConfiguration::identity() const {
    std::ostringstream out;
    for (const auto & provider : providers_) out << provider->id() << '@' << provider->revision() << ';';
    return out.str();
}

CandidateScore score_candidate(const FusionCandidate & candidate) {
    CandidateScore result;
    result.evidence = candidate.economics.evidence;
    result.covered_operations = candidate.operations.size();
    if (result.evidence == CostEvidenceKind::Measured) {
        result.primary_benefit = candidate.economics.measured_nanoseconds_saved;
        result.secondary_benefit = candidate.economics.reference_dispatches - candidate.economics.planned_dispatches;
    } else {
        result.primary_benefit = candidate.economics.reference_dispatches - candidate.economics.planned_dispatches;
        result.secondary_benefit = candidate.economics.eliminated_materialization_bytes -
            candidate.economics.additional_scratch_bytes;
    }
    return result;
}

SearchResult search_fusions(const GraphIndex & index, const PlannerConfiguration & configuration,
                            const SearchOptions & options) {
    SearchResult result;
    if (!index.valid()) {
        result.errors = index.errors();
        return result;
    }
    for (const auto & provider : configuration.providers()) {
        const Decision decision = provider->discover(index, result.facts);
        if (!decision.allowed) {
            result.errors.push_back(std::string(provider->id()) + ": " + decision.detail);
            return result;
        }
    }

    std::vector<FusionCandidate> candidates;
    std::vector<size_t> provider_for_candidate;
    std::set<std::string> candidate_keys;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, WorseQueueEntry> queue;
    std::vector<uint64_t> generations;
    std::vector<uint8_t> expanded;
    std::vector<uint8_t> active;

    auto insert_candidate = [&](FusionCandidate candidate, size_t provider_index, SearchEventKind event_kind) {
        if (candidates.size() >= options.maximum_candidates) {
            result.errors.push_back("fusion search exceeded its candidate limit");
            return;
        }
        std::sort(candidate.operations.begin(), candidate.operations.end());
        std::sort(candidate.materialized_outputs.begin(), candidate.materialized_outputs.end());
        if (candidate.key.empty()) candidate.key = candidate.provider + ":" + candidate.family + ":" +
            (candidate.hero == kInvalidId ? std::string("none") : index.structural_key(candidate.hero));
        if (!candidate_keys.insert(candidate.key).second) return;
        const Decision legality = index.validate_region(candidate.operations, candidate.materialized_outputs,
                                                        candidate.allow_disconnected);
        const CandidateScore score = score_candidate(candidate);
        if (!legality.allowed || (!score.positive() && !candidate.correctness_baseline)) {
            ++result.report.rejected;
            Decision rejection = legality.allowed
                ? Decision::reject(DecisionReason::NoComparableCost,
                                   "candidate has no positive comparable benefit") : legality;
            record(result, options, SearchEventKind::Rejected, candidate, score, std::move(rejection));
            return;
        }
        const size_t id = candidates.size();
        candidates.push_back(std::move(candidate));
        provider_for_candidate.push_back(provider_index);
        generations.push_back(1);
        expanded.push_back(0);
        active.push_back(1);
        queue.push({ score, candidates.back().key, id, generations.back() });
        if (event_kind == SearchEventKind::Seeded) ++result.report.seeded;
        else ++result.report.expanded;
        record(result, options, event_kind, candidates.back(), score, Decision::allow());
    };

    for (size_t provider_index = 0; provider_index < configuration.providers().size(); ++provider_index) {
        std::vector<FusionCandidate> seeds;
        configuration.providers()[provider_index]->seed(index, result.facts, seeds);
        for (FusionCandidate & seed : seeds) insert_candidate(std::move(seed), provider_index, SearchEventKind::Seeded);
        if (!result.errors.empty()) return result;
    }

    std::vector<uint8_t> claimed(index.graph().operations.size(), 0);
    size_t expansion_count = 0;
    while (!queue.empty()) {
        const QueueEntry entry = queue.top();
        queue.pop();
        if (entry.candidate >= candidates.size() || !active[entry.candidate] ||
            entry.generation != generations[entry.candidate]) {
            ++result.report.stale;
            if (entry.candidate < candidates.size()) {
                record(result, options, SearchEventKind::Stale, candidates[entry.candidate], entry.score,
                       Decision::reject(DecisionReason::Overlap, "queue entry is stale"));
            }
            continue;
        }
        FusionCandidate & candidate = candidates[entry.candidate];
        ++result.report.popped;
        record(result, options, SearchEventKind::Popped, candidate, entry.score, Decision::allow());

        const auto overlap = std::find_if(candidate.operations.begin(), candidate.operations.end(),
                                          [&](OperationId operation) { return claimed[operation] != 0; });
        if (overlap != candidate.operations.end()) {
            active[entry.candidate] = 0;
            ++result.report.invalidated;
            record(result, options, SearchEventKind::Invalidated, candidate, entry.score,
                   Decision::reject(DecisionReason::Overlap, "candidate overlaps a committed region", { *overlap }));
            continue;
        }

        if (!expanded[entry.candidate]) {
            expanded[entry.candidate] = 1;
            std::vector<FusionCandidate> expansions;
            configuration.providers()[provider_for_candidate[entry.candidate]]->expand(
                index, result.facts, candidate, expansions);
            expansion_count += expansions.size();
            if (expansion_count > options.maximum_expansions) {
                result.errors.push_back("fusion search exceeded its expansion limit");
                return result;
            }
            for (FusionCandidate & expansion : expansions) {
                insert_candidate(std::move(expansion), provider_for_candidate[entry.candidate], SearchEventKind::Expanded);
            }
            // Reinsert the seed so a better expansion wins first while the
            // original remains a valid fallback.
            ++generations[entry.candidate];
            queue.push({ entry.score, candidate.key, entry.candidate, generations[entry.candidate] });
            continue;
        }

        for (OperationId operation : candidate.operations) claimed[operation] = 1;
        result.selected.push_back(candidate);
        active[entry.candidate] = 0;
        ++result.report.committed;
        record(result, options, SearchEventKind::Committed, candidate, entry.score, Decision::allow());

        // Incremental invalidation is deliberately local to overlapping
        // candidates. Boundary-dependent rescoring can later use the same
        // generation mechanism without rebuilding the queue.
        for (size_t other = 0; other < candidates.size(); ++other) {
            if (!active[other]) continue;
            const bool overlaps = std::any_of(candidates[other].operations.begin(), candidates[other].operations.end(),
                                              [&](OperationId operation) { return claimed[operation] != 0; });
            if (overlaps) {
                active[other] = 0;
                ++generations[other];
                ++result.report.invalidated;
                record(result, options, SearchEventKind::Invalidated, candidates[other], score_candidate(candidates[other]),
                       Decision::reject(DecisionReason::Overlap, "candidate overlaps a committed region"));
            }
        }
    }

    for (OperationId operation = 0; operation < claimed.size(); ++operation) {
        if (!claimed[operation]) result.uncovered_operations.push_back(operation);
    }
    if (options.require_complete_coverage && !result.uncovered_operations.empty()) {
        result.errors.push_back("fusion search left " + std::to_string(result.uncovered_operations.size()) +
                                " operations uncovered");
    }

    std::vector<std::vector<OperationId>> regions;
    for (const FusionCandidate & candidate : result.selected) regions.push_back(candidate.operations);
    std::vector<size_t> order;
    const Decision ordering = index.topologically_order_regions(regions, order);
    if (!ordering.allowed) {
        result.errors.push_back(ordering.detail);
        return result;
    }
    std::vector<FusionCandidate> ordered;
    ordered.reserve(result.selected.size());
    for (size_t position : order) ordered.push_back(std::move(result.selected[position]));
    result.selected = std::move(ordered);
    return result;
}

std::string format_search_report(const SearchResult & result) {
    std::ostringstream out;
    out << "fusion-search valid=" << (result.valid() ? "yes" : "no")
        << " facts=" << result.facts.facts().size()
        << " selected=" << result.selected.size()
        << " uncovered=" << result.uncovered_operations.size() << '\n';
    out << "counters seeded=" << result.report.seeded << " rejected=" << result.report.rejected
        << " popped=" << result.report.popped << " stale=" << result.report.stale
        << " expanded=" << result.report.expanded << " invalidated=" << result.report.invalidated
        << " committed=" << result.report.committed << '\n';
    for (const auto & item : result.facts.facts()) {
        out << "fact " << item.first << '=' << fact_value_text(item.second.value) << " evidence=";
        for (const FactEvidence & evidence : item.second.evidence) {
            out << evidence.source;
            if (evidence.graph_id != kInvalidId) out << '#' << evidence.graph_id;
            out << ',';
        }
        out << '\n';
    }
    for (const FusionCandidate & candidate : result.selected) {
        const CandidateScore score = score_candidate(candidate);
        out << "select " << candidate.key << " family=" << candidate.family
            << " operations=" << candidate.operations.size()
            << " dispatches=" << candidate.economics.planned_dispatches
            << '/' << candidate.economics.reference_dispatches
            << " score=" << score.primary_benefit << '/' << score.secondary_benefit << '\n';
    }
    for (const SearchEvent & event : result.report.events) {
        out << "event " << event_kind_name(event.kind) << ' ' << event.candidate
            << " score=" << event.score.primary_benefit << '/' << event.score.secondary_benefit
            << " decision=" << decision_reason_name(event.decision.reason);
        if (!event.decision.detail.empty()) out << " detail=" << event.decision.detail;
        out << '\n';
    }
    for (const std::string & error : result.errors) out << "error " << error << '\n';
    return out.str();
}

std::string serialize_search_report_json(const SearchResult & result) {
    nlohmann::ordered_json root = {
        { "schema", "ggml-hrx-fusion-search-v1" },
        { "valid", result.valid() },
        { "facts", nlohmann::ordered_json::array() },
        { "selected", nlohmann::ordered_json::array() },
        { "uncovered_operations", result.uncovered_operations },
        { "errors", result.errors },
    };
    for (const auto & item : result.facts.facts()) {
        nlohmann::ordered_json fact = { { "key", item.first }, { "value", fact_value_text(item.second.value) },
                                        { "evidence", nlohmann::ordered_json::array() } };
        for (const FactEvidence & evidence : item.second.evidence) {
            fact["evidence"].push_back({ { "source", evidence.source }, { "graph_id", evidence.graph_id } });
        }
        root["facts"].push_back(std::move(fact));
    }
    for (const FusionCandidate & candidate : result.selected) {
        const CandidateScore score = score_candidate(candidate);
        root["selected"].push_back({
            { "key", candidate.key }, { "provider", candidate.provider }, { "family", candidate.family },
            { "hero", candidate.hero }, { "operations", candidate.operations },
            { "materialized_outputs", candidate.materialized_outputs },
            { "economics", {
                { "reference_dispatches", candidate.economics.reference_dispatches },
                { "planned_dispatches", candidate.economics.planned_dispatches },
                { "eliminated_materialization_bytes", candidate.economics.eliminated_materialization_bytes },
                { "additional_scratch_bytes", candidate.economics.additional_scratch_bytes },
                { "measured_nanoseconds_saved", candidate.economics.measured_nanoseconds_saved },
            } },
            { "score", { { "primary", score.primary_benefit }, { "secondary", score.secondary_benefit } } },
        });
    }
    root["counters"] = {
        { "seeded", result.report.seeded }, { "rejected", result.report.rejected },
        { "popped", result.report.popped }, { "stale", result.report.stale },
        { "expanded", result.report.expanded }, { "invalidated", result.report.invalidated },
        { "committed", result.report.committed },
    };
    return root.dump();
}

std::string fusion_region_dot(const GraphIndex & index, const SearchResult & result) {
    auto quoted = [](const std::string & text) {
        std::string escaped;
        escaped.reserve(text.size());
        for (char ch : text) {
            if (ch == '"' || ch == '\\') escaped.push_back('\\');
            escaped.push_back(ch == '\n' ? ' ' : ch);
        }
        return escaped;
    };
    std::vector<size_t> owner(index.graph().operations.size(), SIZE_MAX);
    for (size_t region = 0; region < result.selected.size(); ++region) {
        for (OperationId operation : result.selected[region].operations) {
            if (operation < owner.size()) owner[operation] = region;
        }
    }
    std::set<std::pair<size_t, size_t>> edges;
    for (OperationId operation = 0; operation < owner.size(); ++operation) {
        if (owner[operation] == SIZE_MAX) continue;
        for (OperationId successor : index.successors(operation)) {
            if (successor < owner.size() && owner[successor] != SIZE_MAX && owner[successor] != owner[operation]) {
                edges.emplace(owner[operation], owner[successor]);
            }
        }
    }
    std::ostringstream out;
    out << "digraph fusion_regions {\n  rankdir=LR;\n  node [shape=box,fontname=\"monospace\"];\n";
    for (size_t region = 0; region < result.selected.size(); ++region) {
        const FusionCandidate & candidate = result.selected[region];
        out << "  r" << region << " [label=\"" << quoted(candidate.family)
            << "\\nops=" << candidate.operations.size()
            << " dispatches=" << candidate.economics.planned_dispatches << "\"];\n";
    }
    for (const auto & edge : edges) out << "  r" << edge.first << " -> r" << edge.second << ";\n";
    out << "}\n";
    return out.str();
}

} // namespace ggml::hrx
