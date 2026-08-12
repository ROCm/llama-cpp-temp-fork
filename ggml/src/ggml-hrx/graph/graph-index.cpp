#include "graph-index.h"

#include <algorithm>
#include <iomanip>
#include <queue>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

static const std::vector<OperationId> kEmptyOperations;

class GraphIndexImplementation {
  public:
    static void add_edge(std::vector<std::vector<OperationId>> & predecessors,
                         std::vector<std::vector<OperationId>> & successors,
                         OperationId                             from,
                         OperationId                             to) {
        if (from == to || from >= successors.size() || to >= predecessors.size()) {
            return;
        }
        successors[from].push_back(to);
        predecessors[to].push_back(from);
    }

    static void canonicalize(std::vector<OperationId> & ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }

    static uint64_t hash_bytes(uint64_t hash, const void * data, size_t size) {
        const uint8_t * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    template <typename T> static uint64_t hash_value(uint64_t hash, const T & value) {
        return hash_bytes(hash, &value, sizeof(value));
    }

    static std::string format_hash(uint64_t hash) {
        std::ostringstream out;
        out << std::hex << std::setw(16) << std::setfill('0') << hash;
        return out.str();
    }
};

}  // namespace

Decision Decision::allow() {
    return { true, DecisionReason::Allowed, {}, {} };
}

Decision Decision::reject(DecisionReason reason, std::string detail, std::vector<uint32_t> implicated_ids) {
    return { false, reason, std::move(detail), std::move(implicated_ids) };
}

const char * Decision::reason_name(DecisionReason reason) {
    switch (reason) {
        case DecisionReason::Allowed:
            return "allowed";
        case DecisionReason::EmptyRegion:
            return "empty_region";
        case DecisionReason::InvalidOperation:
            return "invalid_operation";
        case DecisionReason::InvalidValue:
            return "invalid_value";
        case DecisionReason::DuplicateOperation:
            return "duplicate_operation";
        case DecisionReason::DisconnectedRegion:
            return "disconnected_region";
        case DecisionReason::ContractedCycle:
            return "contracted_cycle";
        case DecisionReason::MissingMaterialization:
            return "missing_materialization";
        case DecisionReason::Overlap:
            return "overlap";
        case DecisionReason::UnsupportedRecipe:
            return "unsupported_recipe";
        case DecisionReason::InconsistentFact:
            return "inconsistent_fact";
        case DecisionReason::NoComparableCost:
            return "no_comparable_cost";
        case DecisionReason::NoNativeCoverage:
            return "no_native_coverage";
        case DecisionReason::ProviderError:
            return "provider_error";
    }
    return "unknown";
}

GraphIndex::GraphIndex(const Graph & graph) : graph_(graph) {
    if (!graph.valid()) {
        errors_ = graph.errors;
        return;
    }
    consumers_.resize(graph.values.size());
    predecessors_.resize(graph.operations.size());
    successors_.resize(graph.operations.size());
    root_values_.resize(graph.values.size(), 0);
    structural_keys_.resize(graph.operations.size());

    for (ValueId root : graph.roots) {
        if (root < root_values_.size()) {
            root_values_[root] = 1;
        }
    }

    for (const Operation & operation : graph.operations) {
        if (operation.id >= graph.operations.size() || operation.output >= graph.values.size()) {
            errors_.push_back("graph index encountered an invalid operation identity");
            continue;
        }
        for (ValueId input : operation.inputs) {
            if (input >= graph.values.size()) {
                errors_.push_back("graph index encountered an invalid input value");
                continue;
            }
            consumers_[input].push_back(operation.id);
            const OperationId producer = graph.values[input].producer;
            if (producer != kInvalidId) {
                GraphIndexImplementation::add_edge(predecessors_, successors_, producer, operation.id);
            }
        }
        for (const Effect & effect : operation.effects) {
            if (effect.kind == EffectKind::Write) {
                const auto key = std::make_pair(effect.storage, effect.after_version);
                if (!writers_.emplace(key, operation.id).second) {
                    errors_.push_back("multiple operations write the same storage version");
                }
            }
        }
    }

    // Storage versions are true ordering edges even when GGML's value edges
    // pass through aliases or views.
    for (const Operation & operation : graph.operations) {
        for (const Effect & effect : operation.effects) {
            if (effect.before_version == 0) {
                continue;
            }
            const auto writer = writers_.find({ effect.storage, effect.before_version });
            if (writer != writers_.end()) {
                GraphIndexImplementation::add_edge(predecessors_, successors_, writer->second, operation.id);
            }
        }
    }
    for (auto & ids : consumers_) {
        GraphIndexImplementation::canonicalize(ids);
    }
    for (auto & ids : predecessors_) {
        GraphIndexImplementation::canonicalize(ids);
    }
    for (auto & ids : successors_) {
        GraphIndexImplementation::canonicalize(ids);
    }

    // Forward structural colors deliberately omit names and raw operation IDs.
    // The operation ID is used only as a final tie-break for truly symmetric
    // nodes, where selecting either node has the same observable semantics.
    std::vector<uint64_t> colors(graph.values.size(), UINT64_C(1469598103934665603));
    for (const Value & value : graph.values) {
        uint64_t color = UINT64_C(1469598103934665603);
        color          = GraphIndexImplementation::hash_value(color, value.type);
        color          = GraphIndexImplementation::hash_value(color, value.op);
        color = GraphIndexImplementation::hash_bytes(color, value.access.shape.data(), sizeof(value.access.shape));
        color = GraphIndexImplementation::hash_bytes(color, value.access.strides.data(), sizeof(value.access.strides));
        color = GraphIndexImplementation::hash_value(color, value.access.offset);
        color = GraphIndexImplementation::hash_value(color, value.access.version);
        colors[value.id] = color;
    }
    for (const Operation & operation : graph.operations) {
        uint64_t color = UINT64_C(1469598103934665603);
        color          = GraphIndexImplementation::hash_value(color, operation.op);
        color = GraphIndexImplementation::hash_bytes(color, operation.raw_params.data(), operation.raw_params.size());
        for (ValueId input : operation.inputs) {
            color = GraphIndexImplementation::hash_value(color, colors[input]);
        }
        for (const Effect & effect : operation.effects) {
            color = GraphIndexImplementation::hash_value(color, effect.kind);
            color = GraphIndexImplementation::hash_value(color, effect.before_version);
            color = GraphIndexImplementation::hash_value(color, effect.after_version);
            color = GraphIndexImplementation::hash_value(color, effect.offset);
            color = GraphIndexImplementation::hash_value(color, effect.size);
        }
        colors[operation.output] = color;
        structural_keys_[operation.id] =
            GraphIndexImplementation::format_hash(color) + ":" + ggml_op_name(operation.op);
    }
}

const std::vector<OperationId> & GraphIndex::consumers(ValueId value) const {
    return value < consumers_.size() ? consumers_[value] : kEmptyOperations;
}

const std::vector<OperationId> & GraphIndex::predecessors(OperationId operation) const {
    return operation < predecessors_.size() ? predecessors_[operation] : kEmptyOperations;
}

const std::vector<OperationId> & GraphIndex::successors(OperationId operation) const {
    return operation < successors_.size() ? successors_[operation] : kEmptyOperations;
}

OperationId GraphIndex::storage_writer(StorageId storage, uint32_t version) const {
    const auto writer = writers_.find({ storage, version });
    return writer == writers_.end() ? kInvalidId : writer->second;
}

const std::string & GraphIndex::structural_key(OperationId operation) const {
    static const std::string kInvalid = "invalid";
    return operation < structural_keys_.size() ? structural_keys_[operation] : kInvalid;
}

RegionBoundary GraphIndex::boundary(const std::vector<OperationId> & operations) const {
    RegionBoundary       result;
    std::vector<uint8_t> covered(graph_.operations.size(), 0);
    std::vector<uint8_t> boundary_values(graph_.values.size(), 0);
    for (OperationId operation : operations) {
        if (operation < covered.size()) {
            covered[operation] = 1;
        }
    }
    for (OperationId operation_id : operations) {
        if (operation_id >= graph_.operations.size()) {
            continue;
        }
        const Operation & operation = graph_.operations[operation_id];
        for (ValueId input : operation.inputs) {
            if (input >= graph_.values.size()) {
                continue;
            }
            const OperationId producer = graph_.values[input].producer;
            if (producer == kInvalidId || covered[producer] == 0) {
                boundary_values[input] |= 1;
            }
        }
        bool escapes = root_values_[operation.output] != 0;
        for (OperationId consumer : consumers(operation.output)) {
            escapes |= covered[consumer] == 0;
        }
        for (const Effect & effect : operation.effects) {
            escapes |= effect.kind == EffectKind::Write;
        }
        if (escapes) {
            boundary_values[operation.output] |= 2;
        }
    }
    for (ValueId value = 0; value < boundary_values.size(); ++value) {
        if ((boundary_values[value] & 1) != 0) {
            result.inputs.push_back(value);
        }
        if ((boundary_values[value] & 2) != 0) {
            result.outputs.push_back(value);
        }
    }
    return result;
}

Decision GraphIndex::validate_region(const std::vector<OperationId> & operations,
                                     const std::vector<ValueId> &     materialized_outputs,
                                     bool                             allow_disconnected) const {
    if (operations.empty()) {
        return Decision::reject(DecisionReason::EmptyRegion, "candidate covers no operations");
    }
    constexpr uint8_t    kCovered        = 1;
    constexpr uint8_t    kReached        = 2;
    constexpr uint8_t    kOutsideReached = 4;
    std::vector<uint8_t> operation_marks(graph_.operations.size(), 0);
    for (OperationId operation : operations) {
        if (operation >= graph_.operations.size()) {
            return Decision::reject(DecisionReason::InvalidOperation, "candidate references an invalid operation",
                                    { operation });
        }
        if ((operation_marks[operation] & kCovered) != 0) {
            return Decision::reject(DecisionReason::DuplicateOperation, "candidate repeats an operation",
                                    { operation });
        }
        operation_marks[operation] |= kCovered;
    }

    std::vector<uint8_t> materialized(graph_.values.size(), 0);
    for (ValueId output : materialized_outputs) {
        if (output >= graph_.values.size()) {
            return Decision::reject(DecisionReason::InvalidValue, "candidate materializes an invalid value",
                                    { output });
        }
        const OperationId producer = graph_.values[output].producer;
        if (producer == kInvalidId || (operation_marks[producer] & kCovered) == 0) {
            return Decision::reject(DecisionReason::InvalidValue, "candidate materializes a value it does not produce",
                                    { output });
        }
        materialized[output] = 1;
    }

    std::vector<OperationId> worklist;
    worklist.reserve(graph_.operations.size());
    if (!allow_disconnected) {
        worklist.push_back(operations.front());
        operation_marks[operations.front()] |= kReached;
        for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
            const OperationId current = worklist[cursor];
            auto              visit   = [&](OperationId adjacent) {
                if ((operation_marks[adjacent] & kCovered) != 0 && (operation_marks[adjacent] & kReached) == 0) {
                    operation_marks[adjacent] |= kReached;
                    worklist.push_back(adjacent);
                }
            };
            for (OperationId predecessor : predecessors(current)) {
                visit(predecessor);
            }
            for (OperationId successor : successors(current)) {
                visit(successor);
            }
        }
        if (worklist.size() != operations.size()) {
            return Decision::reject(DecisionReason::DisconnectedRegion,
                                    "candidate is disconnected in the data and storage dependency graph");
        }
    }

    // Contracting a non-convex set can introduce A->outside->A cycles. Detect
    // this directly by searching from every region successor through outside
    // operations for a path back into the region.
    worklist.clear();
    for (OperationId operation : operations) {
        for (OperationId successor : successors(operation)) {
            if ((operation_marks[successor] & (kCovered | kOutsideReached)) == 0) {
                operation_marks[successor] |= kOutsideReached;
                worklist.push_back(successor);
            }
        }
    }
    std::sort(worklist.begin(), worklist.end());
    for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
        const OperationId current = worklist[cursor];
        for (OperationId successor : successors(current)) {
            if ((operation_marks[successor] & kCovered) != 0) {
                return Decision::reject(DecisionReason::ContractedCycle,
                                        "contracting candidate operations creates a region cycle",
                                        { current, successor });
            }
            if ((operation_marks[successor] & kOutsideReached) == 0) {
                operation_marks[successor] |= kOutsideReached;
                worklist.push_back(successor);
            }
        }
    }

    std::vector<ValueId> required_outputs;
    for (OperationId operation_id : operations) {
        const Operation & operation = graph_.operations[operation_id];
        bool              escapes   = root_values_[operation.output] != 0;
        for (OperationId consumer : consumers(operation.output)) {
            escapes |= (operation_marks[consumer] & kCovered) == 0;
        }
        for (const Effect & effect : operation.effects) {
            escapes |= effect.kind == EffectKind::Write;
        }
        if (escapes) {
            required_outputs.push_back(operation.output);
        }
    }
    std::sort(required_outputs.begin(), required_outputs.end());
    required_outputs.erase(std::unique(required_outputs.begin(), required_outputs.end()), required_outputs.end());
    for (ValueId required : required_outputs) {
        if (materialized[required] == 0) {
            return Decision::reject(DecisionReason::MissingMaterialization,
                                    "candidate recipe does not materialize a required region output", { required });
        }
    }
    return Decision::allow();
}

Decision GraphIndex::topologically_order_regions(const std::vector<std::vector<OperationId>> & regions,
                                                 std::vector<size_t> &                         order) const {
    order.clear();
    std::vector<size_t> owner(graph_.operations.size(), SIZE_MAX);
    for (size_t region = 0; region < regions.size(); ++region) {
        for (OperationId operation : regions[region]) {
            if (operation >= owner.size()) {
                return Decision::reject(DecisionReason::InvalidOperation, "region references an invalid operation",
                                        { operation });
            }
            if (owner[operation] != SIZE_MAX) {
                return Decision::reject(DecisionReason::Overlap, "regions overlap", { operation });
            }
            owner[operation] = region;
        }
    }
    std::vector<std::set<size_t>> edges(regions.size());
    std::vector<size_t>           indegree(regions.size(), 0);
    for (OperationId from = 0; from < successors_.size(); ++from) {
        if (owner[from] == SIZE_MAX) {
            continue;
        }
        for (OperationId to : successors_[from]) {
            if (owner[to] == SIZE_MAX || owner[from] == owner[to]) {
                continue;
            }
            if (edges[owner[from]].insert(owner[to]).second) {
                ++indegree[owner[to]];
            }
        }
    }
    auto stable_key = [&](size_t region) {
        OperationId first_operation = kInvalidId;
        std::string key;
        for (OperationId operation : regions[region]) {
            first_operation = std::min(first_operation, operation);
            if (key.empty() || structural_key(operation) < key) {
                key = structural_key(operation);
            }
        }
        return std::make_pair(std::move(key), first_operation);
    };
    using Ready = std::pair<std::pair<std::string, OperationId>, size_t>;
    std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
    for (size_t region = 0; region < regions.size(); ++region) {
        if (indegree[region] == 0) {
            ready.push({ stable_key(region), region });
        }
    }
    while (!ready.empty()) {
        const size_t region = ready.top().second;
        ready.pop();
        order.push_back(region);
        for (size_t successor : edges[region]) {
            if (--indegree[successor] == 0) {
                ready.push({ stable_key(successor), successor });
            }
        }
    }
    if (order.size() != regions.size()) {
        order.clear();
        return Decision::reject(DecisionReason::ContractedCycle, "selected regions form a cyclic program");
    }
    return Decision::allow();
}

}  // namespace ggml::hrx
