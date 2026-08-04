#include "reactive-plan.h"

#include "optimizer.h"
#include "qwen-bindings.h"
#include "qwen-program.h"
#include "qwen-rules.h"
#include "routed-transformer-bindings.h"
#include "routed-transformer-program.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

static constexpr const char * kPlannerRevision = "reactive-plan-v2";

static bool same_access(const AccessPath & lhs, const AccessPath & rhs) {
    return lhs.storage == rhs.storage && lhs.version == rhs.version && lhs.offset == rhs.offset &&
        lhs.shape == rhs.shape && lhs.strides == rhs.strides;
}

static bool same_effect(const Effect & lhs, const Effect & rhs) {
    return lhs.kind == rhs.kind && lhs.storage == rhs.storage && lhs.before_version == rhs.before_version &&
        lhs.after_version == rhs.after_version && lhs.offset == rhs.offset && lhs.size == rhs.size && lhs.exact == rhs.exact;
}

static void append_error(VerificationResult & result, const std::string & error) {
    result.errors.push_back(error);
}

static bool same_kernel(const KernelSpecialization & lhs, const KernelSpecialization & rhs) {
    return lhs.family == rhs.family && lhs.variant == rhs.variant &&
        lhs.integer_parameters == rhs.integer_parameters &&
        lhs.execution_kind == rhs.execution_kind &&
        lhs.compile_parameters == rhs.compile_parameters;
}

static VerificationResult compare_bound_schedules(const Schedule & oracle, const Schedule & candidate) {
    VerificationResult result;
    if (oracle.workload != candidate.workload) append_error(result, "legacy oracle workload differs");
    if (oracle.roots.size() != candidate.roots.size()) append_error(result, "legacy oracle root count differs");
    for (size_t i = 0; i < std::min(oracle.roots.size(), candidate.roots.size()); ++i) {
        if (oracle.roots[i].value != candidate.roots[i].value ||
            oracle.roots[i].disposition != candidate.roots[i].disposition ||
            oracle.roots[i].replacement != candidate.roots[i].replacement) {
            append_error(result, "legacy oracle root " + std::to_string(i) + " differs");
        }
    }
    if (oracle.invocations.size() != candidate.invocations.size()) {
        append_error(result, "legacy oracle invocation count differs");
        return result;
    }
    size_t dispatch_ordinal = 0;
    for (size_t i = 0; i < oracle.invocations.size(); ++i) {
        const Invocation & expected = oracle.invocations[i];
        const Invocation & actual = candidate.invocations[i];
        // Region ownership is intentionally allowed to improve: the legacy
        // program assigned each following RMSNorm to the preceding layer.
        // Compare the executable contract instead of preserving that artifact.
        if (expected.layer != actual.layer) append_error(
            result, "legacy oracle invocation layer " + std::to_string(i) + " differs");
        if (expected.dispatches.size() != actual.dispatches.size()) {
            append_error(result, "legacy oracle dispatch count differs in invocation " + std::to_string(i));
            dispatch_ordinal += actual.dispatches.size();
            continue;
        }
        for (size_t j = 0; j < expected.dispatches.size(); ++j, ++dispatch_ordinal) {
            const Dispatch & expected_dispatch = expected.dispatches[j];
            const Dispatch & actual_dispatch = actual.dispatches[j];
            if (!same_kernel(expected_dispatch.kernel, actual_dispatch.kernel) ||
                expected_dispatch.bindings != actual_dispatch.bindings ||
                expected_dispatch.dependencies != actual_dispatch.dependencies) {
                append_error(result, "legacy oracle dispatch " + std::to_string(dispatch_ordinal) + " differs");
            }
        }
    }
    return result;
}

static Schedule eager_schedule(const Graph & graph) {
    const std::vector<FusionRule> rules = canonical_qwen3_moe_rules();
    const Selection selection = select_regions(graph, rules);
    Schedule result = materialize_schedule_with_cpu_fallback(graph, rules, selection);
    for (Invocation & invocation : result.invocations) {
        if (invocation.kernel.execution_kind == KernelSpecialization::ExecutionKind::CpuFallback) {
            invocation.kernel.execution_kind = KernelSpecialization::ExecutionKind::NativeEager;
            invocation.kernel.family = "hrx_eager";
            invocation.kernel.variant = ggml_op_name(graph.operations[invocation.covered_operations.front()].op);
            invocation.kernel.kernel_id = kernel_catalog_id(invocation.kernel.family.c_str(), invocation.kernel.variant.c_str());
        }
        for (Dispatch & dispatch : invocation.dispatches) {
            if (dispatch.kernel.execution_kind == KernelSpecialization::ExecutionKind::CpuFallback) {
                dispatch.kernel = invocation.kernel;
            }
        }
    }
    return result;
}

} // namespace

bool eager_capability_declared(enum ggml_op op) {
    switch (op) {
        // The scheduler probes preallocated weight tensors as NONE operations
        // when deciding whether their buffer type is usable by this backend.
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_ARGSORT:
        case GGML_OP_CLAMP:
        case GGML_OP_DIV:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
}

bool graph_semantically_equal(const Graph & lhs, const Graph & rhs) {
    if (lhs.storages.size() < rhs.storages.size() || lhs.values.size() < rhs.values.size() ||
        lhs.operations.size() != rhs.operations.size() || lhs.roots != rhs.roots) return false;
    for (size_t i = rhs.values.size(); i < lhs.values.size(); ++i) {
        if (lhs.values[i].name.rfind("hrx.synthetic.", 0) != 0) return false;
    }
    for (size_t i = 0; i < rhs.storages.size(); ++i) {
        const Storage & a = lhs.storages[i];
        const Storage & b = rhs.storages[i];
        if (a.id != b.id || a.root != b.root || a.size != b.size || a.external != b.external ||
            a.weight != b.weight || a.mutable_state != b.mutable_state || a.final_version != b.final_version) return false;
    }
    for (size_t i = 0; i < rhs.values.size(); ++i) {
        const Value & a = lhs.values[i];
        const Value & b = rhs.values[i];
        if (a.id != b.id || a.type != b.type || a.op != b.op || a.boundary != b.boundary ||
            a.producer != b.producer || a.view_source != b.view_source || !same_access(a.access, b.access)) return false;
    }
    for (size_t i = 0; i < lhs.operations.size(); ++i) {
        const Operation & a = lhs.operations[i];
        const Operation & b = rhs.operations[i];
        if (a.id != b.id || a.op != b.op || a.inputs != b.inputs || a.output != b.output ||
            a.raw_params != b.raw_params || a.effects.size() != b.effects.size()) return false;
        for (size_t j = 0; j < a.effects.size(); ++j) if (!same_effect(a.effects[j], b.effects[j])) return false;
    }
    return true;
}

ResourceProgram build_resource_program(const Graph & graph, const Schedule & schedule) {
    ResourceProgram result;
    result.resources.resize(graph.storages.size());
    std::set<StorageId> root_storages;
    for (ValueId root : graph.roots) if (root < graph.values.size()) root_storages.insert(graph.values[root].access.storage);
    for (const Storage & storage : graph.storages) {
        ResourceContract & resource = result.resources[storage.id];
        resource.storage = storage.id;
        resource.size = storage.size;
        resource.imported = storage.external;
        resource.weight = storage.weight;
        resource.mutable_state = storage.mutable_state;
        resource.exported = storage.mutable_state || root_storages.count(storage.id) != 0;
        resource.elidable = !resource.imported && !resource.exported;
        resource.final_version = storage.final_version;
    }
    for (const Value & value : graph.values) {
        if (value.access.storage < result.resources.size()) result.resources[value.access.storage].aliases.push_back(value.id);
    }
    for (uint32_t invocation_id = 0; invocation_id < schedule.invocations.size(); ++invocation_id) {
        const Invocation & invocation = schedule.invocations[invocation_id];
        std::map<StorageId, ResourceUse> uses;
        auto merge_use = [&](StorageId storage, uint32_t before_version, uint32_t after_version, ResourceAccess access) {
            auto [position, inserted] = uses.try_emplace(storage);
            ResourceUse & use = position->second;
            if (inserted) {
                use.invocation = invocation_id;
                use.storage = storage;
                use.before_version = before_version;
                use.after_version = after_version;
                use.access = access;
            } else {
                use.before_version = std::min(use.before_version, before_version);
                use.after_version = std::max(use.after_version, after_version);
                if (use.access != access) use.access = ResourceAccess::ReadWrite;
            }
        };
        for (OperationId operation_id : invocation.covered_operations) {
            if (operation_id >= graph.operations.size()) continue;
            const Operation & operation = graph.operations[operation_id];
            bool output_write_recorded = false;
            const StorageId output_storage = graph.values[operation.output].access.storage;
            for (const Effect & effect : operation.effects) {
                const ResourceAccess access = effect.kind == EffectKind::Write ? ResourceAccess::Write : ResourceAccess::Read;
                merge_use(effect.storage, effect.before_version, effect.after_version, access);
                output_write_recorded |= effect.kind == EffectKind::Write && effect.storage == output_storage;
            }
            const bool layout_only = operation.op == GGML_OP_VIEW || operation.op == GGML_OP_RESHAPE ||
                operation.op == GGML_OP_PERMUTE || operation.op == GGML_OP_TRANSPOSE;
            if (!layout_only && !output_write_recorded) {
                const uint32_t version = graph.values[operation.output].access.version;
                merge_use(output_storage, version, version, ResourceAccess::Write);
            }
        }
        for (const Dispatch & dispatch : invocation.dispatches) {
            for (const TensorBinding & binding : dispatch.bindings) {
                if (binding.value >= graph.values.size()) continue;
                const Value & value = graph.values[binding.value];
                if (value.name.rfind("hrx.synthetic.", 0) == 0) {
                    merge_use(value.access.storage, 0, 0, ResourceAccess::ReadWrite);
                }
            }
        }
        for (auto & item : uses) {
            ResourceContract & resource = result.resources[item.first];
            resource.first_invocation = std::min(resource.first_invocation, invocation_id);
            resource.last_invocation = std::max(resource.last_invocation, invocation_id);
            result.uses.push_back(item.second);
        }
    }
    return result;
}

VerificationResult verify_resource_program(const Graph & graph, const Schedule & schedule, const ResourceProgram & resources) {
    VerificationResult result;
    if (resources.resources.size() != graph.storages.size()) {
        append_error(result, "resource program does not cover every logical storage");
        return result;
    }
    std::vector<uint32_t> observed_version(graph.storages.size(), 0);
    std::vector<bool> observed_use(graph.storages.size(), false);
    for (size_t i = 0; i < resources.resources.size(); ++i) {
        const ResourceContract & resource = resources.resources[i];
        const Storage & storage = graph.storages[i];
        if (resource.storage != i || resource.size != storage.size) append_error(result, "resource identity or size mismatch");
        if (storage.external && !resource.imported) append_error(result, "external storage is not imported");
        if (storage.mutable_state && !resource.exported) append_error(result, "mutable storage is not exported");
        if (resource.elidable && (resource.imported || resource.exported)) append_error(result, "boundary storage is marked elidable");
        for (ValueId alias : resource.aliases) {
            if (alias >= graph.values.size() || graph.values[alias].access.storage != i) append_error(result, "resource alias escapes its storage root");
        }
    }
    uint32_t last_invocation = 0;
    for (const ResourceUse & use : resources.uses) {
        if (use.storage >= graph.storages.size() || use.invocation >= schedule.invocations.size()) {
            append_error(result, "resource use references an invalid storage or invocation");
            continue;
        }
        if (use.invocation < last_invocation) append_error(result, "resource uses are not invocation ordered");
        last_invocation = use.invocation;
        if (use.before_version > use.after_version || use.before_version > graph.storages[use.storage].final_version ||
            use.after_version > graph.storages[use.storage].final_version) append_error(result, "resource use has an invalid storage version");
        observed_use[use.storage] = true;
        observed_version[use.storage] = std::max(observed_version[use.storage], use.after_version);
    }
    for (const Storage & storage : graph.storages) {
        if (!observed_use[storage.id]) append_error(result, "logical storage has no scheduled use");
        if (observed_version[storage.id] != storage.final_version) append_error(result, "resource program does not reach the final storage version");
    }
    return result;
}

std::string schedule_semantic_witness(const Graph & graph, const Schedule & schedule) {
    std::ostringstream out;
    out << kPlannerRevision << '\n' << schedule.workload << '\n' << graph.operations.size() << '\n';
    auto value_witness = [&](ValueId value_id) {
        if (value_id >= graph.values.size()) return std::string("invalid");
        const Value & value = graph.values[value_id];
        std::ostringstream value_out;
        value_out << ggml_type_name(value.type) << '@' << value.access.offset << 'v' << value.access.version << '[';
        for (int64_t dimension : value.access.shape) value_out << dimension << ',';
        value_out << "]p";
        if (value.producer == kInvalidId) value_out << "external";
        else value_out << graph.operations[value.producer].original_ordinal;
        return value_out.str();
    };
    for (const Operation & operation : graph.operations) {
        out << "op:" << operation.original_ordinal << ':' << ggml_op_name(operation.op) << ':';
        for (ValueId input : operation.inputs) out << value_witness(input) << ';';
        out << "->" << value_witness(operation.output) << ':';
        for (uint8_t byte : operation.raw_params) out << static_cast<unsigned>(byte) << ',';
        out << '\n';
    }
    for (const Invocation & invocation : schedule.invocations) {
        out << invocation.stage << ':' << invocation.layer << ':';
        for (OperationId operation : invocation.covered_operations) out << operation << ',';
        out << '\n';
        for (const TensorBinding & input : invocation.inputs) out << "in:" << input.role << ':' << value_witness(input.value) << '\n';
        for (const TensorBinding & output : invocation.outputs) out << "out:" << output.role << ':' << value_witness(output.value) << '\n';
        for (const Dispatch & dispatch : invocation.dispatches) {
            out << execution_kind_name(dispatch.kernel.execution_kind) << ':' << kernel_specialization_name(dispatch.kernel);
            for (const auto & parameter : dispatch.kernel.integer_parameters) out << ':' << parameter.first << '=' << parameter.second;
            out << ':';
            for (uint32_t dependency : dispatch.dependencies) out << dependency << ',';
            out << '\n';
        }
    }
    for (const RootContract & root : schedule.roots) out << "root:" << root_disposition_name(root.disposition) << ':' << root.replacement << '\n';
    return out.str();
}

ProgramPlan build_reactive_plan(const Graph & graph, const std::string & target) {
    ProgramPlan result;
    result.graph = graph;
    result.target = target;
    if (!graph.valid()) {
        result.errors = graph.errors;
        return result;
    }
    for (const Operation & operation : graph.operations) {
        if (!eager_capability_declared(operation.op)) {
            result.errors.push_back(std::string("no declared HRX eager capability for ") + ggml_op_name(operation.op));
        }
    }
    if (!result.errors.empty()) return result;

    RoutedTransformerProgramProof structural = recover_structural_routed_transformer_program(graph);
    if (structural.structurally_recognized) {
        if (!structural.valid()) {
            result.errors = structural.errors;
            result.errors.insert(result.errors.end(), structural.search.errors.begin(), structural.search.errors.end());
            return result;
        }
        result.schedule = std::move(structural.schedule);
        result.planner_identity = make_structural_routed_transformer_planner().identity();
        result.fusion_search_text = format_search_report(structural.search);
        result.fusion_search_json = serialize_search_report_json(structural.search);
        result.fusion_regions_dot = fusion_region_dot(GraphIndex(graph), structural.search);
        const VerificationResult bindings = materialize_routed_transformer_dispatch_bindings(result.graph, result.schedule);
        result.errors.insert(result.errors.end(), bindings.errors.begin(), bindings.errors.end());
        const VerificationResult bound_schedule = verify_schedule(result.graph, result.schedule);
        result.errors.insert(result.errors.end(), bound_schedule.errors.begin(), bound_schedule.errors.end());
        if (!result.errors.empty()) return result;

        // The positional implementation remains an independent cold-plan
        // oracle while this cutover settles. It is never consulted during a
        // cached execution. Compatible graphs outside its narrow model shape
        // continue using the structural authority without inheriting its
        // operation-count and ordinal assumptions.
        const QwenProgramProof oracle = recover_owned_qwen3_moe_program(graph);
        if (oracle.recognized()) {
            Graph oracle_graph = graph;
            Schedule oracle_schedule = oracle.schedule;
            const VerificationResult oracle_bindings =
                materialize_qwen3_moe_dispatch_bindings(oracle_graph, oracle_schedule);
            result.errors.insert(result.errors.end(), oracle_bindings.errors.begin(), oracle_bindings.errors.end());
            if (!graph_semantically_equal(result.graph, oracle_graph) ||
                !graph_semantically_equal(oracle_graph, result.graph)) {
                result.errors.push_back("structural planner scratch graph differs from the legacy oracle");
            }
            const VerificationResult equivalence = compare_bound_schedules(oracle_schedule, result.schedule);
            result.errors.insert(result.errors.end(), equivalence.errors.begin(), equivalence.errors.end());
            result.legacy_oracle_equivalent = equivalence.valid() && oracle_bindings.valid() && result.errors.empty();
        }
    } else {
        result.schedule = eager_schedule(graph);
        result.planner_identity = "eager-rules-v1";
        const VerificationResult schedule_verification = verify_schedule(graph, result.schedule);
        result.errors.insert(result.errors.end(), schedule_verification.errors.begin(), schedule_verification.errors.end());
    }
    if (!result.errors.empty()) return result;
    result.resources = build_resource_program(result.graph, result.schedule);
    const VerificationResult resource_verification = verify_resource_program(result.graph, result.schedule, result.resources);
    result.errors.insert(result.errors.end(), resource_verification.errors.begin(), resource_verification.errors.end());
    result.semantic_witness = schedule_semantic_witness(result.graph, result.schedule);
    return result;
}

ExecutionFrame ReactivePlanCache::prepare(const ggml_cgraph * cgraph, const std::string & target) {
    ExecutionFrame frame;
    ImportedGraph imported = import_graph_with_bindings(cgraph);
    if (!imported.graph.valid()) {
        frame.errors = imported.graph.errors;
        return frame;
    }
    const std::string key = imported.graph.fingerprint + '|' + target + '|' + kPlannerRevision;
    std::lock_guard<std::mutex> lock(mutex_);
    auto & bucket = plans_[key];
    for (const std::shared_ptr<const ProgramPlan> & candidate : bucket) {
        if (graph_semantically_equal(candidate->graph, imported.graph)) {
            ++stats_.hits;
            frame.plan = candidate;
            frame.values = std::move(imported.value_tensors);
            frame.storage_roots = std::move(imported.storage_roots);
            frame.values.resize(candidate->graph.values.size(), nullptr);
            frame.storage_roots.resize(candidate->graph.storages.size(), nullptr);
            return frame;
        }
        ++stats_.semantic_collisions;
    }
    ProgramPlan plan = build_reactive_plan(imported.graph, target);
    ++stats_.builds;
    if (!plan.valid()) {
        ++stats_.failures;
        frame.errors = plan.errors;
        return frame;
    }
    frame.plan = std::make_shared<const ProgramPlan>(std::move(plan));
    bucket.push_back(frame.plan);
    frame.values = std::move(imported.value_tensors);
    frame.storage_roots = std::move(imported.storage_roots);
    frame.values.resize(frame.plan->graph.values.size(), nullptr);
    frame.storage_roots.resize(frame.plan->graph.storages.size(), nullptr);
    return frame;
}

PlanCacheStats ReactivePlanCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace ggml::hrx
