#include "reactive-plan.h"

#include "hybrid-carrier-transformer.h"
#include "routed-transformer-bindings.h"
#include "routed-transformer-program.h"

#include "ggml-impl.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

static constexpr const char * kPlannerRevision = "reactive-plan-v2";

static void append_error(VerificationResult & result, const std::string & error) {
    result.errors.push_back(error);
}

static bool metadata_only(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
        op == GGML_OP_TRANSPOSE;
}

static Schedule atom_schedule(const Graph & graph) {
    Schedule result;
    result.graph_fingerprint = graph.fingerprint;
    result.workload = "atom-fallback";
    result.oracle_revision = "hrx-atom-recipes-v1";
    const GraphIndex index(graph);
    uint32_t dispatch_ordinal = 0;
    for (const Operation & operation : graph.operations) {
        Invocation invocation;
        invocation.stage = "atom";
        invocation.recipe = std::string("atom.") + ggml_op_name(operation.op);
        invocation.covered_operations.push_back(operation.id);
        invocation.kernel.family = metadata_only(operation.op) ? "hrx_metadata" : "hrx_atom";
        invocation.kernel.variant = ggml_op_name(operation.op);
        if (!metadata_only(operation.op)) invocation.kernel.kernel_id = kernel_catalog_id(
            invocation.kernel.family.c_str(), invocation.kernel.variant.c_str());
        invocation.kernel.execution_kind = KernelSpecialization::ExecutionKind::NativeEager;
        const RegionBoundary boundary = index.boundary(invocation.covered_operations);
        for (size_t i = 0; i < boundary.inputs.size(); ++i) {
            invocation.inputs.push_back({ "arg" + std::to_string(i), boundary.inputs[i] });
        }
        for (size_t i = 0; i < boundary.outputs.size(); ++i) {
            invocation.outputs.push_back({ "result" + std::to_string(i), boundary.outputs[i] });
        }
        if (!metadata_only(operation.op)) {
            Dispatch dispatch;
            dispatch.kernel = invocation.kernel;
            dispatch.bindings.insert(dispatch.bindings.end(), invocation.inputs.begin(), invocation.inputs.end());
            dispatch.bindings.insert(dispatch.bindings.end(), invocation.outputs.begin(), invocation.outputs.end());
            if (dispatch_ordinal != 0) dispatch.dependencies.push_back(dispatch_ordinal - 1);
            invocation.dispatches.push_back(std::move(dispatch));
            ++dispatch_ordinal;
        }
        result.invocations.push_back(std::move(invocation));
    }
    for (ValueId root : graph.roots) {
        result.roots.push_back({ root, RootDisposition::Materialized, "ggml_atom_result" });
    }
    result.expected_dispatch_count = dispatch_ordinal;
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
        case GGML_OP_CONCAT:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DIV:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_FILL:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_L2_NORM:
        case GGML_OP_LIGHTNING_INDEXER:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_PERMUTE:
        case GGML_OP_REPEAT:
        case GGML_OP_RESHAPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SCALE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SQRT:
        case GGML_OP_SUB:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_TOP_K:
        case GGML_OP_UNARY:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
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

    RoutedTransformerProgramProof structural = RoutedTransformerProgramProof::recover(graph);
    if (structural.structurally_recognized) {
        if (!structural.valid()) {
            result.errors = structural.errors;
            result.errors.insert(result.errors.end(), structural.search.errors.begin(), structural.search.errors.end());
            return result;
        }
        result.schedule = std::move(structural.schedule);
        result.atom_fallback_count = std::count_if(
            result.schedule.invocations.begin(), result.schedule.invocations.end(),
            [](const Invocation & invocation) { return invocation.recipe.rfind("atom.", 0) == 0; });
        for (const Invocation & invocation : result.schedule.invocations) {
            if (invocation.recipe.rfind("atom.", 0) == 0) {
                result.warnings.push_back("unoptimized routed-transformer fallback: " + invocation.recipe);
            }
        }
        result.planner_identity = RoutedTransformerProvider::make_planner().identity();
        result.fusion_search_text = SearchResult::format_report(structural.search);
        result.fusion_search_json = SearchResult::serialize_report_json(structural.search);
        result.fusion_regions_dot = SearchResult::region_dot(GraphIndex(graph), structural.search);
        result.logical_program_text = RoutedTransformerModel::format(*structural.logical_program);
        result.logical_program_json = RoutedTransformerModel::serialize_json(*structural.logical_program);
        result.logical_program_dot = RoutedTransformerModel::dot(*structural.logical_program);
        const VerificationResult bindings = RoutedTransformerProgramProof::materialize_dispatch_bindings(
            result.graph, result.schedule, *structural.logical_program);
        result.errors.insert(result.errors.end(), bindings.errors.begin(), bindings.errors.end());
        const VerificationResult bound_schedule = verify_schedule(result.graph, result.schedule);
        result.errors.insert(result.errors.end(), bound_schedule.errors.begin(), bound_schedule.errors.end());
        if (!result.errors.empty()) return result;

    } else {
        HybridCarrierTransformerProgramProof hybrid =
            HybridCarrierTransformerProgramProof::recover(result.graph, target);
        if (hybrid.structurally_recognized) {
            if (!hybrid.valid()) {
                result.errors = hybrid.errors;
                result.errors.insert(result.errors.end(), hybrid.search.errors.begin(), hybrid.search.errors.end());
                return result;
            }
            result.schedule = std::move(hybrid.schedule);
            result.atom_fallback_count = std::count_if(
                result.schedule.invocations.begin(), result.schedule.invocations.end(),
                [](const Invocation & invocation) { return invocation.recipe.rfind("atom.", 0) == 0; });
            result.planner_identity = HybridCarrierTransformerProvider::make_planner().identity();
            result.fusion_search_text = SearchResult::format_report(hybrid.search);
            result.fusion_search_json = SearchResult::serialize_report_json(hybrid.search);
            result.fusion_regions_dot = SearchResult::region_dot(GraphIndex(graph), hybrid.search);
            result.logical_program_text = HybridCarrierTransformerModel::format(*hybrid.logical_program);
            result.logical_program_json = HybridCarrierTransformerModel::serialize_json(*hybrid.logical_program);
            result.logical_program_dot = HybridCarrierTransformerModel::dot(*hybrid.logical_program);
            if (result.atom_fallback_count != 0) {
                result.warnings.push_back("hybrid-carrier structure recognized; the exact plan contains " +
                                          std::to_string(result.atom_fallback_count) +
                                          " native-eager atom recipes");
            }
        } else {
        result.schedule = atom_schedule(graph);
        result.planner_identity = "atom-recipes-v1";
        result.atom_fallback_count = graph.operations.size();
        for (const std::string & error : structural.errors) {
            result.warnings.push_back("routed-transformer recognition rejected: " + error);
        }
        result.warnings.push_back("routed-transformer structure not recognized; the plan contains " +
                                  std::to_string(result.atom_fallback_count) + " native-eager atom recipes");
        const VerificationResult schedule_verification = verify_schedule(graph, result.schedule);
        result.errors.insert(result.errors.end(), schedule_verification.errors.begin(), schedule_verification.errors.end());
        }
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
    if (cgraph == nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.failures;
        frame.errors.push_back("HRX cannot prepare a null ggml_cgraph");
        return frame;
    }
    const uint64_t uid = cgraph->uid;
    if (uid != 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto position = plans_.find(uid);
        if (position != plans_.end()) {
            const UidPlanEntry & entry = position->second;
            if (entry.target != target) {
                ++stats_.failures;
                frame.errors.push_back("HRX graph UID " + std::to_string(uid) +
                    " was reused with target '" + target + "' after being planned for target '" + entry.target + "'");
                return frame;
            }
            ++stats_.hits;
            frame.plan = entry.plan;
            frame.values = entry.values;
            frame.storage_roots = entry.storage_roots;
            return frame;
        }
    }

    ImportedGraph imported = ImportedGraph::import(cgraph);
    if (!imported.graph.valid()) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.failures;
        frame.errors = imported.graph.errors;
        return frame;
    }

    // UID zero is the ggml convention for a graph without a stable scheduler
    // identity. Such graphs are valid, but there is no sound cache key for
    // them: import and plan this execution without either querying or
    // publishing the UID cache.
    if (uid == 0) {
        ProgramPlan plan = build_reactive_plan(imported.graph, target);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.builds;
            if (!plan.valid()) ++stats_.failures;
        }
        if (!plan.valid()) {
            frame.errors = plan.errors;
            return frame;
        }
        frame.plan = std::make_shared<const ProgramPlan>(std::move(plan));
        frame.values = std::move(imported.value_tensors);
        frame.storage_roots = std::move(imported.storage_roots);
        frame.values.resize(frame.plan->graph.values.size(), nullptr);
        frame.storage_roots.resize(frame.plan->graph.storages.size(), nullptr);
        return frame;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = plans_.find(uid);
    if (existing != plans_.end()) {
        const UidPlanEntry & entry = existing->second;
        if (entry.target != target) {
            ++stats_.failures;
            frame.errors.push_back("HRX graph UID " + std::to_string(uid) +
                " was reused with target '" + target + "' after being planned for target '" + entry.target + "'");
            return frame;
        }
        ++stats_.hits;
        frame.plan = entry.plan;
        frame.values = entry.values;
        frame.storage_roots = entry.storage_roots;
        return frame;
    }

    ProgramPlan plan = build_reactive_plan(imported.graph, target);
    ++stats_.builds;
    if (!plan.valid()) {
        ++stats_.failures;
        frame.errors = plan.errors;
        return frame;
    }
    UidPlanEntry entry;
    entry.target = target;
    entry.plan = std::make_shared<const ProgramPlan>(std::move(plan));
    entry.values = std::move(imported.value_tensors);
    entry.storage_roots = std::move(imported.storage_roots);
    entry.values.resize(entry.plan->graph.values.size(), nullptr);
    entry.storage_roots.resize(entry.plan->graph.storages.size(), nullptr);
    frame.plan = entry.plan;
    frame.values = entry.values;
    frame.storage_roots = entry.storage_roots;
    plans_.emplace(uid, std::move(entry));
    return frame;
}

PlanCacheStats ReactivePlanCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace ggml::hrx
