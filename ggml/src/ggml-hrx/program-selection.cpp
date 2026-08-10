#include "program-selection.h"

#include "domains/llm-patterns.h"
#include "domains/llm-recipes.h"
#include "ggml-impl.h"
#include "matcher.h"
#include "planner.h"
#include "primitive-capabilities.h"

#include <algorithm>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

bool bind_source_graph(const ggml_cgraph *                graph,
                       const graph_plan &                 plan,
                       std::vector<const ggml_tensor *> & values,
                       std::vector<const ggml_tensor *> & storage_roots,
                       std::vector<std::string> &         errors) {
    values.assign(plan.values().size(), nullptr);
    for (const graph_value & value : plan.values()) {
        const source_tensor_ref source = value.source;
        if (source.kind == source_tensor_kind::SOURCE_TENSOR_KIND_LEAF &&
            source.ordinal < static_cast<uint32_t>(graph->n_leafs)) {
            values[value.id] = graph->leafs[source.ordinal];
        } else if (source.kind == source_tensor_kind::SOURCE_TENSOR_KIND_NODE &&
                   source.ordinal < static_cast<uint32_t>(graph->n_nodes)) {
            values[value.id] = graph->nodes[source.ordinal];
        } else if (source.kind == source_tensor_kind::SOURCE_TENSOR_KIND_NODE_INPUT &&
                   source.ordinal < static_cast<uint32_t>(graph->n_nodes) && source.slot < GGML_MAX_SRC) {
            values[value.id] = graph->nodes[source.ordinal]->src[source.slot];
        } else {
            errors.push_back("cached graph source slot is absent for value " + std::to_string(value.id));
        }
    }
    storage_roots.assign(plan.storages().size(), nullptr);
    for (const graph_storage & storage : plan.storages()) {
        if (storage.root >= values.size() || values[storage.root] == nullptr) {
            errors.push_back("cached graph source slot is absent for storage " + std::to_string(storage.id));
        } else {
            storage_roots[storage.id] = values[storage.root];
        }
    }
    return errors.empty();
}

}  // namespace

program_selection program_selection::select(const graph_plan & plan, const std::string & target) {
    program_selection result;
    if (!plan.valid() || plan.phase() != plan_phase::PLAN_PHASE_PLANNED) {
        result.errors.push_back("program selection requires a valid planned graph");
        return result;
    }
    const auto query   = plan.facts().find("query_token_count");
    const auto context = plan.facts().find("key_value_token_count");
    const auto output  = plan.facts().find("output_token_count");
    const bool has_workload_facts =
        query != plan.facts().end() && context != plan.facts().end() && output != plan.facts().end();
    const bool loom_decode_576 = has_workload_facts && llm_recipes::supports_qwen3_30b_decode_576(plan.facts());
    result.implementation      = loom_decode_576 ? program_implementation::PROGRAM_IMPLEMENTATION_LOOM :
                                                   program_implementation::PROGRAM_IMPLEMENTATION_TRANSITIONAL;
    result.library = loom_decode_576 ? "hrx_system.experimental.qwen.command_programs" : "ggml_hrx_transitional_llm";
    if (has_workload_facts) {
        if (loom_decode_576) {
            result.roots.push_back("qwen3_30b_decode_576");
        } else {
            const std::string workload = query->second == 1 ? "decode" : "prefill";
            result.roots.push_back(workload + "_q" + std::to_string(query->second) + "_kv" +
                                   std::to_string(context->second) + "_out" + std::to_string(output->second));
        }
    } else {
        result.roots.push_back("primitive");
    }
    result.source_configuration = plan.facts();
    result.target_specialization.emplace("architecture", target);

    fixed_parameter_root parameters;
    parameters.role = "parameters";
    for (const graph_storage & storage : plan.storages()) {
        if (storage.weight) {
            parameters.storages.push_back(storage.id);
        }
    }
    result.fixed_parameters.push_back(std::move(parameters));
    if (loom_decode_576) {
        result.fixed_parameters.push_back({ "auxiliary_parameters", {} });
    }

    std::set<storage_id> key_caches;
    std::set<storage_id> value_caches;
    for (const graph_region & region : plan.regions()) {
        if (const auto * qkv = std::get_if<attention_qkv_region>(&region.payload)) {
            key_caches.insert(qkv->key_cache);
            value_caches.insert(qkv->value_cache);
        }
    }
    std::vector<storage_id> cache_overlap;
    std::set_intersection(key_caches.begin(), key_caches.end(), value_caches.begin(), value_caches.end(),
                          std::back_inserter(cache_overlap));
    if ((!key_caches.empty() || !value_caches.empty()) &&
        (key_caches.size() != value_caches.size() || !cache_overlap.empty())) {
        result.errors.push_back("program selection found incoherent key and value cache resources");
    }
    for (storage_id storage : key_caches) {
        result.runtime_resources.push_back({ "key_cache", storage, ID_INVALID });
    }
    for (storage_id storage : value_caches) {
        result.runtime_resources.push_back({ "value_cache", storage, ID_INVALID });
    }
    for (const graph_value & value : plan.values()) {
        if (value.boundary == boundary_kind::BOUNDARY_KIND_INPUT) {
            result.runtime_resources.push_back(
                { loom_decode_576 ? "request_state" : "input", value.access.storage, value.id });
        } else if (value.boundary == boundary_kind::BOUNDARY_KIND_OUTPUT) {
            result.runtime_resources.push_back(
                { loom_decode_576 ? "output_staging" : "output", value.access.storage, value.id });
        }
    }
    std::sort(result.runtime_resources.begin(), result.runtime_resources.end(),
              [](const runtime_resource & lhs, const runtime_resource & rhs) {
                  if (lhs.role != rhs.role) {
                      return lhs.role < rhs.role;
                  }
                  if (lhs.storage != rhs.storage) {
                      return lhs.storage < rhs.storage;
                  }
                  return lhs.value < rhs.value;
              });
    result.runtime_resources.erase(std::unique(result.runtime_resources.begin(), result.runtime_resources.end(),
                                               [](const runtime_resource & lhs, const runtime_resource & rhs) {
                                                   return lhs.role == rhs.role && lhs.storage == rhs.storage &&
                                                          lhs.value == rhs.value;
                                               }),
                                   result.runtime_resources.end());

    std::set<op_id> covered;
    for (region_id id : plan.active_regions()) {
        const graph_region & region = plan.regions()[id];
        result.coverage.regions.push_back(id);
        for (op_id operation : region.source_ops) {
            if (!covered.insert(operation).second) {
                result.errors.push_back("program coverage overlaps an operation");
            }
        }
        if (region.selected_recipe == ID_INVALID) {
            for (op_id operation : region.source_ops) {
                result.missing.push_back({ id, operation, plan.operations()[operation].op, "no selected recipe" });
            }
        }
        if (region.kind == region_kind::REGION_KIND_ATOM) {
            const op_id operation = std::get<atom_region>(region.payload).operation;
            if (!primitive_capability_declared(plan.operations()[operation].op)) {
                result.missing.push_back({ id, operation, plan.operations()[operation].op, "no primitive capability" });
            }
        }
    }
    result.coverage.operations.assign(covered.begin(), covered.end());
    if (result.coverage.operations.size() != plan.operations().size()) {
        result.errors.push_back("program selection does not cover the complete graph");
    }
    return result;
}

std::string program_selection::format() const {
    std::ostringstream out;
    out << "program-selection implementation="
        << (implementation == program_implementation::PROGRAM_IMPLEMENTATION_TRANSITIONAL ? "transitional" : "loom")
        << " library=" << library << " roots=" << roots.size() << " operations=" << coverage.operations.size()
        << " missing=" << missing.size() << '\n';
    for (const std::string & root : roots) {
        out << "root=" << root << '\n';
    }
    for (const auto & fact : source_configuration) {
        out << "config " << fact.first << '=' << fact.second << '\n';
    }
    for (const fixed_parameter_root & root : fixed_parameters) {
        out << "fixed " << root.role << " storages=" << root.storages.size() << '\n';
    }
    for (const runtime_resource & resource : runtime_resources) {
        out << "runtime " << resource.role << " storage=" << resource.storage << " value=" << resource.value << '\n';
    }
    for (const missing_program_step & step : missing) {
        out << "missing region=" << step.region << " op=" << step.operation << " kind=" << ggml_op_name(step.op)
            << " reason=" << step.reason << '\n';
    }
    for (const std::string & error : errors) {
        out << "error=" << error << '\n';
    }
    return out.str();
}

std::string program_selection::serialize_json() const {
    auto escape = [](const std::string & value) {
        std::string result;
        for (char character : value) {
            if (character == '"' || character == '\\') {
                result.push_back('\\');
            }
            result.push_back(character);
        }
        return result;
    };
    std::ostringstream out;
    out << "{\"schema\":\"ggml-hrx-program-selection-v1\",\"implementation\":\""
        << (implementation == program_implementation::PROGRAM_IMPLEMENTATION_TRANSITIONAL ? "transitional" : "loom")
        << "\",\"library\":\"" << escape(library) << "\",\"roots\":[";
    for (size_t i = 0; i < roots.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '\"' << escape(roots[i]) << '\"';
    }
    out << "],\"source_configuration\":{";
    size_t configuration_ordinal = 0;
    for (const auto & item : source_configuration) {
        if (configuration_ordinal++ != 0) {
            out << ',';
        }
        out << '\"' << escape(item.first) << "\":" << item.second;
    }
    out << "},\"target_specialization\":{";
    size_t target_ordinal = 0;
    for (const auto & item : target_specialization) {
        if (target_ordinal++ != 0) {
            out << ',';
        }
        out << '\"' << escape(item.first) << "\":\"" << escape(item.second) << '\"';
    }
    out << "},\"fixed_parameters\":[";
    for (size_t i = 0; i < fixed_parameters.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << "{\"role\":\"" << escape(fixed_parameters[i].role) << "\",\"storages\":[";
        for (size_t j = 0; j < fixed_parameters[i].storages.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << fixed_parameters[i].storages[j];
        }
        out << "]}";
    }
    out << "],\"runtime_resources\":[";
    for (size_t i = 0; i < runtime_resources.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        const runtime_resource & resource = runtime_resources[i];
        out << "{\"role\":\"" << escape(resource.role) << "\",\"storage\":" << resource.storage
            << ",\"value\":" << resource.value << '}';
    }
    out << "],\"coverage\":{\"regions\":[";
    for (size_t i = 0; i < coverage.regions.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << coverage.regions[i];
    }
    out << "],\"operations\":[";
    for (size_t i = 0; i < coverage.operations.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << coverage.operations[i];
    }
    out << "]},\"missing\":[";
    for (size_t i = 0; i < missing.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << "{\"region\":" << missing[i].region << ",\"operation\":" << missing[i].operation << ",\"kind\":\""
            << ggml_op_name(missing[i].op) << "\",\"reason\":\"" << escape(missing[i].reason) << "\"}";
    }
    out << "],\"errors\":[";
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '\"' << escape(errors[i]) << '\"';
    }
    out << "]}";
    return out.str();
}

graph_execution_frame graph_plan_cache::prepare(const ggml_cgraph * graph, const std::string & target) {
    graph_execution_frame frame;
    if (graph == nullptr) {
        frame.errors.push_back("cannot prepare a null cgraph");
        return frame;
    }
    frame.uid = graph->uid;
    if (frame.uid != 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = entries_.find(frame.uid);
        if (found != entries_.end()) {
            if (found->second.target != target) {
                frame.errors.push_back("graph UID was reused for a different target");
                ++stats_.failures;
                return frame;
            }
            entry & cached  = found->second;
            frame.plan      = cached.plan;
            frame.selection = cached.selection;
            if (cached.graph == graph) {
                frame.values        = cached.values;
                frame.storage_roots = cached.storage_roots;
            } else if (bind_source_graph(graph, *cached.plan, frame.values, frame.storage_roots, frame.errors)) {
                cached.graph         = graph;
                cached.values        = frame.values;
                cached.storage_roots = frame.storage_roots;
            }
            if (frame.errors.empty()) {
                ++stats_.hits;
            } else {
                ++stats_.failures;
            }
            return frame;
        }
    }

    graph_import     imported = graph_import::import(graph);
    pattern_registry patterns;
    llm_patterns::register_patterns(patterns);
    if (imported.plan.valid()) {
        matcher::recognize(imported.plan, patterns);
    }
    if (imported.plan.valid()) {
        planner::select_recipes(imported.plan, target);
    }
    auto plan      = std::make_shared<graph_plan>(std::move(imported.plan));
    auto selection = std::make_shared<program_selection>(program_selection::select(*plan, target));
    if (!plan->valid() || !selection->valid()) {
        frame.errors = plan->diagnostics().errors;
        frame.errors.insert(frame.errors.end(), selection->errors.begin(), selection->errors.end());
        for (const missing_program_step & step : selection->missing) {
            frame.errors.push_back("missing program step for op " + std::to_string(step.operation));
        }
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.failures;
        return frame;
    }
    frame.plan          = plan;
    frame.selection     = selection;
    frame.values        = std::move(imported.value_tensors);
    frame.storage_roots = std::move(imported.storage_roots);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.builds;
        if (frame.uid != 0) {
            auto [position, inserted] =
                entries_.emplace(frame.uid, entry{ target, plan, selection, graph, frame.values, frame.storage_roots });
            if (!inserted) {
                if (position->second.target != target) {
                    frame.errors.push_back("concurrent graph UID build disagreed on target");
                    ++stats_.failures;
                } else {
                    frame.plan      = position->second.plan;
                    frame.selection = position->second.selection;
                    if (position->second.graph == graph) {
                        frame.values        = position->second.values;
                        frame.storage_roots = position->second.storage_roots;
                    } else if (bind_source_graph(graph, *position->second.plan, frame.values, frame.storage_roots,
                                                 frame.errors)) {
                        position->second.graph         = graph;
                        position->second.values        = frame.values;
                        position->second.storage_roots = frame.storage_roots;
                    }
                    if (frame.errors.empty()) {
                        ++stats_.hits;
                    } else {
                        ++stats_.failures;
                    }
                }
            }
        }
    }
    return frame;
}

graph_plan_cache_stats graph_plan_cache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace ggml::hrx
