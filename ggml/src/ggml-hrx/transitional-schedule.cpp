#include "transitional-schedule.h"

#include <algorithm>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ggml::hrx {
namespace {

static void error(VerificationResult & result, const std::string & message) {
    result.errors.push_back(message);
}

static std::string escape_json(const std::string & value) {
    std::ostringstream stream;
    for (unsigned char ch : value) {
        if (ch == '\\' || ch == '\"') {
            stream << '\\' << ch;
        } else if (ch == '\n') {
            stream << "\\n";
        } else if (ch < 0x20) {
            stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
        } else {
            stream << ch;
        }
    }
    return stream.str();
}

static KernelSpecialization::ExecutionKind parse_execution_kind(const std::string & value) {
    if (value == "native") {
        return KernelSpecialization::ExecutionKind::Native;
    }
    if (value == "native_gap") {
        return KernelSpecialization::ExecutionKind::NativeGap;
    }
    if (value == "native_eager") {
        return KernelSpecialization::ExecutionKind::NativeEager;
    }
    if (value == "cpu_fallback") {
        return KernelSpecialization::ExecutionKind::CpuFallback;
    }
    throw std::runtime_error("unknown schedule execution kind");
}

static RootDisposition parse_root_disposition(const std::string & value) {
    if (value == "materialized") {
        return RootDisposition::Materialized;
    }
    if (value == "owned_endpoint_replacement") {
        return RootDisposition::OwnedEndpointReplacement;
    }
    if (value == "unresolved") {
        return RootDisposition::Unresolved;
    }
    throw std::runtime_error("unknown root disposition");
}

}  // namespace

VerificationResult verify_schedule(const Graph & graph, const Schedule & schedule) {
    VerificationResult result;
    if (!graph.valid()) {
        error(result, "cannot verify a schedule for an invalid graph");
        return result;
    }
    if (schedule.graph_fingerprint != graph.fingerprint) {
        error(result, "schedule graph fingerprint does not match");
    }
    if (schedule.expected_dispatch_count != 0 &&
        schedule_dispatch_count(schedule) != schedule.expected_dispatch_count) {
        error(result, "schedule dispatch count does not match its oracle");
    }
    std::set<ValueId> declared_roots;
    for (const RootContract & root : schedule.roots) {
        if (root.value >= graph.values.size() ||
            std::find(graph.roots.begin(), graph.roots.end(), root.value) == graph.roots.end()) {
            error(result, "schedule declares an invalid graph root");
        } else if (!declared_roots.insert(root.value).second) {
            error(result, "schedule declares a graph root more than once");
        }
        if (root.disposition == RootDisposition::OwnedEndpointReplacement && root.replacement.empty()) {
            error(result, "owned endpoint replacement has no replacement contract");
        }
    }

    std::vector<size_t> owner(graph.operations.size(), SIZE_MAX);
    size_t              dispatch_ordinal = 0;
    for (size_t invocation_id = 0; invocation_id < schedule.invocations.size(); ++invocation_id) {
        const Invocation & invocation = schedule.invocations[invocation_id];
        if (invocation.kernel.family.empty() || invocation.kernel.variant.empty()) {
            error(result, "invocation " + std::to_string(invocation_id) + " has no concrete kernel specialization");
        }
        if (invocation.covered_operations.empty()) {
            error(result, "invocation " + std::to_string(invocation_id) + " covers no operations");
        }
        for (OperationId operation : invocation.covered_operations) {
            if (operation >= graph.operations.size()) {
                error(result, "invocation " + std::to_string(invocation_id) + " references an invalid operation");
            } else if (owner[operation] != SIZE_MAX) {
                error(result, "operation " + std::to_string(operation) + " is covered more than once");
            } else {
                owner[operation] = invocation_id;
            }
        }
        auto check_bindings = [&](const std::vector<TensorBinding> & bindings, const char * kind) {
            for (const TensorBinding & binding : bindings) {
                if (binding.role.empty() || binding.value >= graph.values.size()) {
                    error(result,
                          "invocation " + std::to_string(invocation_id) + " has an invalid " + kind + " binding");
                }
            }
        };
        check_bindings(invocation.inputs, "input");
        check_bindings(invocation.outputs, "output");
        if (invocation.dispatches.empty()) {
            error(result, "invocation " + std::to_string(invocation_id) + " has no concrete dispatches");
        }
        for (const Dispatch & dispatch : invocation.dispatches) {
            if (dispatch.kernel.family.empty() || dispatch.kernel.variant.empty()) {
                error(result, "dispatch " + std::to_string(dispatch_ordinal) + " has no kernel specialization");
            }
            for (const TensorBinding & binding : dispatch.bindings) {
                if (binding.role.empty() || binding.value >= graph.values.size()) {
                    error(result, "dispatch " + std::to_string(dispatch_ordinal) + " has an invalid tensor binding");
                } else {
                    const Value &   value   = graph.values[binding.value];
                    const Storage & storage = graph.storages[value.access.storage];
                    const size_t    available =
                        storage.size > value.access.offset ? storage.size - value.access.offset : 0;
                    if (binding.offset > available ||
                        (binding.length != 0 && binding.length > available - binding.offset)) {
                        error(result,
                              "dispatch " + std::to_string(dispatch_ordinal) + " tensor binding range escapes storage");
                    }
                }
            }
            for (uint32_t dependency : dispatch.dependencies) {
                if (dependency >= dispatch_ordinal) {
                    error(result,
                          "dispatch " + std::to_string(dispatch_ordinal) + " has a forward or invalid dependency");
                }
            }
            ++dispatch_ordinal;
        }
    }

    for (OperationId operation = 0; operation < graph.operations.size(); ++operation) {
        if (owner[operation] == SIZE_MAX) {
            error(result, "operation " + std::to_string(operation) + " is not covered");
        }
    }
    for (const Operation & operation : graph.operations) {
        if (owner[operation.id] == SIZE_MAX) {
            continue;
        }
        for (ValueId input : operation.inputs) {
            if (input >= graph.values.size()) {
                continue;
            }
            OperationId producer = graph.values[input].producer;
            if (producer != kInvalidId && producer < owner.size() && owner[producer] > owner[operation.id]) {
                error(result, "schedule reverses dependency into operation " + std::to_string(operation.id));
            }
        }
        for (const Effect & effect : operation.effects) {
            if (effect.kind == EffectKind::Write && effect.after_version != effect.before_version + 1) {
                error(result, "operation " + std::to_string(operation.id) + " has an invalid write version transition");
            }
        }
    }

    std::map<std::pair<StorageId, uint32_t>, OperationId> version_writers;
    for (const Operation & operation : graph.operations) {
        for (const Effect & effect : operation.effects) {
            if (effect.kind == EffectKind::Write) {
                version_writers[{ effect.storage, effect.after_version }] = operation.id;
            }
        }
    }
    for (const Operation & operation : graph.operations) {
        if (owner[operation.id] == SIZE_MAX) {
            continue;
        }
        for (const Effect & effect : operation.effects) {
            if (effect.kind != EffectKind::Read || effect.before_version == 0) {
                continue;
            }
            auto writer = version_writers.find({ effect.storage, effect.before_version });
            if (writer == version_writers.end()) {
                error(result, "operation " + std::to_string(operation.id) + " reads a storage version with no writer");
            } else if (owner[writer->second] > owner[operation.id]) {
                error(result, "schedule reverses a storage dependency into operation " + std::to_string(operation.id));
            }
        }
    }

    std::vector<std::vector<OperationId>> consumers(graph.values.size());
    for (const Operation & operation : graph.operations) {
        for (ValueId input : operation.inputs) {
            if (input < consumers.size()) {
                consumers[input].push_back(operation.id);
            }
        }
    }
    std::set<ValueId> roots(graph.roots.begin(), graph.roots.end());
    for (size_t invocation_id = 0; invocation_id < schedule.invocations.size(); ++invocation_id) {
        const Invocation &    invocation = schedule.invocations[invocation_id];
        std::set<OperationId> covered(invocation.covered_operations.begin(), invocation.covered_operations.end());
        std::set<ValueId>     required_inputs;
        std::set<ValueId>     required_outputs;
        for (OperationId operation_id : invocation.covered_operations) {
            if (operation_id >= graph.operations.size()) {
                continue;
            }
            const Operation & operation = graph.operations[operation_id];
            for (ValueId input : operation.inputs) {
                if (input >= graph.values.size()) {
                    continue;
                }
                OperationId producer = graph.values[input].producer;
                if (producer == kInvalidId || covered.count(producer) == 0) {
                    required_inputs.insert(input);
                }
            }
            bool escapes = roots.count(operation.output) != 0;
            for (OperationId consumer : consumers[operation.output]) {
                escapes |= covered.count(consumer) == 0;
            }
            for (const Effect & effect : operation.effects) {
                escapes |= effect.kind == EffectKind::Write;
            }
            if (escapes) {
                required_outputs.insert(operation.output);
            }
        }
        auto binding_values = [&](const std::vector<TensorBinding> & bindings, const char * kind) {
            std::set<ValueId>     values;
            std::set<std::string> roles;
            for (const TensorBinding & binding : bindings) {
                if (!values.insert(binding.value).second) {
                    error(result,
                          "invocation " + std::to_string(invocation_id) + " binds a " + kind + " value more than once");
                }
                if (!roles.insert(binding.role).second) {
                    error(result, "invocation " + std::to_string(invocation_id) + " repeats a " + kind + " role");
                }
            }
            return values;
        };
        if (binding_values(invocation.inputs, "input") != required_inputs) {
            error(result,
                  "invocation " + std::to_string(invocation_id) + " input bindings do not match its region boundary");
        }
        if (binding_values(invocation.outputs, "output") != required_outputs) {
            error(result,
                  "invocation " + std::to_string(invocation_id) + " output bindings do not match its region boundary");
        }
    }
    return result;
}

Schedule deserialize_schedule_json(const std::string & text, std::vector<std::string> & errors) {
    Schedule schedule;
    try {
        const nlohmann::json root    = nlohmann::json::parse(text);
        const int            version = root.at("version").get<int>();
        if (version < 1 || version > 4) {
            errors.emplace_back("unsupported schedule manifest version");
            return schedule;
        }
        schedule.graph_fingerprint = root.at("graph_fingerprint").get<std::string>();
        if (version >= 2) {
            schedule.workload                = root.value("workload", "");
            schedule.expected_dispatch_count = root.value("expected_dispatch_count", 0);
        }
        if (version >= 3) {
            schedule.oracle_revision = root.value("oracle_revision", "");
            for (const nlohmann::json & root_item : root.value("roots", nlohmann::json::array())) {
                schedule.roots.push_back({ root_item.at("value").get<ValueId>(),
                                           parse_root_disposition(root_item.at("disposition").get<std::string>()),
                                           root_item.value("replacement", "") });
            }
        }
        for (const nlohmann::json & item : root.at("invocations")) {
            Invocation             invocation;
            const nlohmann::json & kernel        = item.at("kernel");
            invocation.kernel.execution_kind     = parse_execution_kind(kernel.at("execution").get<std::string>());
            invocation.kernel.family             = kernel.at("family").get<std::string>();
            invocation.kernel.variant            = kernel.at("variant").get<std::string>();
            invocation.kernel.kernel_id          = kernel.value("id", uint64_t{ 0 });
            invocation.kernel.integer_parameters = kernel.at("parameters").get<std::map<std::string, int64_t>>();
            invocation.covered_operations        = item.at("operations").get<std::vector<OperationId>>();
            if (version >= 3) {
                invocation.stage = item.value("stage", "");
                invocation.layer = item.value("layer", -1);
            }
            if (version >= 4) {
                invocation.recipe             = item.value("recipe", "");
                invocation.logical_components = item.value("logical_components", std::vector<uint32_t>());
            }
            auto read_bindings = [](const nlohmann::json & bindings) {
                std::vector<TensorBinding> result;
                for (const nlohmann::json & binding : bindings) {
                    result.push_back({ binding.at("role").get<std::string>(), binding.at("value").get<ValueId>(),
                                       binding.value("offset", size_t{ 0 }), binding.value("length", size_t{ 0 }) });
                }
                return result;
            };
            invocation.inputs  = read_bindings(item.at("inputs"));
            invocation.outputs = read_bindings(item.at("outputs"));
            if (version >= 2) {
                for (const nlohmann::json & dispatch_item : item.at("dispatches")) {
                    Dispatch               dispatch;
                    const nlohmann::json & dispatch_kernel = dispatch_item.at("kernel");
                    dispatch.kernel.execution_kind =
                        parse_execution_kind(dispatch_kernel.at("execution").get<std::string>());
                    dispatch.kernel.family    = dispatch_kernel.at("family").get<std::string>();
                    dispatch.kernel.variant   = dispatch_kernel.at("variant").get<std::string>();
                    dispatch.kernel.kernel_id = dispatch_kernel.value("id", uint64_t{ 0 });
                    dispatch.kernel.integer_parameters =
                        dispatch_kernel.at("parameters").get<std::map<std::string, int64_t>>();
                    dispatch.kernel.compile_parameters =
                        dispatch_kernel.value("compile_parameters", std::map<std::string, std::string>());
                    dispatch.bindings     = read_bindings(dispatch_item.at("bindings"));
                    dispatch.dependencies = dispatch_item.at("dependencies").get<std::vector<uint32_t>>();
                    invocation.dispatches.push_back(std::move(dispatch));
                }
            } else {
                Dispatch dispatch;
                dispatch.kernel = invocation.kernel;
                dispatch.bindings.insert(dispatch.bindings.end(), invocation.inputs.begin(), invocation.inputs.end());
                dispatch.bindings.insert(dispatch.bindings.end(), invocation.outputs.begin(), invocation.outputs.end());
                invocation.dispatches.push_back(std::move(dispatch));
            }
            schedule.invocations.push_back(std::move(invocation));
        }
    } catch (const std::exception & error) {
        errors.emplace_back(std::string("invalid schedule manifest: ") + error.what());
    }
    return schedule;
}

std::string serialize_schedule_json(const Schedule & schedule) {
    std::ostringstream out;
    out << "{\"version\":4,\"graph_fingerprint\":\"" << escape_json(schedule.graph_fingerprint) << "\",\"workload\":\""
        << escape_json(schedule.workload) << "\",\"oracle_revision\":\"" << escape_json(schedule.oracle_revision)
        << "\",\"expected_dispatch_count\":" << schedule.expected_dispatch_count << ",\"roots\":[";
    for (size_t i = 0; i < schedule.roots.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << "{\"value\":" << schedule.roots[i].value << ",\"disposition\":\""
            << root_disposition_name(schedule.roots[i].disposition) << "\",\"replacement\":\""
            << escape_json(schedule.roots[i].replacement) << "\"}";
    }
    out << "],\"invocations\":[";
    for (size_t i = 0; i < schedule.invocations.size(); ++i) {
        const Invocation & invocation = schedule.invocations[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"stage\":\"" << escape_json(invocation.stage) << "\",\"layer\":" << invocation.layer
            << ",\"recipe\":\"" << escape_json(invocation.recipe) << "\",\"logical_components\":[";
        for (size_t j = 0; j < invocation.logical_components.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << invocation.logical_components[j];
        }
        out << "],\"kernel\":{\"execution\":\"" << execution_kind_name(invocation.kernel.execution_kind)
            << "\",\"family\":\"" << escape_json(invocation.kernel.family) << "\",\"variant\":\""
            << escape_json(invocation.kernel.variant) << "\",\"id\":" << invocation.kernel.kernel_id
            << ",\"parameters\":{";
        size_t parameter_index = 0;
        for (const auto & parameter : invocation.kernel.integer_parameters) {
            if (parameter_index++ != 0) {
                out << ',';
            }
            out << '\"' << escape_json(parameter.first) << "\":" << parameter.second;
        }
        out << "}},\"operations\":[";
        for (size_t j = 0; j < invocation.covered_operations.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << invocation.covered_operations[j];
        }
        auto write_bindings = [&](const char * name, const std::vector<TensorBinding> & bindings) {
            out << "],\"" << name << "\":[";
            for (size_t j = 0; j < bindings.size(); ++j) {
                if (j != 0) {
                    out << ',';
                }
                out << "{\"role\":\"" << escape_json(bindings[j].role) << "\",\"value\":" << bindings[j].value << '}';
            }
        };
        write_bindings("inputs", invocation.inputs);
        write_bindings("outputs", invocation.outputs);
        out << "],\"dispatches\":[";
        for (size_t j = 0; j < invocation.dispatches.size(); ++j) {
            const Dispatch & dispatch = invocation.dispatches[j];
            if (j != 0) {
                out << ',';
            }
            out << "{\"kernel\":{\"execution\":\"" << execution_kind_name(dispatch.kernel.execution_kind)
                << "\",\"family\":\"" << escape_json(dispatch.kernel.family) << "\",\"variant\":\""
                << escape_json(dispatch.kernel.variant) << "\",\"id\":" << dispatch.kernel.kernel_id
                << ",\"parameters\":{";
            size_t dispatch_parameter_index = 0;
            for (const auto & parameter : dispatch.kernel.integer_parameters) {
                if (dispatch_parameter_index++ != 0) {
                    out << ',';
                }
                out << '\"' << escape_json(parameter.first) << "\":" << parameter.second;
            }
            out << "},\"compile_parameters\":{";
            size_t compile_parameter_index = 0;
            for (const auto & parameter : dispatch.kernel.compile_parameters) {
                if (compile_parameter_index++ != 0) {
                    out << ',';
                }
                out << '\"' << escape_json(parameter.first) << "\":\"" << escape_json(parameter.second) << '\"';
            }
            out << "}},\"bindings\":[";
            for (size_t k = 0; k < dispatch.bindings.size(); ++k) {
                if (k != 0) {
                    out << ',';
                }
                out << "{\"role\":\"" << escape_json(dispatch.bindings[k].role)
                    << "\",\"value\":" << dispatch.bindings[k].value << ",\"offset\":" << dispatch.bindings[k].offset
                    << ",\"length\":" << dispatch.bindings[k].length << '}';
            }
            out << "],\"dependencies\":[";
            for (size_t k = 0; k < dispatch.dependencies.size(); ++k) {
                if (k != 0) {
                    out << ',';
                }
                out << dispatch.dependencies[k];
            }
            out << "]}";
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

size_t schedule_dispatch_count(const Schedule & schedule) {
    size_t result = 0;
    for (const Invocation & invocation : schedule.invocations) {
        result += invocation.dispatches.size();
    }
    return result;
}

size_t schedule_execution_kind_count(const Schedule & schedule, KernelSpecialization::ExecutionKind kind) {
    size_t result = 0;
    for (const Invocation & invocation : schedule.invocations) {
        for (const Dispatch & dispatch : invocation.dispatches) {
            result += dispatch.kernel.execution_kind == kind;
        }
    }
    return result;
}

std::string kernel_specialization_name(const KernelSpecialization & kernel) {
    return kernel.family + ":" + kernel.variant;
}

const char * execution_kind_name(KernelSpecialization::ExecutionKind kind) {
    switch (kind) {
        case KernelSpecialization::ExecutionKind::Native:
            return "native";
        case KernelSpecialization::ExecutionKind::NativeGap:
            return "native_gap";
        case KernelSpecialization::ExecutionKind::NativeEager:
            return "native_eager";
        case KernelSpecialization::ExecutionKind::CpuFallback:
            return "cpu_fallback";
    }
    return "unknown";
}

const char * root_disposition_name(RootDisposition disposition) {
    switch (disposition) {
        case RootDisposition::Materialized:
            return "materialized";
        case RootDisposition::OwnedEndpointReplacement:
            return "owned_endpoint_replacement";
        case RootDisposition::Unresolved:
            return "unresolved";
    }
    return "unknown";
}

}  // namespace ggml::hrx
