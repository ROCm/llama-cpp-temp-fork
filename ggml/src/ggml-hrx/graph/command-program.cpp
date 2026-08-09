#include "command-program.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

static size_t align_up(size_t value, size_t alignment) {
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

static size_t value_span(const Value & value) {
    const size_t blocks = (static_cast<size_t>(value.access.shape[0]) + ggml_blck_size(value.type) - 1) /
        ggml_blck_size(value.type);
    if (blocks == 0) return 0;
    size_t result = ggml_type_size(value.type) + (blocks - 1) * value.access.strides[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (value.access.shape[i] > 0) result += static_cast<size_t>(value.access.shape[i] - 1) * value.access.strides[i];
    }
    return result;
}

static ValueId find_named_value(const Graph & graph, const std::string & name) {
    const auto position = std::find_if(graph.values.begin(), graph.values.end(), [&](const Value & value) {
        return value.name == name;
    });
    return position == graph.values.end() ? kInvalidId : position->id;
}

static void append_rope_initialization(const ProgramPlan & plan, CommandProgram & program) {
    const ValueId frequencies = find_named_value(plan.graph, "hrx.synthetic.inverse_frequencies");
    if (frequencies == kInvalidId) return;
    const Operation * reference = nullptr;
    for (const Operation & operation : plan.graph.operations) {
        if (operation.op != GGML_OP_ROPE) continue;
        if (operation.inputs.size() != 2) {
            program.errors.push_back("selected attention kernel requires ROPE without a frequency-factor tensor");
            return;
        }
        if (reference == nullptr) reference = &operation;
        else if (operation.raw_params != reference->raw_params) {
            program.errors.push_back("ROPE parameters are not coherent across the owned program");
            return;
        }
    }
    if (reference == nullptr) {
        program.errors.push_back("owned program has no ROPE witness for inverse-frequency initialization");
        return;
    }
    int32_t params[15] = {};
    std::memcpy(params, reference->raw_params.data(), sizeof(params));
    float frequency_base = 0.0f;
    float frequency_scale = 0.0f;
    float extension_factor = 0.0f;
    float attention_factor = 0.0f;
    std::memcpy(&frequency_base, params + 5, sizeof(float));
    std::memcpy(&frequency_scale, params + 6, sizeof(float));
    std::memcpy(&extension_factor, params + 7, sizeof(float));
    std::memcpy(&attention_factor, params + 8, sizeof(float));
    const size_t n_dims = static_cast<size_t>(params[1]);
    if (params[2] != GGML_ROPE_TYPE_NEOX || n_dims == 0 || n_dims % 2 != 0 ||
        !std::isfinite(frequency_base) || frequency_base <= 0.0f ||
        !std::isfinite(frequency_scale) || frequency_scale <= 0.0f ||
        extension_factor != 0.0f || attention_factor != 1.0f) {
        program.errors.push_back("ROPE parameters are outside the selected NEOX inverse-frequency contract");
        return;
    }
    const Value & destination = plan.graph.values[frequencies];
    if (plan.graph.storages[destination.access.storage].size != n_dims / 2 * sizeof(float)) {
        program.errors.push_back("inverse-frequency storage does not agree with recovered ROPE dimensions");
        return;
    }
    ConstantInitialization initialization;
    initialization.label = "rope.inverse_frequencies";
    initialization.storage = destination.access.storage;
    initialization.data.resize(n_dims / 2 * sizeof(float));
    const float theta_scale = std::pow(frequency_base, -2.0f / static_cast<float>(n_dims));
    float theta = frequency_scale;
    for (size_t i = 0; i < n_dims / 2; ++i) {
        std::memcpy(initialization.data.data() + i * sizeof(float), &theta, sizeof(float));
        theta *= theta_scale;
    }
    program.initializations.push_back(std::move(initialization));
}

static std::vector<Command> expand_synthetic_commands(const std::vector<Command> & kernels,
                                                       std::vector<std::string> & errors) {
    std::vector<Command> result;
    std::vector<uint32_t> remap(kernels.size(), UINT32_MAX);
    auto mapped_dependencies = [&](const Command & command) {
        std::vector<uint32_t> dependencies;
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= remap.size() || remap[dependency] == UINT32_MAX) {
                errors.push_back("cannot remap a forward synthetic command dependency");
            } else {
                dependencies.push_back(remap[dependency]);
            }
        }
        return dependencies;
    };
    for (const Command & original : kernels) {
        std::vector<uint32_t> dependencies = mapped_dependencies(original);
        uint32_t synthetic = UINT32_MAX;
        if (original.kernel.variant == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma" &&
                   original.bindings.size() >= 8) {
            Command fill;
            fill.ordinal = static_cast<uint32_t>(result.size());
            fill.kind = CommandKind::Fill;
            fill.label = original.label + ".clear_completion_counter";
            fill.kernel.integer_parameters["fill_byte"] = 0;
            fill.dependencies = dependencies;
            CommandBinding destination = original.bindings[7];
            destination.name = "destination";
            destination.access = ResourceAccess::Write;
            fill.bindings = { destination };
            result.push_back(std::move(fill));
            synthetic = result.back().ordinal;
        } else if (original.kernel.variant == "qwen3_moe_build_expert_table_partition_prefill_512" &&
                   original.bindings.size() >= 4) {
            Command fill;
            fill.ordinal = static_cast<uint32_t>(result.size());
            fill.kind = CommandKind::Fill;
            fill.label = original.label + ".clear_completion_counter";
            fill.kernel.integer_parameters["fill_byte"] = 0;
            fill.dependencies = dependencies;
            CommandBinding destination = original.bindings[3];
            destination.name = "destination";
            destination.access = ResourceAccess::Write;
            fill.bindings = { destination };
            result.push_back(std::move(fill));
            synthetic = result.back().ordinal;
        }
        Command command = original;
        command.ordinal = static_cast<uint32_t>(result.size());
        command.dependencies = std::move(dependencies);
        if (synthetic != UINT32_MAX) command.dependencies.push_back(synthetic);
        result.push_back(std::move(command));
        remap[original.ordinal] = result.back().ordinal;
    }
    return result;
}

static std::string stable_hash(const std::string & text) {
    // FNV-1a is used only as an in-process cache witness.
    uint64_t hash = UINT64_C(1469598103934665603);
    for (unsigned char byte : text) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

struct TransientExecutionOrder {
    std::vector<std::vector<uint64_t>> uses;
    std::vector<std::vector<uint64_t>> common_ancestors;
    std::vector<bool> used;

    bool ordered_before(uint32_t before, uint32_t after) const {
        if (before >= uses.size() || after >= uses.size() || !used[before] || !used[after]) return false;
        for (size_t word = 0; word < uses[before].size(); ++word) {
            if ((uses[before][word] & ~common_ancestors[after][word]) != 0) return false;
        }
        return true;
    }

    bool lifetimes_overlap(uint32_t a, uint32_t b) const {
        if (a >= uses.size() || b >= uses.size() || !used[a] || !used[b]) return false;
        return !ordered_before(a, b) && !ordered_before(b, a);
    }
};

static TransientExecutionOrder analyze_transient_execution(
        const std::vector<Command> & commands, size_t transient_count) {
    const size_t word_count = (commands.size() + 63) / 64;
    std::vector<std::vector<uint64_t>> command_ancestors(
        commands.size(), std::vector<uint64_t>(word_count));
    for (size_t command_index = 0; command_index < commands.size(); ++command_index) {
        for (uint32_t dependency : commands[command_index].dependencies) {
            if (dependency >= command_index) continue;
            command_ancestors[command_index][dependency / 64] |= UINT64_C(1) << (dependency % 64);
            for (size_t word = 0; word < word_count; ++word) {
                command_ancestors[command_index][word] |= command_ancestors[dependency][word];
            }
        }
    }

    TransientExecutionOrder result;
    result.uses.assign(transient_count, std::vector<uint64_t>(word_count));
    result.common_ancestors.assign(transient_count, std::vector<uint64_t>(word_count, UINT64_MAX));
    result.used.assign(transient_count, false);
    for (size_t command_index = 0; command_index < commands.size(); ++command_index) {
        for (const CommandBinding & binding : commands[command_index].bindings) {
            if (binding.origin != BindingOrigin::Transient || binding.transient >= transient_count) continue;
            result.uses[binding.transient][command_index / 64] |= UINT64_C(1) << (command_index % 64);
        }
    }
    for (size_t transient = 0; transient < transient_count; ++transient) {
        for (size_t command_index = 0; command_index < commands.size(); ++command_index) {
            if ((result.uses[transient][command_index / 64] &
                 (UINT64_C(1) << (command_index % 64))) == 0) continue;
            result.used[transient] = true;
            for (size_t word = 0; word < word_count; ++word) {
                result.common_ancestors[transient][word] &= command_ancestors[command_index][word];
            }
        }
    }
    return result;
}

static void pack_transient_plan(
        TransientPlan & result, const TransientExecutionOrder & execution_order) {
    result.arena_size = 0;
    for (TransientAllocation & allocation : result.allocations) allocation.arena_offset = 0;
    std::vector<size_t> order(result.allocations.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const TransientAllocation & a = result.allocations[lhs];
        const TransientAllocation & b = result.allocations[rhs];
        if (a.first_command != b.first_command) return a.first_command < b.first_command;
        if (a.size != b.size) return a.size > b.size;
        return a.storage < b.storage;
    });
    std::vector<bool> is_placed(result.allocations.size(), false);
    for (size_t index : order) {
        TransientAllocation & allocation = result.allocations[index];
        size_t candidate = 0;
        for (;;) {
            candidate = align_up(candidate, allocation.alignment);
            bool conflict = false;
            size_t next_candidate = candidate;
            for (size_t placed_index = 0; placed_index < result.allocations.size(); ++placed_index) {
                if (!is_placed[placed_index]) continue;
                const TransientAllocation & placed = result.allocations[placed_index];
                const bool range_overlap = candidate < placed.arena_offset + placed.size &&
                    placed.arena_offset < candidate + allocation.size;
                if (execution_order.lifetimes_overlap(allocation.id, placed.id) && range_overlap) {
                    conflict = true;
                    next_candidate = std::max(next_candidate, placed.arena_offset + placed.size);
                }
            }
            if (!conflict) break;
            candidate = next_candidate;
        }
        allocation.arena_offset = candidate;
        is_placed[index] = true;
        result.arena_size = std::max(result.arena_size, candidate + allocation.size);
    }
    result.arena_size = align_up(result.arena_size, result.arena_alignment);
    std::sort(result.allocations.begin(), result.allocations.end(),
              [](const TransientAllocation & a, const TransientAllocation & b) { return a.id < b.id; });
}

static PersistentConstantPlan build_persistent_constant_plan(
        const ProgramPlan & plan, const std::vector<ConstantInitialization> & initializations) {
    PersistentConstantPlan result;
    result.arena_alignment = 256;
    std::set<StorageId> storages;
    for (const ConstantInitialization & initialization : initializations) {
        if (!storages.insert(initialization.storage).second || initialization.storage >= plan.graph.storages.size()) {
            continue;
        }
        PersistentConstantAllocation allocation;
        allocation.id = static_cast<uint32_t>(result.allocations.size());
        allocation.storage = initialization.storage;
        allocation.size = plan.graph.storages[initialization.storage].size;
        allocation.alignment = 256;
        allocation.arena_offset = align_up(result.arena_size, allocation.alignment);
        result.arena_size = allocation.arena_offset + allocation.size;
        result.allocations.push_back(allocation);
    }
    result.arena_size = align_up(result.arena_size, result.arena_alignment);
    return result;
}

static TransientPlan build_transient_plan(
        const ProgramPlan & plan, const PersistentConstantPlan & persistent_constants) {
    TransientPlan result;
    result.arena_alignment = 256;
    for (const ResourceContract & resource : plan.resources.resources) {
        if (!resource.elidable || resource.first_invocation == UINT32_MAX) continue;
        const bool persistent = std::any_of(
            persistent_constants.allocations.begin(), persistent_constants.allocations.end(),
            [&](const PersistentConstantAllocation & allocation) { return allocation.storage == resource.storage; });
        if (persistent) continue;
        TransientAllocation allocation;
        allocation.id = static_cast<uint32_t>(result.allocations.size());
        allocation.storage = resource.storage;
        allocation.size = resource.size;
        allocation.alignment = 256;
        result.allocations.push_back(allocation);
    }
    return result;
}

static nlohmann::ordered_json binding_json(const CommandBinding & binding) {
    const char * origin = "graph_value";
    if (binding.origin == BindingOrigin::Transient) origin = "transient";
    if (binding.origin == BindingOrigin::PersistentConstant) origin = "persistent_constant";
    return {
        { "name", binding.name },
        { "origin", origin },
        { "value", binding.value },
        { "transient", binding.transient },
        { "persistent_constant", binding.persistent_constant },
        { "storage", binding.storage },
        { "offset", binding.offset },
        { "length", binding.length },
        { "storage_binding", binding.storage_binding },
        { "access", resource_access_name(binding.access) },
    };
}

} // namespace

const char * command_kind_name(CommandKind kind) {
    switch (kind) {
        case CommandKind::Kernel: return "kernel";
        case CommandKind::Fill: return "fill";
        case CommandKind::Copy: return "copy";
        case CommandKind::Barrier: return "barrier";
    }
    return "unknown";
}

const char * resource_access_name(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read: return "read";
        case ResourceAccess::Write: return "write";
        case ResourceAccess::ReadWrite: return "read_write";
    }
    return "unknown";
}

CommandProgram build_command_program(const ProgramPlan & plan, const kernel_corpus & corpus) {
    CommandProgram result;
    result.workload = plan.schedule.workload;
    result.target = plan.target;
    result.graph_fingerprint = plan.graph.fingerprint;
    result.recipe_revision = plan.schedule.oracle_revision;
    result.roots = plan.schedule.roots;
    const VerificationResult corpus_verification = verify_kernel_corpus(corpus);
    result.errors.insert(result.errors.end(), corpus_verification.errors.begin(), corpus_verification.errors.end());

    uint32_t ordinal = 0;
    for (const Invocation & invocation : plan.schedule.invocations) {
        for (const Dispatch & dispatch : invocation.dispatches) {
            Command command;
            command.ordinal = ordinal++;
            command.kind = CommandKind::Kernel;
            command.label = invocation.stage + (invocation.layer >= 0 ? "." + std::to_string(invocation.layer) : "");
            command.kernel = dispatch.kernel;
            command.dependencies = dispatch.dependencies;
            if (dispatch.streamed()) {
                command.streamed.layer = invocation.layer;
                command.streamed.weight = dispatch.streamed_weight;
                command.streamed.expert_ids = dispatch.streamed_expert_ids;
                command.streamed.missing_suffix = dispatch.streamed_missing_suffix;
                command.streamed.expert_begin = dispatch.streamed_expert_begin;
                command.streamed.expert_end = dispatch.streamed_expert_end;
                command.streamed.load_chunk_size = dispatch.streamed_load_chunk_size;
            }
            const kernel_resolve_result resolved = resolve_kernel_definition(corpus, plan.target, command.kernel);
            const kernel_definition * definition = resolved.definition;
            if (!resolved.found()) {
                result.errors.push_back(format_kernel_resolve_error(resolved, command.kernel));
            }
            for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
                const TensorBinding & tensor_binding = dispatch.bindings[binding_index];
                if (tensor_binding.value >= plan.graph.values.size()) continue;
                const Value & value = plan.graph.values[tensor_binding.value];
                CommandBinding binding;
                binding.name = tensor_binding.role;
                binding.value = tensor_binding.value;
                binding.storage = value.access.storage;
                binding.offset = value.access.offset + tensor_binding.offset;
                const size_t span = value_span(value);
                const size_t available = tensor_binding.offset < span ? span - tensor_binding.offset : 0;
                binding.length = tensor_binding.length == 0 ? available : tensor_binding.length;
                binding.storage_binding = tensor_binding.storage_binding;
                binding.access = definition != nullptr && binding_index < definition->bindings.size()
                    ? definition->bindings[binding_index].access : ResourceAccess::ReadWrite;
                command.bindings.push_back(std::move(binding));
            }
            result.commands.push_back(std::move(command));
        }
    }
    result.commands = expand_synthetic_commands(result.commands, result.errors);
    append_rope_initialization(plan, result);
    result.persistent_constants = build_persistent_constant_plan(plan, result.initializations);
    result.transients = build_transient_plan(plan, result.persistent_constants);
    for (Command & command : result.commands) {
        for (CommandBinding & binding : command.bindings) {
            const auto persistent = std::find_if(
                result.persistent_constants.allocations.begin(), result.persistent_constants.allocations.end(),
                [&](const PersistentConstantAllocation & allocation) { return allocation.storage == binding.storage; });
            if (persistent != result.persistent_constants.allocations.end()) {
                binding.origin = BindingOrigin::PersistentConstant;
                binding.persistent_constant = persistent->id;
                continue;
            }
            const auto position = std::find_if(result.transients.allocations.begin(), result.transients.allocations.end(),
                [&](const TransientAllocation & allocation) { return allocation.storage == binding.storage; });
            if (position != result.transients.allocations.end()) {
                binding.origin = BindingOrigin::Transient;
                binding.transient = position->id;
            }
        }
    }
    for (TransientAllocation & allocation : result.transients.allocations) {
        allocation.first_command = UINT32_MAX;
        allocation.last_command = 0;
    }
    for (const Command & command : result.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != BindingOrigin::Transient || binding.transient >= result.transients.allocations.size()) continue;
            TransientAllocation & allocation = result.transients.allocations[binding.transient];
            allocation.first_command = std::min(allocation.first_command, command.ordinal);
            allocation.last_command = std::max(allocation.last_command, command.ordinal);
        }
    }
    // Schedule dependencies describe authored ordering. Add the conservative
    // resource hazards required by concrete command recording so a future
    // recipe may expose independent commands without weakening mutation or
    // alias correctness.
    std::map<StorageId, uint32_t> last_writer;
    std::map<StorageId, std::set<uint32_t>> readers;
    for (Command & command : result.commands) {
        std::set<uint32_t> dependencies(command.dependencies.begin(), command.dependencies.end());
        for (const CommandBinding & binding : command.bindings) {
            auto writer = last_writer.find(binding.storage);
            if (writer != last_writer.end()) dependencies.insert(writer->second);
            if (binding.access != ResourceAccess::Read) {
                const auto storage_readers = readers.find(binding.storage);
                if (storage_readers != readers.end()) dependencies.insert(storage_readers->second.begin(), storage_readers->second.end());
            }
        }
        dependencies.erase(command.ordinal);
        command.dependencies.assign(dependencies.begin(), dependencies.end());
        for (const CommandBinding & binding : command.bindings) {
            if (binding.access == ResourceAccess::Read) {
                readers[binding.storage].insert(command.ordinal);
            } else {
                last_writer[binding.storage] = command.ordinal;
                readers[binding.storage].clear();
            }
        }
    }
    pack_transient_plan(
        result.transients, analyze_transient_execution(result.commands, result.transients.allocations.size()));
    return result;
}

VerificationResult verify_command_program(const ProgramPlan & plan, const kernel_corpus & corpus,
                                          const CommandProgram & commands) {
    VerificationResult result;
    if (!commands.valid()) result.errors.insert(result.errors.end(), commands.errors.begin(), commands.errors.end());
    if (commands.graph_fingerprint != plan.graph.fingerprint || commands.workload != plan.schedule.workload ||
        commands.target != plan.target) {
        result.errors.push_back("command program identity does not match its plan");
    }
    const size_t kernel_count = std::count_if(commands.commands.begin(), commands.commands.end(), [](const Command & command) {
        return command.kind == CommandKind::Kernel;
    });
    if (kernel_count != schedule_dispatch_count(plan.schedule)) {
        result.errors.push_back("command program does not cover every scheduled dispatch");
    }
    for (size_t i = 0; i < commands.commands.size(); ++i) {
        const Command & command = commands.commands[i];
        if (command.ordinal != i) result.errors.push_back("command ordinals are not contiguous");
        kernel_resolve_result resolved;
        const kernel_definition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved = resolve_kernel_definition(corpus, commands.target, command.kernel);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            result.errors.push_back("command " + std::to_string(command.ordinal) + ": " +
                                    format_kernel_resolve_error(resolved, command.kernel));
        } else if (definition != nullptr) {
            for (const char * scalar : definition->scalar_parameters) {
                const std::string scalar_name = scalar != nullptr ? scalar : "";
                if (command.kernel.integer_parameters.count(scalar_name) == 0) {
                    result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " +
                        kernel_specialization_name(command.kernel) + " omits scalar " + scalar_name);
                }
            }
            if (command.bindings.size() != definition->bindings.size()) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " +
                    kernel_specialization_name(command.kernel) +
                    " has " + std::to_string(command.bindings.size()) + " bindings but its ABI requires " +
                    std::to_string(definition->bindings.size()));
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                if (command.bindings[binding_index].name != definition->bindings[binding_index].name ||
                    command.bindings[binding_index].access != definition->bindings[binding_index].access) {
                    result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " +
                        kernel_specialization_name(command.kernel) + " binding " + std::to_string(binding_index) +
                        " does not match the kernel ABI");
                }
            }
        }
        if (command.kind == CommandKind::Copy && command.bindings.size() != 2) {
            result.errors.push_back("copy command does not have source and destination bindings");
        }
        if (command.kind == CommandKind::Fill && command.bindings.size() != 1) {
            result.errors.push_back("fill command does not have one destination binding");
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) result.errors.push_back("command has a forward dependency");
        }
        if ((command.streamed.weight != kInvalidId || command.streamed.expert_ids != kInvalidId ||
             command.streamed.missing_suffix) &&
            !command.streamed.valid()) {
            result.errors.push_back("command has an incomplete streamed-expert contract");
        }
        if (command.streamed.valid()) {
            if (command.streamed.weight >= plan.graph.values.size() ||
                command.streamed.expert_ids >= plan.graph.values.size() ||
                plan.graph.values[command.streamed.expert_ids].type != GGML_TYPE_I32) {
                result.errors.push_back("command has an invalid streamed-expert value");
            }
            const bool binds_weight_view = std::any_of(
                command.bindings.begin(), command.bindings.end(), [&](const CommandBinding & binding) {
                    return binding.value == command.streamed.weight && !binding.storage_binding.empty();
                });
            if (!binds_weight_view) {
                result.errors.push_back("streamed-expert command does not bind its physical cache view");
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            if (binding.storage >= plan.graph.storages.size() || binding.length == 0 ||
                binding.offset + binding.length > plan.graph.storages[binding.storage].size) {
                result.errors.push_back("command has an invalid resource range");
            }
            if (binding.origin == BindingOrigin::PersistentConstant) {
                const auto allocation = std::find_if(
                    commands.persistent_constants.allocations.begin(), commands.persistent_constants.allocations.end(),
                    [&](const PersistentConstantAllocation & item) {
                        return item.id == binding.persistent_constant && item.storage == binding.storage;
                    });
                if (allocation == commands.persistent_constants.allocations.end()) {
                    result.errors.push_back("command has an invalid persistent constant binding");
                }
                if (binding.access != ResourceAccess::Read) {
                    result.errors.push_back("command writes persistent constant storage");
                }
            } else if (binding.origin == BindingOrigin::Transient) {
                const auto allocation = std::find_if(
                    commands.transients.allocations.begin(), commands.transients.allocations.end(),
                    [&](const TransientAllocation & item) {
                        return item.id == binding.transient && item.storage == binding.storage;
                    });
                if (allocation == commands.transients.allocations.end()) {
                    result.errors.push_back("command has an invalid transient binding");
                }
            }
        }
    }
    std::set<StorageId> initialized_storages;
    for (const ConstantInitialization & initialization : commands.initializations) {
        if (initialization.storage >= plan.graph.storages.size() || initialization.data.empty() ||
            initialization.data.size() > plan.graph.storages[initialization.storage].size ||
            !initialized_storages.insert(initialization.storage).second) {
            result.errors.push_back("invalid constant initialization payload");
        }
        const auto allocation = std::find_if(commands.persistent_constants.allocations.begin(),
            commands.persistent_constants.allocations.end(), [&](const PersistentConstantAllocation & item) {
                return item.storage == initialization.storage;
            });
        if (allocation == commands.persistent_constants.allocations.end()) {
            result.errors.push_back("constant initialization does not resolve to persistent storage");
        }
    }
    std::set<StorageId> persistent_storages;
    std::set<uint32_t> persistent_ids;
    for (const PersistentConstantAllocation & allocation : commands.persistent_constants.allocations) {
        if (allocation.storage >= plan.graph.storages.size() ||
            !plan.resources.resources[allocation.storage].elidable ||
            allocation.size != plan.graph.storages[allocation.storage].size ||
            allocation.alignment == 0 || allocation.arena_offset % allocation.alignment != 0 ||
            allocation.arena_offset + allocation.size > commands.persistent_constants.arena_size ||
            !persistent_storages.insert(allocation.storage).second ||
            !persistent_ids.insert(allocation.id).second ||
            initialized_storages.count(allocation.storage) == 0) {
            result.errors.push_back("invalid persistent constant allocation");
        }
        for (const PersistentConstantAllocation & other : commands.persistent_constants.allocations) {
            if (allocation.id >= other.id) continue;
            const bool overlap = allocation.arena_offset < other.arena_offset + other.size &&
                other.arena_offset < allocation.arena_offset + allocation.size;
            if (overlap) result.errors.push_back("persistent constant allocations overlap");
        }
    }
    const TransientExecutionOrder execution_order = analyze_transient_execution(
        commands.commands, commands.transients.allocations.size());
    for (size_t allocation_index = 0; allocation_index < commands.transients.allocations.size(); ++allocation_index) {
        const TransientAllocation & a = commands.transients.allocations[allocation_index];
        if (a.storage >= plan.graph.storages.size() || !plan.resources.resources[a.storage].elidable ||
            a.size != plan.graph.storages[a.storage].size || a.alignment == 0 ||
            a.arena_offset % a.alignment != 0 ||
            a.arena_offset + a.size > commands.transients.arena_size || a.id != allocation_index) {
            result.errors.push_back("invalid transient allocation");
        }
        if (persistent_storages.count(a.storage) != 0) {
            result.errors.push_back("persistent constant is also allocated as a transient");
        }
        for (const TransientAllocation & b : commands.transients.allocations) {
            if (a.id >= b.id) continue;
            const bool range_overlap = a.arena_offset < b.arena_offset + b.size && b.arena_offset < a.arena_offset + a.size;
            if (range_overlap && execution_order.lifetimes_overlap(a.id, b.id)) {
                result.errors.push_back("aliased transient lifetimes overlap in the command dependency graph");
            }
        }
    }
    return result;
}

VerificationResult verify_binding_snapshot(const ProgramPlan & plan, const BindingSnapshot & snapshot) {
    VerificationResult result;
    if (snapshot.device_identity.empty()) result.errors.push_back("binding snapshot has no device identity");
    std::set<StorageId> storages;
    for (const ConcreteBinding & binding : snapshot.bindings) {
        if (binding.storage >= plan.graph.storages.size() || !storages.insert(binding.storage).second) {
            result.errors.push_back("binding snapshot has invalid or duplicate storage");
            continue;
        }
        if (binding.buffer_identity == 0 || binding.generation == 0 || binding.length == 0 ||
            binding.offset + binding.length > binding.capacity || binding.length < plan.graph.storages[binding.storage].size) {
            result.errors.push_back("binding snapshot range or identity is invalid");
        }
    }
    for (const ResourceContract & resource : plan.resources.resources) {
        if (!resource.elidable && storages.count(resource.storage) == 0) {
            result.errors.push_back("binding snapshot omits pinned storage " + std::to_string(resource.storage));
        }
    }
    return result;
}

AllocationFingerprint fingerprint_bindings(const BindingSnapshot & snapshot) {
    std::ostringstream witness;
    witness << snapshot.device_identity << '\n';
    std::vector<ConcreteBinding> bindings = snapshot.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const ConcreteBinding & a, const ConcreteBinding & b) {
        return a.storage < b.storage;
    });
    for (const ConcreteBinding & binding : bindings) {
        witness << binding.storage << ':' << binding.buffer_identity << ':' << binding.generation << ':'
                << binding.capacity << ':' << binding.offset << ':' << binding.length << '\n';
    }
    return { stable_hash(witness.str()) };
}

std::string format_resource_program(const ResourceProgram & resources) {
    std::ostringstream out;
    out << "resources count=" << resources.resources.size() << " uses=" << resources.uses.size() << '\n';
    for (const ResourceContract & resource : resources.resources) {
        out << "  storage " << resource.storage << " bytes=" << resource.size << " versions=0.." << resource.final_version
            << " lifetime=";
        if (resource.first_invocation == UINT32_MAX) out << "unused";
        else out << resource.first_invocation << ".." << resource.last_invocation;
        out << " flags=" << (resource.imported ? "I" : "-") << (resource.weight ? "W" : "-")
            << (resource.mutable_state ? "M" : "-") << (resource.exported ? "E" : "-")
            << (resource.elidable ? "T" : "-") << " aliases=" << resource.aliases.size() << '\n';
    }
    return out.str();
}

std::string format_command_program(const CommandProgram & program) {
    std::ostringstream out;
    out << "command-program " << program.schema << " workload=" << program.workload << " target=" << program.target
        << " graph=" << program.graph_fingerprint << " recipe=" << program.recipe_revision
        << " commands=" << program.commands.size() << '\n';
    for (const Command & command : program.commands) {
        out << "  command " << command.ordinal << ' ' << command_kind_name(command.kind) << ' '
            << kernel_specialization_name(command.kernel)
            << " execution=" << execution_kind_name(command.kernel.execution_kind)
            << " label=" << command.label << " deps=[";
        for (size_t i = 0; i < command.dependencies.size(); ++i) out << (i ? "," : "") << command.dependencies[i];
        out << "] scalars={";
        size_t scalar_index = 0;
        for (const auto & scalar : command.kernel.integer_parameters) out << (scalar_index++ ? "," : "") << scalar.first << '=' << scalar.second;
        out << "} configs={";
        size_t config_index = 0;
        for (const auto & config : command.kernel.compile_parameters) out << (config_index++ ? "," : "") << config.first << '=' << config.second;
        out << '}';
        if (command.streamed.valid()) {
            out << " streamed={layer=" << command.streamed.layer << ",weight=" << command.streamed.weight
                << ",expert_ids=" << command.streamed.expert_ids
                << ",missing_suffix=" << (command.streamed.missing_suffix ? "true" : "false") << '}';
        }
        out << '\n';
        for (size_t i = 0; i < command.bindings.size(); ++i) {
            const CommandBinding & binding = command.bindings[i];
            out << "    binding[" << i << "] " << binding.name << ' ' << resource_access_name(binding.access)
                << " storage=" << binding.storage << " range=" << binding.offset << "+" << binding.length;
            if (binding.origin == BindingOrigin::Transient) out << " transient=" << binding.transient;
            if (binding.origin == BindingOrigin::PersistentConstant) {
                out << " persistent_constant=" << binding.persistent_constant;
            }
            if (!binding.storage_binding.empty()) out << " storage_binding=" << binding.storage_binding;
            out << '\n';
        }
    }
    out << "transients arena=" << program.transients.arena_size << " alignment=" << program.transients.arena_alignment
        << " allocations=" << program.transients.allocations.size() << '\n';
    for (const TransientAllocation & allocation : program.transients.allocations) {
        out << "  transient " << allocation.id << " storage=" << allocation.storage << " range="
            << allocation.arena_offset << "+" << allocation.size << " live=" << allocation.first_command
            << ".." << allocation.last_command << '\n';
    }
    out << "persistent-constants arena=" << program.persistent_constants.arena_size
        << " alignment=" << program.persistent_constants.arena_alignment
        << " allocations=" << program.persistent_constants.allocations.size() << '\n';
    for (const PersistentConstantAllocation & allocation : program.persistent_constants.allocations) {
        out << "  persistent-constant " << allocation.id << " storage=" << allocation.storage << " range="
            << allocation.arena_offset << "+" << allocation.size << '\n';
    }
    for (const ConstantInitialization & initialization : program.initializations) {
        out << "  initialize " << initialization.label << " storage=" << initialization.storage
            << " bytes=" << initialization.data.size() << '\n';
    }
    return out.str();
}

std::string format_verification_errors(const std::vector<std::string> & errors) {
    std::ostringstream out;
    for (const std::string & error : errors) out << "error=" << error << '\n';
    return out.str();
}

std::string format_verification_summary(const std::vector<std::string> & errors) {
    std::map<std::string, size_t> counts;
    for (std::string error : errors) {
        if (error.rfind("command ", 0) == 0 && error.size() > 8 && std::isdigit(static_cast<unsigned char>(error[8]))) {
            const size_t ordinal_end = error.find(' ', 8);
            if (ordinal_end != std::string::npos) error.replace(8, ordinal_end - 8, "<ordinal>");
        }
        ++counts[error];
    }
    std::ostringstream out;
    out << "error_count=" << errors.size() << '\n';
    for (const auto & item : counts) out << "error_summary=" << item.second << '\t' << item.first << '\n';
    return out.str();
}

std::string format_binding_snapshot(const BindingSnapshot & snapshot, bool include_runtime_identities) {
    std::ostringstream out;
    out << "binding-snapshot device=" << snapshot.device_identity << " bindings=" << snapshot.bindings.size()
        << " fingerprint=" << fingerprint_bindings(snapshot).value << '\n';
    std::vector<ConcreteBinding> bindings = snapshot.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const ConcreteBinding & a, const ConcreteBinding & b) {
        return a.storage < b.storage;
    });
    for (const ConcreteBinding & binding : bindings) {
        out << "  storage " << binding.storage;
        if (include_runtime_identities) out << " buffer=0x" << std::hex << binding.buffer_identity << std::dec;
        else out << " buffer=<runtime>";
        out << " generation=" << binding.generation << " capacity=" << binding.capacity
            << " range=" << binding.offset << "+" << binding.length << '\n';
    }
    return out.str();
}

std::string serialize_command_program_json(const CommandProgram & program) {
    nlohmann::ordered_json root = {
        { "schema", program.schema }, { "workload", program.workload }, { "target", program.target },
        { "graph_fingerprint", program.graph_fingerprint }, { "recipe_revision", program.recipe_revision },
        { "commands", nlohmann::ordered_json::array() }, { "initializations", nlohmann::ordered_json::array() },
        { "transients", { { "arena_size", program.transients.arena_size },
            { "arena_alignment", program.transients.arena_alignment }, { "allocations", nlohmann::ordered_json::array() } } },
        { "persistent_constants", { { "arena_size", program.persistent_constants.arena_size },
            { "arena_alignment", program.persistent_constants.arena_alignment },
            { "allocations", nlohmann::ordered_json::array() } } },
    };
    for (const Command & command : program.commands) {
        nlohmann::ordered_json item = {
            { "ordinal", command.ordinal }, { "kind", command_kind_name(command.kind) }, { "label", command.label },
            { "family", command.kernel.family }, { "kernel", command.kernel.variant }, { "scalars", command.kernel.integer_parameters },
            { "execution", execution_kind_name(command.kernel.execution_kind) },
            { "compile_parameters", command.kernel.compile_parameters },
            { "deferred_compile_parameters", nlohmann::ordered_json::object() },
            { "workgroup_count", command.workgroup_count }, { "workgroup_size", command.workgroup_size },
            { "subgroup_size", command.subgroup_size }, { "dependencies", command.dependencies },
            { "bindings", nlohmann::ordered_json::array() },
        };
        for (const auto & parameter : command.kernel.deferred_compile_parameters) {
            item["deferred_compile_parameters"][parameter.first] = {
                { "value", parameter.second.value }, { "property", parameter.second.property },
            };
        }
        if (command.streamed.valid()) {
            item["streamed"] = {
                { "layer", command.streamed.layer }, { "weight", command.streamed.weight },
                { "expert_ids", command.streamed.expert_ids },
                { "missing_suffix", command.streamed.missing_suffix },
                { "expert_begin", command.streamed.expert_begin },
                { "expert_end", command.streamed.expert_end },
                { "load_chunk_size", command.streamed.load_chunk_size },
            };
        }
        for (const CommandBinding & binding : command.bindings) item["bindings"].push_back(binding_json(binding));
        root["commands"].push_back(std::move(item));
    }
    for (const ConstantInitialization & initialization : program.initializations) {
        std::ostringstream bytes;
        bytes << std::hex << std::setfill('0');
        for (uint8_t byte : initialization.data) bytes << std::setw(2) << static_cast<unsigned>(byte);
        root["initializations"].push_back({ { "label", initialization.label }, { "storage", initialization.storage },
                                             { "bytes_hex", bytes.str() } });
    }
    for (const TransientAllocation & allocation : program.transients.allocations) {
        root["transients"]["allocations"].push_back({
            { "id", allocation.id }, { "storage", allocation.storage }, { "size", allocation.size },
            { "alignment", allocation.alignment }, { "arena_offset", allocation.arena_offset },
            { "first_command", allocation.first_command }, { "last_command", allocation.last_command },
        });
    }
    for (const PersistentConstantAllocation & allocation : program.persistent_constants.allocations) {
        root["persistent_constants"]["allocations"].push_back({
            { "id", allocation.id }, { "storage", allocation.storage }, { "size", allocation.size },
            { "alignment", allocation.alignment }, { "arena_offset", allocation.arena_offset },
        });
    }
    return root.dump();
}

std::string serialize_binding_snapshot_json(const BindingSnapshot & snapshot, bool include_runtime_identities) {
    nlohmann::ordered_json root = {
        { "schema", "ggml-hrx-binding-snapshot-v1" }, { "device", snapshot.device_identity },
        { "fingerprint", fingerprint_bindings(snapshot).value }, { "bindings", nlohmann::ordered_json::array() },
    };
    std::vector<ConcreteBinding> bindings = snapshot.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const ConcreteBinding & a, const ConcreteBinding & b) {
        return a.storage < b.storage;
    });
    for (const ConcreteBinding & binding : bindings) {
        nlohmann::ordered_json item = {
            { "storage", binding.storage }, { "generation", binding.generation }, { "capacity", binding.capacity },
            { "offset", binding.offset }, { "length", binding.length },
        };
        if (include_runtime_identities) item["buffer_identity"] = binding.buffer_identity;
        root["bindings"].push_back(std::move(item));
    }
    return root.dump();
}

std::string command_program_dot(const CommandProgram & program) {
    std::ostringstream out;
    out << "digraph hrx_commands {\n  rankdir=LR;\n  node [shape=box,fontname=monospace];\n";
    std::vector<std::set<uint32_t>> ancestors(program.commands.size());
    for (const Command & command : program.commands) {
        out << "  c" << command.ordinal << " [label=\"" << command.ordinal << ": "
            << kernel_specialization_name(command.kernel)
            << "\\n" << command.label << "\"];\n";
        for (uint32_t dependency : command.dependencies) {
            bool redundant = false;
            for (uint32_t other : command.dependencies) {
                if (other != dependency && other < ancestors.size() && ancestors[other].count(dependency) != 0) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) out << "  c" << dependency << " -> c" << command.ordinal << ";\n";
            if (dependency < ancestors.size()) {
                ancestors[command.ordinal].insert(dependency);
                ancestors[command.ordinal].insert(ancestors[dependency].begin(), ancestors[dependency].end());
            }
        }
    }
    out << "}\n";
    return out.str();
}

} // namespace ggml::hrx
