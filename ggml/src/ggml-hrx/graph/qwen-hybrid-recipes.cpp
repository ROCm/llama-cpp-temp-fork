#include "qwen-hybrid-recipes.h"

#include "../kernel-corpus.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ggml::hrx {
namespace {

bool metadata_only(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

bool no_dispatch_operation(const Graph & graph, OperationId operation_id) {
    const Operation & operation = graph.operations[operation_id];
    if (metadata_only(operation.op)) {
        return true;
    }
    const Value & output = graph.values[operation.output];
    return std::any_of(output.access.shape.begin(), output.access.shape.end(),
                       [](int64_t extent) { return extent == 0; });
}

void close_over_no_dispatch_paths(const GraphIndex &               index,
                                  const std::vector<OperationId> & domain,
                                  std::vector<OperationId> &       operations) {
    const Graph &          graph = index.graph();
    std::vector<uint8_t>   allowed(graph.operations.size(), 0);
    std::vector<uint8_t>   forward(graph.operations.size(), 0);
    std::vector<uint8_t>   backward(graph.operations.size(), 0);
    std::queue<OperationId> worklist;
    for (OperationId operation : domain) {
        allowed[operation] = 1;
    }
    for (OperationId operation : operations) {
        forward[operation] = 1;
        worklist.push(operation);
    }
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId successor : index.successors(current)) {
            if (allowed[successor] != 0 && forward[successor] == 0 && no_dispatch_operation(graph, successor)) {
                forward[successor] = 1;
                worklist.push(successor);
            }
        }
    }
    for (OperationId operation : operations) {
        backward[operation] = 1;
        worklist.push(operation);
    }
    while (!worklist.empty()) {
        const OperationId current = worklist.front();
        worklist.pop();
        for (OperationId predecessor : index.predecessors(current)) {
            if (allowed[predecessor] != 0 && backward[predecessor] == 0 &&
                no_dispatch_operation(graph, predecessor)) {
                backward[predecessor] = 1;
                worklist.push(predecessor);
            }
        }
    }
    for (OperationId operation : domain) {
        if (forward[operation] != 0 && backward[operation] != 0 && no_dispatch_operation(graph, operation)) {
            operations.push_back(operation);
        }
    }
    std::sort(operations.begin(), operations.end());
    operations.erase(std::unique(operations.begin(), operations.end()), operations.end());
}

enum ggml_op parse_operation(const std::string & name) {
    for (int value = 0; value < GGML_OP_COUNT; ++value) {
        const auto   op        = static_cast<enum ggml_op>(value);
        const char * candidate = ggml_op_name(op);
        if (candidate != nullptr && name == candidate) {
            return op;
        }
    }
    return GGML_OP_COUNT;
}

bool parse_params_hex(const std::string & encoded, std::array<uint8_t, GGML_MAX_OP_PARAMS> & result) {
    if (encoded.size() != result.size() * 2) {
        return false;
    }
    const auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') {
            return value - '0';
        }
        if (value >= 'a' && value <= 'f') {
            return value - 'a' + 10;
        }
        if (value >= 'A' && value <= 'F') {
            return value - 'A' + 10;
        }
        return -1;
    };
    for (size_t i = 0; i < result.size(); ++i) {
        const int high = nibble(encoded[2 * i]);
        const int low  = nibble(encoded[2 * i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        result[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

constexpr int32_t kDestinationPosition = -1;
constexpr int32_t kInvalidPosition     = -2;

int32_t parse_position(const std::string & position) {
    if (position == "dst") {
        return kDestinationPosition;
    }
    if (position.rfind("src", 0) != 0 || position.size() == 3) {
        return kInvalidPosition;
    }
    char *              end   = nullptr;
    const unsigned long index = std::strtoul(position.c_str() + 3, &end, 10);
    if (end == nullptr || *end != '\0' || index > static_cast<unsigned long>(std::numeric_limits<int32_t>::max())) {
        return kInvalidPosition;
    }
    return static_cast<int32_t>(index);
}

ValueId operation_value(const Operation & operation, int32_t position) {
    if (position == kDestinationPosition) {
        return operation.output;
    }
    if (position < 0 || static_cast<size_t>(position) >= operation.inputs.size()) {
        return kInvalidId;
    }
    return operation.inputs[static_cast<size_t>(position)];
}

struct TensorDescriptor {
    enum ggml_type                     type    = GGML_TYPE_COUNT;
    size_t                             offset  = std::numeric_limits<size_t>::max();
    std::array<int64_t, GGML_MAX_DIMS> shape   = {};
    std::array<size_t, GGML_MAX_DIMS>  strides = {};
    bool                               valid   = false;
};

enum ggml_type parse_type(const std::string & name) {
    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = static_cast<enum ggml_type>(value);
        if (name == ggml_type_name(type)) {
            return type;
        }
    }
    return GGML_TYPE_COUNT;
}

TensorDescriptor parse_descriptor(const nlohmann::json & encoded) {
    TensorDescriptor result;
    result.type        = parse_type(encoded.value("type", std::string()));
    result.offset      = encoded.value("offset", std::numeric_limits<size_t>::max());
    const auto shape   = encoded.value("shape", std::vector<int64_t>());
    const auto strides = encoded.value("strides", std::vector<size_t>());
    if (result.type == GGML_TYPE_COUNT || shape.size() != result.shape.size() ||
        strides.size() != result.strides.size()) {
        return result;
    }
    std::copy(shape.begin(), shape.end(), result.shape.begin());
    std::copy(strides.begin(), strides.end(), result.strides.begin());
    result.valid = true;
    return result;
}

bool descriptor_matches(const Value & value, const TensorDescriptor & descriptor) {
    if (!descriptor.valid || descriptor.type != value.type || descriptor.offset != value.access.offset) {
        return false;
    }
    return descriptor.shape == value.access.shape && descriptor.strides == value.access.strides;
}

struct Region {
    int32_t                                  layer = -1;
    std::vector<OperationId>                 operations;
    std::vector<const QwenHybridComponent *> components;
};

std::vector<Region> regions(const QwenHybridTransformerModel & model) {
    std::vector<Region> result;
    result.push_back({ -1, model.preamble.operations, { &model.preamble } });
    for (size_t block_index = 0; block_index < model.blocks.size(); ++block_index) {
        const QwenHybridBlock & block = model.blocks[block_index];
        Region                  region;
        region.layer      = static_cast<int32_t>(block.ordinal);
        region.operations = block.operations;
        for (const QwenHybridComponent & component : block.components) {
            region.components.push_back(&component);
        }
        if (block_index + 1 < model.blocks.size()) {
            const QwenHybridBlock & next = model.blocks[block_index + 1];
            region.operations.insert(region.operations.end(), next.operations.begin(), next.operations.end());
            for (const QwenHybridComponent & component : next.components) {
                region.components.push_back(&component);
            }
        }
        result.push_back(std::move(region));
    }
    result.push_back({ -1, model.endpoint.operations, { &model.endpoint } });
    return result;
}

int32_t owning_layer(const QwenHybridTransformerModel & model, const std::vector<uint32_t> & components) {
    int32_t                  result = -1;
    const std::set<uint32_t> covered(components.begin(), components.end());
    for (const QwenHybridBlock & block : model.blocks) {
        const bool owns =
            std::any_of(block.components.begin(), block.components.end(),
                        [&](const QwenHybridComponent & component) { return covered.count(component.id) != 0; });
        if (owns && (result < 0 || static_cast<int32_t>(block.ordinal) < result)) {
            result = static_cast<int32_t>(block.ordinal);
        }
    }
    return result;
}

std::vector<uint32_t> owning_components(const Region & region, const std::vector<OperationId> & operations) {
    const std::set<OperationId> covered(operations.begin(), operations.end());
    std::vector<uint32_t>       result;
    for (const QwenHybridComponent * component : region.components) {
        if (std::any_of(component->operations.begin(), component->operations.end(),
                        [&](OperationId operation) { return covered.count(operation) != 0; })) {
            result.push_back(component->id);
        }
    }
    return result;
}

size_t eliminated_bytes(const GraphIndex & index, const std::vector<OperationId> & operations) {
    const Graph &           graph    = index.graph();
    const RegionBoundary    boundary = index.boundary(operations);
    const std::set<ValueId> materialized(boundary.outputs.begin(), boundary.outputs.end());
    size_t                  result = 0;
    for (OperationId operation : operations) {
        const Value & value = graph.values[graph.operations[operation].output];
        if (materialized.count(value.id) != 0 || metadata_only(graph.operations[operation].op)) {
            continue;
        }
        const Storage & storage   = graph.storages[value.access.storage];
        const size_t    available = storage.size > value.access.offset ? storage.size - value.access.offset : 0;
        if (available < ggml_type_size(value.type)) {
            continue;
        }
        size_t span = ggml_type_size(value.type);
        for (size_t dimension = 0; dimension < value.access.shape.size(); ++dimension) {
            if (value.access.shape[dimension] > 0) {
                span += static_cast<size_t>(value.access.shape[dimension] - 1) * value.access.strides[dimension];
            }
        }
        result += std::min(span, available);
    }
    return result;
}

struct PatternMatch {
    std::map<std::string, OperationId> operations;
    std::map<std::string, ValueId>     tensors;
};

struct PatternTensor {
    int32_t     position = kInvalidPosition;
    std::string role;
};

struct PatternOperation {
    std::string                             label;
    enum ggml_op                            op     = GGML_OP_COUNT;
    std::array<uint8_t, GGML_MAX_OP_PARAMS> params = {};
    std::vector<PatternTensor>              tensors;
};

struct TypedPattern {
    std::vector<PatternOperation>           operations;
    std::map<std::string, size_t>           operation_by_label;
    std::map<std::string, TensorDescriptor> tensors;
};

std::vector<PatternMatch> match_pattern(const Graph & graph, const Region & region, const TypedPattern & pattern) {
    std::map<std::string, std::vector<OperationId>> candidates;
    for (const PatternOperation & expected : pattern.operations) {
        for (OperationId operation_id : region.operations) {
            const Operation & operation = graph.operations[operation_id];
            if (operation.op != expected.op || operation.raw_params != expected.params) {
                continue;
            }
            bool compatible = true;
            for (const PatternTensor & tensor : expected.tensors) {
                const ValueId value_id   = operation_value(operation, tensor.position);
                const auto    descriptor = pattern.tensors.find(tensor.role);
                if (value_id >= graph.values.size() || descriptor == pattern.tensors.end() ||
                    !descriptor_matches(graph.values[value_id], descriptor->second)) {
                    compatible = false;
                    break;
                }
            }
            if (compatible) {
                candidates[expected.label].push_back(operation_id);
            }
        }
        if (candidates[expected.label].empty()) {
            return {};
        }
    }

    std::vector<std::string> labels;
    for (const auto & candidate : candidates) {
        labels.push_back(candidate.first);
    }
    std::sort(labels.begin(), labels.end(), [&](const std::string & lhs, const std::string & rhs) {
        if (candidates[lhs].size() != candidates[rhs].size()) {
            return candidates[lhs].size() < candidates[rhs].size();
        }
        return lhs < rhs;
    });

    std::vector<PatternMatch>   matches;
    PatternMatch                current;
    std::set<OperationId>       used;
    std::function<void(size_t)> visit = [&](size_t depth) {
        if (matches.size() >= 16384) {
            return;
        }
        if (depth == labels.size()) {
            matches.push_back(current);
            return;
        }
        const std::string &      label    = labels[depth];
        const PatternOperation & expected = pattern.operations[pattern.operation_by_label.at(label)];
        for (OperationId operation_id : candidates[label]) {
            if (!used.insert(operation_id).second) {
                continue;
            }
            const Operation &        operation  = graph.operations[operation_id];
            bool                     compatible = true;
            std::vector<std::string> additions;
            for (const PatternTensor & tensor : expected.tensors) {
                const ValueId value    = operation_value(operation, tensor.position);
                const auto    existing = current.tensors.find(tensor.role);
                if (existing != current.tensors.end() && existing->second != value) {
                    compatible = false;
                    break;
                }
                if (existing == current.tensors.end()) {
                    current.tensors.emplace(tensor.role, value);
                    additions.push_back(tensor.role);
                }
            }
            if (compatible) {
                current.operations.emplace(label, operation_id);
                visit(depth + 1);
                current.operations.erase(label);
            }
            for (const std::string & role : additions) {
                current.tensors.erase(role);
            }
            used.erase(operation_id);
        }
    };
    visit(0);
    return matches;
}

struct LoadedRecipe {
    std::shared_ptr<const nlohmann::json> definition;
    bool                                  has_target = false;
    std::string                           target;
    std::string                           id;
    size_t                                dispatch_count = 0;
    std::vector<TypedPattern>             patterns;
};

struct LoadedRecipes {
    std::vector<LoadedRecipe> recipes;
    std::vector<std::string>  errors;
};

const LoadedRecipes & load_recipes() {
    static const LoadedRecipes result = [] {
        LoadedRecipes         loaded;
        const kernel_source * source = get_kernel_source("qwen36/native-recipes.json");
        if (source == nullptr || source->source.data == nullptr || source->source.length == 0) {
            loaded.errors.push_back("Qwen hybrid native recipe corpus is absent");
            return loaded;
        }
        try {
            const nlohmann::json root =
                nlohmann::json::parse(source->source.data, source->source.data + source->source.length);
            if (root.value("schema", std::string()) != "ggml-hrx-native-physical-recipes-v1" ||
                root.value("domain", std::string()) != "llm.qwen36_hybrid_transformer") {
                loaded.errors.push_back("Qwen hybrid native recipe corpus has an invalid schema");
                return loaded;
            }
            for (const nlohmann::json & encoded_recipe : root.at("recipes")) {
                LoadedRecipe recipe;
                recipe.definition         = std::make_shared<const nlohmann::json>(encoded_recipe);
                const auto encoded_target = encoded_recipe.find("target");
                recipe.has_target         = encoded_target != encoded_recipe.end();
                if (recipe.has_target) {
                    recipe.target = encoded_target->get<std::string>();
                }
                recipe.id             = encoded_recipe.at("id").get<std::string>();
                recipe.dispatch_count = encoded_recipe.at("dispatches").size();
                for (const nlohmann::json & encoded_pattern : encoded_recipe.at("patterns")) {
                    TypedPattern pattern;
                    for (auto descriptor = encoded_pattern.at("tensors").begin();
                         descriptor != encoded_pattern.at("tensors").end(); ++descriptor) {
                        TensorDescriptor parsed = parse_descriptor(descriptor.value());
                        if (!parsed.valid) {
                            throw std::runtime_error("invalid tensor descriptor in recipe " + recipe.id);
                        }
                        pattern.tensors.emplace(descriptor.key(), std::move(parsed));
                    }
                    for (auto encoded_operation = encoded_pattern.at("operations").begin();
                         encoded_operation != encoded_pattern.at("operations").end(); ++encoded_operation) {
                        PatternOperation operation;
                        operation.label = encoded_operation.key();
                        operation.op    = parse_operation(encoded_operation.value().at("op").get<std::string>());
                        if (operation.op == GGML_OP_COUNT ||
                            !parse_params_hex(encoded_operation.value().at("params").get<std::string>(),
                                              operation.params)) {
                            throw std::runtime_error("invalid operation descriptor in recipe " + recipe.id);
                        }
                        for (auto tensor = encoded_operation.value().at("tensors").begin();
                             tensor != encoded_operation.value().at("tensors").end(); ++tensor) {
                            PatternTensor binding;
                            binding.position = parse_position(tensor.key());
                            binding.role     = tensor.value().get<std::string>();
                            if (binding.position == kInvalidPosition || pattern.tensors.count(binding.role) == 0) {
                                throw std::runtime_error("invalid tensor binding in recipe " + recipe.id);
                            }
                            operation.tensors.push_back(std::move(binding));
                        }
                        pattern.operation_by_label.emplace(operation.label, pattern.operations.size());
                        pattern.operations.push_back(std::move(operation));
                    }
                    recipe.patterns.push_back(std::move(pattern));
                }
                loaded.recipes.push_back(std::move(recipe));
            }
        } catch (const std::exception & error) {
            loaded.errors.push_back(std::string("cannot parse Qwen hybrid native recipe corpus: ") + error.what());
        }
        return loaded;
    }();
    return result;
}

}  // namespace

QwenHybridRecipeDiscovery QwenHybridRecipeDiscovery::discover(const GraphIndex &                 index,
                                                              const QwenHybridTransformerModel & model,
                                                              const std::string &                target) {
    QwenHybridRecipeDiscovery result;
    const LoadedRecipes &     loaded = load_recipes();
    result.errors                    = loaded.errors;
    if (!result.errors.empty()) {
        return result;
    }
    const Graph &             graph         = index.graph();
    const std::vector<Region> model_regions = regions(model);
    std::set<std::string>     unique;
    for (const LoadedRecipe & recipe : loaded.recipes) {
        if (recipe.has_target && recipe.target != target) {
            continue;
        }
        for (const Region & region : model_regions) {
            for (const TypedPattern & pattern : recipe.patterns) {
                for (PatternMatch & match : match_pattern(graph, region, pattern)) {
                    std::vector<OperationId> operations;
                    for (const auto & item : match.operations) {
                        operations.push_back(item.second);
                    }
                    std::sort(operations.begin(), operations.end());
                    std::ostringstream key;
                    key << recipe.id << ':';
                    for (OperationId operation : operations) {
                        key << operation << ',';
                    }
                    if (!unique.insert(key.str()).second) {
                        continue;
                    }
                    auto native                = std::make_shared<QwenHybridRecipeMatch>();
                    native->recipe             = recipe.id;
                    native->definition         = recipe.definition;
                    native->operations_by_role = std::move(match.operations);
                    native->values_by_role     = std::move(match.tensors);
                    native->operations         = std::move(operations);
                    close_over_no_dispatch_paths(index, region.operations, native->operations);
                    native->logical_components               = owning_components(region, native->operations);
                    native->layer                            = owning_layer(model, native->logical_components);
                    native->dispatch_count                   = recipe.dispatch_count;
                    native->eliminated_materialization_bytes = eliminated_bytes(index, native->operations);
                    native->allow_disconnected =
                        index.validate_region(native->operations, index.boundary(native->operations).outputs, false)
                            .reason == DecisionReason::DisconnectedRegion;
                    result.matches.push_back(std::move(native));
                }
            }
        }
    }
    std::sort(result.matches.begin(), result.matches.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs->operations != rhs->operations) {
            return lhs->operations < rhs->operations;
        }
        return lhs->recipe < rhs->recipe;
    });
    return result;
}

}  // namespace ggml::hrx
