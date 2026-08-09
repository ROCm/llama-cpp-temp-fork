#include "hybrid-carrier-recipes.h"

#include "../kernel-corpus.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

bool metadata_only(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
        op == GGML_OP_TRANSPOSE;
}

enum ggml_op parse_operation(const std::string & name) {
    for (int value = 0; value < GGML_OP_COUNT; ++value) {
        const auto op = static_cast<enum ggml_op>(value);
        const char * candidate = ggml_op_name(op);
        if (candidate != nullptr && name == candidate) return op;
    }
    return GGML_OP_COUNT;
}

std::string params_hex(const Operation & operation) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result(operation.raw_params.size() * 2, '0');
    for (size_t i = 0; i < operation.raw_params.size(); ++i) {
        result[2 * i] = kHex[operation.raw_params[i] >> 4];
        result[2 * i + 1] = kHex[operation.raw_params[i] & 15];
    }
    return result;
}

ValueId operation_value(const Operation & operation, const std::string & position) {
    if (position == "dst") return operation.output;
    if (position.rfind("src", 0) != 0 || position.size() == 3) return kInvalidId;
    char * end = nullptr;
    const unsigned long index = std::strtoul(position.c_str() + 3, &end, 10);
    if (end == nullptr || *end != '\0' || index >= operation.inputs.size()) return kInvalidId;
    return operation.inputs[index];
}

bool descriptor_matches(const Value & value, const nlohmann::json & descriptor) {
    if (descriptor.value("type", std::string()) != ggml_type_name(value.type) ||
        descriptor.value("offset", std::numeric_limits<size_t>::max()) != value.access.offset) {
        return false;
    }
    const auto shape = descriptor.value("shape", std::vector<int64_t>());
    const auto strides = descriptor.value("strides", std::vector<size_t>());
    return shape.size() == value.access.shape.size() && strides.size() == value.access.strides.size() &&
        std::equal(shape.begin(), shape.end(), value.access.shape.begin()) &&
        std::equal(strides.begin(), strides.end(), value.access.strides.begin());
}

struct Region {
    int32_t layer = -1;
    std::vector<OperationId> operations;
    std::vector<const HybridCarrierComponent *> components;
};

std::vector<Region> regions(const HybridCarrierTransformerModel & model) {
    std::vector<Region> result;
    result.push_back({ -1, model.preamble.operations, { &model.preamble } });
    for (const HybridCarrierBlock & block : model.blocks) {
        Region region;
        region.layer = static_cast<int32_t>(block.ordinal);
        region.operations = block.operations;
        for (const HybridCarrierComponent & component : block.components) {
            region.components.push_back(&component);
        }
        result.push_back(std::move(region));
    }
    result.push_back({ -1, model.endpoint.operations, { &model.endpoint } });
    return result;
}

std::vector<uint32_t> owning_components(
        const Region & region, const std::vector<OperationId> & operations) {
    const std::set<OperationId> covered(operations.begin(), operations.end());
    std::vector<uint32_t> result;
    for (const HybridCarrierComponent * component : region.components) {
        if (std::any_of(component->operations.begin(), component->operations.end(),
                [&](OperationId operation) { return covered.count(operation) != 0; })) {
            result.push_back(component->id);
        }
    }
    return result;
}

size_t eliminated_bytes(const GraphIndex & index, const std::vector<OperationId> & operations) {
    const Graph & graph = index.graph();
    const RegionBoundary boundary = index.boundary(operations);
    const std::set<ValueId> materialized(boundary.outputs.begin(), boundary.outputs.end());
    size_t result = 0;
    for (OperationId operation : operations) {
        const Value & value = graph.values[graph.operations[operation].output];
        if (materialized.count(value.id) != 0 || metadata_only(graph.operations[operation].op)) continue;
        const Storage & storage = graph.storages[value.access.storage];
        const size_t available = storage.size > value.access.offset ? storage.size - value.access.offset : 0;
        if (available < ggml_type_size(value.type)) continue;
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
    std::map<std::string, ValueId> tensors;
};

std::vector<PatternMatch> match_pattern(
        const Graph & graph, const Region & region, const nlohmann::json & pattern) {
    const nlohmann::json & pattern_operations = pattern.at("operations");
    const nlohmann::json & tensor_descriptors = pattern.at("tensors");
    std::map<std::string, std::vector<OperationId>> candidates;
    for (auto item = pattern_operations.begin(); item != pattern_operations.end(); ++item) {
        const enum ggml_op expected_op = parse_operation(item.value().at("op").get<std::string>());
        const std::string expected_params = item.value().at("params").get<std::string>();
        for (OperationId operation_id : region.operations) {
            const Operation & operation = graph.operations[operation_id];
            if (operation.op != expected_op || params_hex(operation) != expected_params) continue;
            bool compatible = true;
            for (auto tensor = item.value().at("tensors").begin();
                 tensor != item.value().at("tensors").end(); ++tensor) {
                const ValueId value_id = operation_value(operation, tensor.key());
                const auto descriptor = tensor_descriptors.find(tensor.value().get<std::string>());
                if (value_id >= graph.values.size() || descriptor == tensor_descriptors.end() ||
                    !descriptor_matches(graph.values[value_id], *descriptor)) {
                    compatible = false;
                    break;
                }
            }
            if (compatible) candidates[item.key()].push_back(operation_id);
        }
        if (candidates[item.key()].empty()) return {};
    }

    std::vector<std::string> labels;
    for (const auto & candidate : candidates) labels.push_back(candidate.first);
    std::sort(labels.begin(), labels.end(), [&](const std::string & lhs, const std::string & rhs) {
        if (candidates[lhs].size() != candidates[rhs].size()) return candidates[lhs].size() < candidates[rhs].size();
        return lhs < rhs;
    });

    std::vector<PatternMatch> matches;
    PatternMatch current;
    std::set<OperationId> used;
    std::function<void(size_t)> visit = [&](size_t depth) {
        if (matches.size() >= 16384) return;
        if (depth == labels.size()) {
            matches.push_back(current);
            return;
        }
        const std::string & label = labels[depth];
        const nlohmann::json & expected = pattern_operations.at(label).at("tensors");
        for (OperationId operation_id : candidates[label]) {
            if (!used.insert(operation_id).second) continue;
            const Operation & operation = graph.operations[operation_id];
            bool compatible = true;
            std::vector<std::string> additions;
            for (auto tensor = expected.begin(); tensor != expected.end(); ++tensor) {
                const std::string role = tensor.value().get<std::string>();
                const ValueId value = operation_value(operation, tensor.key());
                const auto existing = current.tensors.find(role);
                if (existing != current.tensors.end() && existing->second != value) {
                    compatible = false;
                    break;
                }
                if (existing == current.tensors.end()) {
                    current.tensors.emplace(role, value);
                    additions.push_back(role);
                }
            }
            if (compatible) {
                current.operations.emplace(label, operation_id);
                visit(depth + 1);
                current.operations.erase(label);
            }
            for (const std::string & role : additions) current.tensors.erase(role);
            used.erase(operation_id);
        }
    };
    visit(0);
    return matches;
}

std::vector<std::shared_ptr<const nlohmann::json>> load_recipes(std::vector<std::string> & errors) {
    std::vector<std::shared_ptr<const nlohmann::json>> result;
    const kernel_source * source = get_kernel_source("deepseek4/native-recipes.json");
    if (source == nullptr || source->source.data == nullptr || source->source.length == 0) {
        errors.push_back("hybrid-carrier native recipe corpus is absent");
        return result;
    }
    try {
        const nlohmann::json root = nlohmann::json::parse(
            source->source.data, source->source.data + source->source.length);
        if (root.value("schema", std::string()) != "ggml-hrx-native-physical-recipes-v1" ||
            root.value("domain", std::string()) != "llm.hybrid_carrier_transformer") {
            errors.push_back("hybrid-carrier native recipe corpus has an invalid schema");
            return result;
        }
        for (const nlohmann::json & recipe : root.at("recipes")) {
            result.push_back(std::make_shared<const nlohmann::json>(recipe));
        }
    } catch (const std::exception & error) {
        errors.push_back(std::string("cannot parse hybrid-carrier native recipe corpus: ") + error.what());
    }
    return result;
}

} // namespace

HybridCarrierRecipeDiscovery HybridCarrierRecipeDiscovery::discover(
        const GraphIndex & index, const HybridCarrierTransformerModel & model,
        const std::string & target) {
    HybridCarrierRecipeDiscovery result;
    std::vector<std::shared_ptr<const nlohmann::json>> recipes = load_recipes(result.errors);
    if (!result.errors.empty()) return result;
    const Graph & graph = index.graph();
    std::set<std::string> unique;
    for (const auto & recipe : recipes) {
        if (recipe->value("target", target) != target) continue;
        const std::string recipe_id = recipe->at("id").get<std::string>();
        const size_t dispatch_count = recipe->at("dispatches").size();
        for (const Region & region : regions(model)) {
            for (const nlohmann::json & pattern : recipe->at("patterns")) {
                for (PatternMatch & match : match_pattern(graph, region, pattern)) {
                    std::vector<OperationId> operations;
                    for (const auto & item : match.operations) operations.push_back(item.second);
                    std::sort(operations.begin(), operations.end());
                    std::ostringstream key;
                    key << recipe_id << '@' << region.layer << ':';
                    for (OperationId operation : operations) key << operation << ',';
                    if (!unique.insert(key.str()).second) continue;
                    auto native = std::make_shared<HybridCarrierRecipeMatch>();
                    native->recipe = recipe_id;
                    native->definition = recipe;
                    native->operations_by_role = std::move(match.operations);
                    native->values_by_role = std::move(match.tensors);
                    native->operations = std::move(operations);
                    native->logical_components = owning_components(region, native->operations);
                    native->layer = region.layer;
                    native->dispatch_count = dispatch_count;
                    native->eliminated_materialization_bytes = eliminated_bytes(index, native->operations);
                    native->allow_disconnected = index.validate_region(
                        native->operations, index.boundary(native->operations).outputs, false).reason ==
                        DecisionReason::DisconnectedRegion;
                    result.matches.push_back(std::move(native));
                }
            }
        }
    }
    std::sort(result.matches.begin(), result.matches.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs->operations != rhs->operations) return lhs->operations < rhs->operations;
        return lhs->recipe < rhs->recipe;
    });
    return result;
}

} // namespace ggml::hrx
