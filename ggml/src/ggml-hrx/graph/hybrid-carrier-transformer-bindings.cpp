#include "hybrid-carrier-transformer-bindings.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

struct ScalarValue {
    enum class Kind { Integer, Floating } kind = Kind::Integer;
    int64_t integer = 0;
    double floating = 0.0;

    static ScalarValue from_integer(int64_t value) {
        ScalarValue result;
        result.integer = value;
        return result;
    }

    static ScalarValue from_floating(double value) {
        ScalarValue result;
        result.kind = Kind::Floating;
        result.floating = value;
        return result;
    }
};

static bool checked_add(int64_t lhs, int64_t rhs, int64_t & result) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) return false;
    result = lhs + rhs;
    return true;
}

static bool checked_multiply(int64_t lhs, int64_t rhs, int64_t & result) {
    const __int128 product = static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
    if (product < std::numeric_limits<int64_t>::min() ||
        product > std::numeric_limits<int64_t>::max()) return false;
    result = static_cast<int64_t>(product);
    return true;
}

static ValueId append_scratch(Graph & graph, const std::string & name, size_t byte_length) {
    Storage storage;
    storage.id = static_cast<StorageId>(graph.storages.size());
    storage.root = static_cast<ValueId>(graph.values.size());
    storage.size = byte_length;
    graph.storages.push_back(storage);

    Value value;
    value.id = storage.root;
    value.type = GGML_TYPE_I8;
    value.op = GGML_OP_NONE;
    value.access.storage = storage.id;
    value.access.shape = { static_cast<int64_t>(byte_length), 1, 1, 1 };
    value.access.strides = { 1, byte_length, byte_length, byte_length };
    value.name = "hrx.synthetic.hybrid." + name;
    graph.values.push_back(std::move(value));
    return storage.root;
}

class RecipeEvaluator {
public:
    RecipeEvaluator(const Graph & graph, const HybridCarrierRecipeMatch & match,
                    std::vector<std::string> & errors)
        : graph_(graph), match_(match), definition_(*match.definition), errors_(errors) {
        seed_tensors();
        seed_attributes();
    }

    bool integer(const nlohmann::json & expression, int64_t & result) {
        ScalarValue value;
        if (!evaluate(expression, value)) return false;
        if (value.kind != ScalarValue::Kind::Integer) {
            errors_.push_back(context() + ": expected an integer expression");
            return false;
        }
        result = value.integer;
        return true;
    }

    bool scalar(const nlohmann::json & expression, ScalarValue & result) {
        return evaluate(expression, result);
    }

    bool deferred_storage(const std::string & source, DeferredCompileParameter & result) {
        static constexpr const char prefix[] = "storage.";
        if (source.rfind(prefix, 0) != 0) return false;
        const size_t role_end = source.find('.', sizeof(prefix) - 1);
        if (role_end == std::string::npos || role_end + 1 == source.size()) {
            errors_.push_back(context() + ": malformed storage property " + source);
            return true;
        }
        const std::string role = source.substr(sizeof(prefix) - 1, role_end - (sizeof(prefix) - 1));
        const auto value = match_.values_by_role.find(role);
        if (value == match_.values_by_role.end()) {
            errors_.push_back(context() + ": storage property references unknown tensor role " + role);
            return true;
        }
        result.value = value->second;
        result.property = source.substr(role_end + 1);
        return true;
    }

private:
    std::string context() const { return "native recipe " + match_.recipe; }

    void set_integer(const std::string & name, int64_t value) {
        auto [position, inserted] = facts_.emplace(name, ScalarValue::from_integer(value));
        if (!inserted && (position->second.kind != ScalarValue::Kind::Integer ||
                          position->second.integer != value)) {
            errors_.push_back(context() + ": incoherent fact " + name);
        }
    }

    void set_floating(const std::string & name, double value) {
        auto [position, inserted] = facts_.emplace(name, ScalarValue::from_floating(value));
        if (!inserted && (position->second.kind != ScalarValue::Kind::Floating ||
                          position->second.floating != value)) {
            errors_.push_back(context() + ": incoherent fact " + name);
        }
    }

    void seed_tensors() {
        const nlohmann::json & schemas = definition_.at("tensor_schema");
        for (auto schema = schemas.begin(); schema != schemas.end(); ++schema) {
            const auto mapped = match_.values_by_role.find(schema.key());
            if (mapped == match_.values_by_role.end() || mapped->second >= graph_.values.size()) {
                errors_.push_back(context() + ": tensor schema has no graph value for " + schema.key());
                continue;
            }
            const Value & value = graph_.values[mapped->second];
            const std::string shape_prefix = "shape." + schema.key() + '.';
            const std::string tensor_prefix = "tensor." + schema.key() + '.';
            int64_t element_count = 1;
            for (size_t dimension = 0; dimension < value.access.shape.size(); ++dimension) {
                set_integer(shape_prefix + "d" + std::to_string(dimension), value.access.shape[dimension]);
                int64_t next_element_count = 0;
                if (value.access.shape[dimension] < 0 ||
                    !checked_multiply(element_count, value.access.shape[dimension], next_element_count)) {
                    errors_.push_back(context() + ": tensor " + schema.key() +
                                      " has an invalid element count");
                } else {
                    element_count = next_element_count;
                }
                set_integer(tensor_prefix + "strides." + std::to_string(dimension),
                            static_cast<int64_t>(value.access.strides[dimension]));
                const size_t element_size = ggml_type_size(value.type);
                if (element_size == 0 || value.access.strides[dimension] % element_size != 0) {
                    errors_.push_back(context() + ": tensor " + schema.key() +
                                      " has a non-integral element stride");
                } else {
                    set_integer(tensor_prefix + "element_strides." + std::to_string(dimension),
                                static_cast<int64_t>(value.access.strides[dimension] / element_size));
                }
            }
            set_integer(tensor_prefix + "element_count", element_count);
            set_integer(tensor_prefix + "view_offset_bytes", static_cast<int64_t>(value.access.offset));

            const nlohmann::json shape = schema.value().value("shape", nlohmann::json::array());
            if (!shape.is_array() || shape.size() != value.access.shape.size()) {
                errors_.push_back(context() + ": tensor schema rank disagrees for " + schema.key());
                continue;
            }
            for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
                if (shape[dimension].is_string()) {
                    set_integer(shape_prefix + shape[dimension].get<std::string>(),
                                value.access.shape[dimension]);
                } else if (shape[dimension].is_number_integer() &&
                           shape[dimension].get<int64_t>() != value.access.shape[dimension]) {
                    errors_.push_back(context() + ": tensor schema dimension disagrees for " + schema.key());
                }
            }
        }
    }

    void seed_attributes() {
        for (auto operation = definition_.at("attributes").begin();
             operation != definition_.at("attributes").end(); ++operation) {
            const auto mapped = match_.operations_by_role.find(operation.key());
            if (mapped == match_.operations_by_role.end() || mapped->second >= graph_.operations.size()) {
                errors_.push_back(context() + ": attribute schema has no operation for " + operation.key());
                continue;
            }
            const Operation & graph_operation = graph_.operations[mapped->second];
            for (auto attribute = operation.value().begin(); attribute != operation.value().end(); ++attribute) {
                const std::string source = attribute.value().value("source", std::string());
                if (source.rfind("op_param.", 0) != 0) {
                    errors_.push_back(context() + ": attribute " + attribute.key() +
                                      " has no explicit op-parameter source");
                    continue;
                }
                char * end = nullptr;
                const unsigned long index = std::strtoul(source.c_str() + 9, &end, 10);
                if (end == nullptr || *end != '\0') {
                    errors_.push_back(context() + ": malformed attribute source " + source);
                    continue;
                }
                const size_t offset = static_cast<size_t>(index) * sizeof(uint32_t);
                const std::string type = attribute.value().value("type", std::string());
                const std::string qualified = "attribute." + operation.key() + '.' + attribute.key();
                const std::string compact = "attribute." + attribute.key();
                if (type == "f32") {
                    if (offset + sizeof(float) > graph_operation.raw_params.size()) {
                        errors_.push_back(context() + ": f32 attribute exceeds operation parameters");
                        continue;
                    }
                    float value = 0.0f;
                    std::memcpy(&value, graph_operation.raw_params.data() + offset, sizeof(value));
                    set_floating(qualified, value);
                    if (operation.key() == "op") set_floating(compact, value);
                } else if (type == "i32") {
                    if (offset + sizeof(int32_t) > graph_operation.raw_params.size()) {
                        errors_.push_back(context() + ": i32 attribute exceeds operation parameters");
                        continue;
                    }
                    int32_t value = 0;
                    std::memcpy(&value, graph_operation.raw_params.data() + offset, sizeof(value));
                    set_integer(qualified, value);
                    if (operation.key() == "op") set_integer(compact, value);
                } else if (type == "i64") {
                    if (offset + sizeof(int64_t) > graph_operation.raw_params.size()) {
                        errors_.push_back(context() + ": i64 attribute exceeds operation parameters");
                        continue;
                    }
                    int64_t value = 0;
                    std::memcpy(&value, graph_operation.raw_params.data() + offset, sizeof(value));
                    set_integer(qualified, value);
                    if (operation.key() == "op") set_integer(compact, value);
                } else {
                    errors_.push_back(context() + ": unsupported attribute type " + type);
                }
            }
        }
    }

    bool evaluate_name(const std::string & name, ScalarValue & result) {
        const auto fact = facts_.find(name);
        if (fact != facts_.end()) {
            result = fact->second;
            return true;
        }
        if (name.rfind("derived.", 0) == 0) {
            const std::string key = name.substr(8);
            const auto derived = definition_.at("derived").find(key);
            if (derived == definition_.at("derived").end()) {
                errors_.push_back(context() + ": unknown derived value " + key);
                return false;
            }
            if (!active_derived_.insert(key).second) {
                errors_.push_back(context() + ": cyclic derived value " + key);
                return false;
            }
            const bool ok = evaluate(*derived, result);
            active_derived_.erase(key);
            if (ok) facts_[name] = result;
            return ok;
        }
        errors_.push_back(context() + ": unknown recipe value " + name);
        return false;
    }

    bool evaluate_integer_operand(const nlohmann::json & expression, int64_t & result) {
        ScalarValue value;
        if (!evaluate(expression, value)) return false;
        if (value.kind != ScalarValue::Kind::Integer) {
            errors_.push_back(context() + ": integer recipe expression consumed a floating value");
            return false;
        }
        result = value.integer;
        return true;
    }

    bool evaluate(const nlohmann::json & expression, ScalarValue & result) {
        if (expression.is_number_integer() || expression.is_number_unsigned()) {
            result = ScalarValue::from_integer(expression.get<int64_t>());
            return true;
        }
        if (expression.is_number_float()) {
            result = ScalarValue::from_floating(expression.get<double>());
            return true;
        }
        if (expression.is_string()) return evaluate_name(expression.get<std::string>(), result);
        if (!expression.is_object()) {
            errors_.push_back(context() + ": recipe expression is not scalar");
            return false;
        }
        if (expression.contains("field")) return evaluate(expression.at("field"), result);
        if (expression.contains("product") || expression.contains("sum")) {
            const bool product = expression.contains("product");
            const nlohmann::json & operands = expression.at(product ? "product" : "sum");
            int64_t accumulator = product ? 1 : 0;
            for (const nlohmann::json & operand : operands) {
                int64_t value = 0;
                if (!evaluate_integer_operand(operand, value)) return false;
                int64_t next = 0;
                const bool valid = product ? checked_multiply(accumulator, value, next)
                                           : checked_add(accumulator, value, next);
                if (!valid) {
                    errors_.push_back(context() + ": integer recipe expression overflow");
                    return false;
                }
                accumulator = next;
            }
            result = ScalarValue::from_integer(accumulator);
            return true;
        }
        if (expression.contains("ceil_div")) {
            const nlohmann::json & operands = expression.at("ceil_div");
            if (!operands.is_array() || operands.size() != 2) {
                errors_.push_back(context() + ": ceil_div requires two operands");
                return false;
            }
            int64_t numerator = 0;
            int64_t denominator = 0;
            if (!evaluate_integer_operand(operands[0], numerator) ||
                !evaluate_integer_operand(operands[1], denominator)) return false;
            if (numerator < 0 || denominator <= 0) {
                errors_.push_back(context() + ": ceil_div requires a nonnegative numerator and positive denominator");
                return false;
            }
            result = ScalarValue::from_integer(numerator / denominator + (numerator % denominator != 0));
            return true;
        }
        errors_.push_back(context() + ": unsupported recipe expression");
        return false;
    }

    const Graph & graph_;
    const HybridCarrierRecipeMatch & match_;
    const nlohmann::json & definition_;
    std::vector<std::string> & errors_;
    std::map<std::string, ScalarValue> facts_;
    std::set<std::string> active_derived_;
};

static std::string format_scalar(const ScalarValue & value, const std::string & type,
                                 const std::string & context, std::vector<std::string> & errors) {
    std::ostringstream out;
    if (type == "f32") {
        const double floating = value.kind == ScalarValue::Kind::Floating
            ? value.floating : static_cast<double>(value.integer);
        if (std::isfinite(floating) && (floating < -std::numeric_limits<float>::max() ||
            floating > std::numeric_limits<float>::max())) {
            errors.push_back(context + ": value does not fit f32");
            return {};
        }
        const float narrowed = static_cast<float>(floating);
        if (std::isnan(narrowed)) out << "nan";
        else if (std::isinf(narrowed)) out << (narrowed < 0 ? "-inf" : "inf");
        else out << std::setprecision(std::numeric_limits<float>::max_digits10) << narrowed;
    } else if (type == "i32") {
        if (value.kind != ScalarValue::Kind::Integer ||
            value.integer < std::numeric_limits<int32_t>::min() ||
            value.integer > std::numeric_limits<int32_t>::max()) {
            errors.push_back(context + ": value does not fit i32");
            return {};
        }
        out << value.integer;
    } else if (type == "i64" || type == "index") {
        if (value.kind != ScalarValue::Kind::Integer) {
            errors.push_back(context + ": floating value used as integer config");
            return {};
        }
        out << value.integer;
    } else {
        errors.push_back(context + ": unsupported scalar type " + type);
        return {};
    }
    return out.str();
}

static bool encode_launch_scalar(const ScalarValue & value, const std::string & type,
                                 int64_t & encoded, const std::string & context,
                                 std::vector<std::string> & errors) {
    if (type == "f32") {
        const double floating = value.kind == ScalarValue::Kind::Floating
            ? value.floating : static_cast<double>(value.integer);
        if (std::isfinite(floating) && (floating < -std::numeric_limits<float>::max() ||
            floating > std::numeric_limits<float>::max())) {
            errors.push_back(context + ": launch scalar does not fit f32");
            return false;
        }
        const float narrowed = static_cast<float>(floating);
        uint32_t bits = 0;
        std::memcpy(&bits, &narrowed, sizeof(bits));
        encoded = static_cast<int64_t>(bits);
        return true;
    }
    if (value.kind != ScalarValue::Kind::Integer) {
        errors.push_back(context + ": floating launch scalar used as " + type);
        return false;
    }
    if (type == "i32" && (value.integer < std::numeric_limits<int32_t>::min() ||
                           value.integer > std::numeric_limits<int32_t>::max())) {
        errors.push_back(context + ": launch scalar does not fit i32");
        return false;
    }
    if (type != "i32" && type != "i64" && type != "index") {
        errors.push_back(context + ": unsupported launch scalar type " + type);
        return false;
    }
    encoded = value.integer;
    return true;
}

} // namespace

VerificationResult materialize_hybrid_carrier_recipe(
        Graph & graph, Invocation & invocation,
        const HybridCarrierRecipeMatch & recipe, uint32_t & dispatch_ordinal) {
    VerificationResult result;
    if (!recipe.definition) {
        result.errors.push_back("native hybrid-carrier recipe has no physical definition");
        return result;
    }
    RecipeEvaluator evaluator(graph, recipe, result.errors);
    if (!result.errors.empty()) return result;

    std::map<std::string, ValueId> transients;
    for (auto item = recipe.definition->at("transient_buffers").begin();
         item != recipe.definition->at("transient_buffers").end(); ++item) {
        int64_t byte_length = 0;
        if (!evaluator.integer(item.value().at("size"), byte_length)) continue;
        if (byte_length <= 0 || static_cast<uint64_t>(byte_length) > std::numeric_limits<size_t>::max()) {
            result.errors.push_back("native recipe " + recipe.recipe +
                                    ": invalid transient size for " + item.key());
            continue;
        }
        const std::string label = std::to_string(invocation.layer) + '.' + recipe.recipe + '.' +
            std::to_string(graph.storages.size()) + '.' + item.key();
        transients[item.key()] = append_scratch(graph, label, static_cast<size_t>(byte_length));
    }
    if (!result.errors.empty()) return result;

    for (const nlohmann::json & physical : recipe.definition->at("dispatches")) {
        Dispatch dispatch;
        dispatch.kernel.family = "deepseek4";
        dispatch.kernel.variant = physical.at("definition").get<std::string>();
        dispatch.kernel.kernel_id = kernel_catalog_id(
            dispatch.kernel.family.c_str(), dispatch.kernel.variant.c_str());
        dispatch.kernel.execution_kind = KernelSpecialization::ExecutionKind::Native;

        std::vector<nlohmann::json> buffers = physical.at("buffers").get<std::vector<nlohmann::json>>();
        std::sort(buffers.begin(), buffers.end(), [](const nlohmann::json & lhs, const nlohmann::json & rhs) {
            return lhs.at("position").get<int64_t>() < rhs.at("position").get<int64_t>();
        });
        for (const nlohmann::json & buffer : buffers) {
            TensorBinding binding;
            binding.role = buffer.at("name").get<std::string>();
            if (buffer.contains("tensor")) {
                const std::string tensor = buffer.at("tensor").get<std::string>();
                const auto mapped = recipe.values_by_role.find(tensor);
                if (mapped == recipe.values_by_role.end()) {
                    result.errors.push_back("native recipe " + recipe.recipe +
                                            ": unknown tensor binding " + tensor);
                    continue;
                }
                binding.value = mapped->second;
                binding.storage_binding = buffer.value("storage_binding", std::string());
            } else if (buffer.contains("transient")) {
                const std::string transient = buffer.at("transient").get<std::string>();
                const auto mapped = transients.find(transient);
                if (mapped == transients.end()) {
                    result.errors.push_back("native recipe " + recipe.recipe +
                                            ": unknown transient binding " + transient);
                    continue;
                }
                binding.value = mapped->second;
            } else {
                result.errors.push_back("native recipe " + recipe.recipe +
                                        ": buffer has no tensor or transient source");
                continue;
            }
            dispatch.bindings.push_back(std::move(binding));
        }

        const bool consumes_streamed_storage = std::any_of(
            dispatch.bindings.begin(), dispatch.bindings.end(),
            [](const TensorBinding & binding) { return !binding.storage_binding.empty(); });
        if (consumes_streamed_storage) {
            if (!recipe.definition->contains("streaming")) {
                result.errors.push_back("native recipe " + recipe.recipe +
                                        ": cache-bound dispatch has no streaming contract");
            } else {
                const nlohmann::json & streaming = recipe.definition->at("streaming");
                const std::string weight_role = physical.value(
                    "streamed_weight_tensor",
                    streaming.at("weight_tensor").get<std::string>());
                const std::string expert_ids_role = streaming.at("expert_ids_tensor").get<std::string>();
                const auto weight = recipe.values_by_role.find(weight_role);
                const auto expert_ids = recipe.values_by_role.find(expert_ids_role);
                if (weight == recipe.values_by_role.end() || expert_ids == recipe.values_by_role.end()) {
                    result.errors.push_back("native recipe " + recipe.recipe +
                                            ": streaming contract references an unknown tensor role");
                } else {
                    dispatch.streamed_weight = weight->second;
                    dispatch.streamed_expert_ids = expert_ids->second;
                    dispatch.streamed_missing_suffix = physical.value("streamed_missing_suffix", false);
                    dispatch.streamed_expert_begin = physical.value("streamed_expert_begin", uint32_t { 0 });
                    dispatch.streamed_expert_end = physical.value("streamed_expert_end", uint32_t { 0xffffffffu });
                    dispatch.streamed_load_chunk_size = physical.value("streamed_load_chunk_size", uint32_t { 0 });
                }
            }
        } else if (physical.value("streamed_missing_suffix", false)) {
            result.errors.push_back("native recipe " + recipe.recipe +
                                    ": streamed execution marker does not consume streamed storage");
        }

        const nlohmann::json & config = physical.at("config");
        if (config.value("mode", std::string()) != "compile") {
            result.errors.push_back("native recipe " + recipe.recipe + ": unsupported config mode");
        }
        for (const nlohmann::json & binding : config.at("bindings")) {
            const std::string name = binding.at("name").get<std::string>();
            const std::string type = binding.at("type").get<std::string>();
            if (binding.contains("source")) {
                const std::string source = binding.at("source").get<std::string>();
                DeferredCompileParameter deferred;
                if (evaluator.deferred_storage(source, deferred)) {
                    if (deferred.value != kInvalidId) {
                        dispatch.kernel.deferred_compile_parameters[name] = std::move(deferred);
                    }
                    continue;
                }
                ScalarValue value;
                if (!evaluator.scalar(source, value)) continue;
                dispatch.kernel.compile_parameters[name] = format_scalar(
                    value, type, "native recipe " + recipe.recipe + " config " + name, result.errors);
            } else if (binding.contains("value")) {
                ScalarValue value;
                if (!evaluator.scalar(binding.at("value"), value)) continue;
                dispatch.kernel.compile_parameters[name] = format_scalar(
                    value, type, "native recipe " + recipe.recipe + " config " + name, result.errors);
            } else {
                result.errors.push_back("native recipe " + recipe.recipe +
                                        ": config " + name + " has no source or value");
            }
        }

        std::vector<nlohmann::json> scalars = physical.at("scalars").get<std::vector<nlohmann::json>>();
        std::sort(scalars.begin(), scalars.end(), [](const nlohmann::json & lhs, const nlohmann::json & rhs) {
            return lhs.at("position").get<int64_t>() < rhs.at("position").get<int64_t>();
        });
        for (const nlohmann::json & scalar : scalars) {
            const std::string name = scalar.at("name").get<std::string>();
            const std::string type = scalar.at("type").get<std::string>();
            ScalarValue value;
            if (!evaluator.scalar(scalar.at("source"), value)) continue;
            int64_t encoded = 0;
            if (encode_launch_scalar(value, type, encoded,
                    "native recipe " + recipe.recipe + " scalar " + name, result.errors)) {
                dispatch.kernel.integer_parameters[name] = encoded;
            }
        }

        // Verify that authored launch geometry and compile-time shape bindings agree.
        const nlohmann::json & launch = physical.at("dispatch");
        for (const nlohmann::json & value : launch.at("workgroup_size")) {
            int64_t dimension = 0;
            if (evaluator.integer(value, dimension) &&
                (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<uint32_t>::max())) {
                result.errors.push_back("native recipe " + recipe.recipe + ": invalid workgroup size");
            }
        }
        const char * extent_key = launch.contains("workgroups") ? "workgroups" : "work_items";
        for (const nlohmann::json & value : launch.at(extent_key)) {
            int64_t dimension = 0;
            if (evaluator.integer(value, dimension) &&
                (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<uint32_t>::max())) {
                result.errors.push_back("native recipe " + recipe.recipe + ": invalid dispatch extent");
            }
        }

        if (dispatch_ordinal != 0) dispatch.dependencies.push_back(dispatch_ordinal - 1);
        invocation.dispatches.push_back(std::move(dispatch));
        ++dispatch_ordinal;
    }
    if (!invocation.dispatches.empty()) invocation.kernel = invocation.dispatches.front().kernel;
    return result;
}

} // namespace ggml::hrx
