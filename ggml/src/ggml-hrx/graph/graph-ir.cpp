#include "graph-ir.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

class GraphImplementation {
public:

static bool is_layout_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static const ggml_tensor * storage_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static size_t access_span(const AccessPath & access, enum ggml_type type) {
    const size_t blocks = (static_cast<size_t>(access.shape[0]) + ggml_blck_size(type) - 1) / ggml_blck_size(type);
    size_t result = blocks * access.strides[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (access.shape[i] > 0) {
            result += static_cast<size_t>(access.shape[i] - 1) * access.strides[i];
        }
    }
    return result;
}

static bool buffer_is_weight(const ggml_tensor * tensor) {
    const ggml_tensor * root = storage_root(tensor);
    return root != nullptr && root->buffer != nullptr &&
        ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static void hash_bytes(uint64_t & hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
}

template <typename T>
static void hash_value(uint64_t & hash, const T & value) {
    hash_bytes(hash, &value, sizeof(value));
}

static std::string fingerprint_graph(const Graph & graph) {
    std::vector<uint64_t> colors(graph.values.size(), UINT64_C(1469598103934665603));
    for (const Value & value : graph.values) {
        uint64_t color = UINT64_C(1469598103934665603);
        hash_value(color, value.type);
        hash_value(color, value.op);
        hash_bytes(color, value.access.shape.data(), sizeof(value.access.shape));
        hash_bytes(color, value.access.strides.data(), sizeof(value.access.strides));
        hash_value(color, value.access.offset);
        hash_value(color, value.access.version);
        colors[value.id] = color;
    }
    for (const Operation & operation : graph.operations) {
        uint64_t color = colors[operation.output];
        hash_value(color, operation.op);
        hash_bytes(color, operation.raw_params.data(), operation.raw_params.size());
        for (ValueId input : operation.inputs) {
            hash_value(color, colors[input]);
        }
        for (const Effect & effect : operation.effects) {
            hash_value(color, effect.kind);
            hash_value(color, effect.before_version);
            hash_value(color, effect.after_version);
            hash_value(color, effect.offset);
            hash_value(color, effect.size);
            hash_value(color, effect.exact);
        }
        colors[operation.output] = color;
    }
    std::vector<uint64_t> operation_colors;
    operation_colors.reserve(graph.operations.size());
    for (const Operation & operation : graph.operations) {
        operation_colors.push_back(colors[operation.output]);
    }
    std::sort(operation_colors.begin(), operation_colors.end());
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t color : operation_colors) {
        hash_value(hash, color);
    }
    std::vector<uint64_t> storage_colors;
    for (const Storage & storage : graph.storages) {
        uint64_t color = UINT64_C(1469598103934665603);
        hash_value(color, storage.size);
        hash_value(color, storage.external);
        hash_value(color, storage.weight);
        hash_value(color, storage.mutable_state);
        hash_value(color, storage.final_version);
        storage_colors.push_back(color);
    }
    std::sort(storage_colors.begin(), storage_colors.end());
    for (uint64_t color : storage_colors) {
        hash_value(hash, color);
    }
    std::vector<uint64_t> root_colors;
    for (ValueId root : graph.roots) {
        if (root < colors.size()) {
            root_colors.push_back(colors[root]);
        }
    }
    std::sort(root_colors.begin(), root_colors.end());
    for (uint64_t color : root_colors) {
        hash_value(hash, color);
    }
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

static std::string escape_json(const std::string & value) {
    std::ostringstream stream;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': stream << "\\\\"; break;
            case '"': stream << "\\\""; break;
            case '\n': stream << "\\n"; break;
            case '\r': stream << "\\r"; break;
            case '\t': stream << "\\t"; break;
            default:
                if (ch < 0x20) {
                    stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
                } else {
                    stream << ch;
                }
        }
    }
    return stream.str();
}

static std::string bytes_as_hex(const uint8_t * data, size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (size_t i = 0; i < size; ++i) {
        result[2 * i] = digits[data[i] >> 4];
        result[2 * i + 1] = digits[data[i] & 0xf];
    }
    return result;
}

static void validate_graph(Graph & graph) {
    for (size_t i = 0; i < graph.storages.size(); ++i) {
        const Storage & storage = graph.storages[i];
        if (storage.id != i || storage.root >= graph.values.size()) {
            graph.errors.emplace_back("invalid storage identity");
        }
    }
    for (size_t i = 0; i < graph.values.size(); ++i) {
        const Value & value = graph.values[i];
        if (value.id != i || value.access.storage >= graph.storages.size()) {
            graph.errors.emplace_back("invalid value identity or storage");
            continue;
        }
        const size_t span = access_span(value.access, value.type);
        const Storage & storage = graph.storages[value.access.storage];
        if (value.access.offset > storage.size || span > storage.size - value.access.offset) {
            graph.errors.emplace_back("value access exceeds its storage");
        }
        if (value.view_source != kInvalidId &&
            (value.view_source >= graph.values.size() || graph.values[value.view_source].access.storage != value.access.storage)) {
            graph.errors.emplace_back("view does not preserve storage identity");
        }
    }
    for (size_t i = 0; i < graph.operations.size(); ++i) {
        const Operation & operation = graph.operations[i];
        if (operation.id != i || operation.output >= graph.values.size() || graph.values[operation.output].producer != operation.id) {
            graph.errors.emplace_back("invalid operation identity or output");
            continue;
        }
        for (ValueId input : operation.inputs) {
            if (input >= graph.values.size() ||
                (graph.values[input].producer != kInvalidId && graph.values[input].producer >= operation.id)) {
                graph.errors.emplace_back("operation input is not topologically available");
            }
        }
        for (const Effect & effect : operation.effects) {
            if (effect.storage >= graph.storages.size() || effect.before_version > graph.storages[effect.storage].final_version ||
                effect.after_version > graph.storages[effect.storage].final_version) {
                graph.errors.emplace_back("invalid operation storage effect");
            }
        }
        if (operation.op == GGML_OP_SET_ROWS) {
            if (operation.inputs.size() != 3 ||
                graph.values[operation.output].access.storage != graph.values[operation.inputs[2]].access.storage) {
                graph.errors.emplace_back("SET_ROWS destination storage is not legacy operand 2");
            }
        }
    }
}

};

} // namespace

Graph Graph::deserialize_json(const std::string & text) {
    Graph graph;
    try {
        const nlohmann::json root = nlohmann::json::parse(text);
        if (root.at("version").get<int>() != 1) {
            graph.errors.emplace_back("unsupported graph fixture version");
            return graph;
        }
        graph.fingerprint = root.at("fingerprint").get<std::string>();
        auto parse_type = [](const std::string & name) {
            for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                if (name == ggml_type_name(static_cast<enum ggml_type>(i))) return static_cast<enum ggml_type>(i);
            }
            return GGML_TYPE_COUNT;
        };
        auto parse_op = [](const std::string & name) {
            for (int i = 0; i < GGML_OP_COUNT; ++i) {
                if (name == ggml_op_name(static_cast<enum ggml_op>(i))) return static_cast<enum ggml_op>(i);
            }
            return GGML_OP_COUNT;
        };
        auto parse_boundary = [](const std::string & name) {
            for (BoundaryKind kind : { BoundaryKind::Internal, BoundaryKind::Input, BoundaryKind::Weight,
                                       BoundaryKind::MutableState, BoundaryKind::Output }) {
                if (name == Value::boundary_kind_name(kind)) return kind;
            }
            return BoundaryKind::Internal;
        };
        for (const nlohmann::json & item : root.at("storages")) {
            Storage storage;
            storage.id = item.at("id").get<StorageId>();
            storage.root = item.at("root").get<ValueId>();
            storage.size = item.at("size").get<size_t>();
            storage.external = item.at("external").get<bool>();
            storage.weight = item.at("weight").get<bool>();
            storage.mutable_state = item.at("mutable").get<bool>();
            storage.final_version = item.at("version").get<uint32_t>();
            graph.storages.push_back(storage);
        }
        for (const nlohmann::json & item : root.at("values")) {
            Value value;
            value.id = item.at("id").get<ValueId>();
            value.type = parse_type(item.at("type").get<std::string>());
            value.op = parse_op(item.at("op").get<std::string>());
            value.access.storage = item.at("storage").get<StorageId>();
            value.access.version = item.at("version").get<uint32_t>();
            value.access.offset = item.at("offset").get<size_t>();
            value.flags = item.at("flags").get<int32_t>();
            value.producer = item.at("producer").get<OperationId>();
            value.view_source = item.at("view_source").get<ValueId>();
            value.boundary = parse_boundary(item.at("boundary").get<std::string>());
            value.name = item.at("name").get<std::string>();
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                value.access.shape[i] = item.at("shape").at(i).get<int64_t>();
                value.access.strides[i] = item.at("strides").at(i).get<size_t>();
            }
            graph.values.push_back(std::move(value));
        }
        for (const nlohmann::json & item : root.at("operations")) {
            Operation operation;
            operation.id = item.at("id").get<OperationId>();
            operation.op = parse_op(item.at("op").get<std::string>());
            operation.output = item.at("output").get<ValueId>();
            operation.original_ordinal = item.at("ordinal").get<int>();
            operation.inputs = item.at("inputs").get<std::vector<ValueId>>();
            const std::string params = item.at("params").get<std::string>();
            if (params.size() != operation.raw_params.size() * 2) throw std::runtime_error("invalid op parameter encoding");
            auto nibble = [](char ch) -> uint8_t {
                if (ch >= '0' && ch <= '9') return ch - '0';
                if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
                throw std::runtime_error("invalid hexadecimal digit");
            };
            for (size_t i = 0; i < operation.raw_params.size(); ++i) {
                operation.raw_params[i] = static_cast<uint8_t>((nibble(params[2*i]) << 4) | nibble(params[2*i + 1]));
            }
            for (const nlohmann::json & effect_item : item.at("effects")) {
                Effect effect;
                effect.kind = effect_item.at("kind").get<std::string>() == "read" ? EffectKind::Read : EffectKind::Write;
                effect.storage = effect_item.at("storage").get<StorageId>();
                effect.before_version = effect_item.at("before").get<uint32_t>();
                effect.after_version = effect_item.at("after").get<uint32_t>();
                effect.offset = effect_item.at("offset").get<size_t>();
                effect.size = effect_item.at("size").get<size_t>();
                effect.exact = effect_item.at("exact").get<bool>();
                operation.effects.push_back(effect);
            }
            graph.operations.push_back(std::move(operation));
        }
        graph.roots = root.at("roots").get<std::vector<ValueId>>();
        graph.errors = root.at("errors").get<std::vector<std::string>>();
        GraphImplementation::validate_graph(graph);
        const std::string computed_fingerprint = GraphImplementation::fingerprint_graph(graph);
        if (computed_fingerprint != graph.fingerprint) graph.errors.emplace_back("graph fixture fingerprint does not match contents");
    } catch (const std::exception & error) {
        graph.errors.emplace_back(std::string("invalid graph fixture: ") + error.what());
    }
    return graph;
}

Graph Graph::import(const ggml_cgraph * cgraph) {
    Graph graph;
    if (cgraph == nullptr) {
        graph.errors.emplace_back("null cgraph");
        return graph;
    }

    std::vector<const ggml_tensor *> ordered;
    std::unordered_map<const ggml_tensor *, ValueId> ids;
    auto visit = [&](auto && self, const ggml_tensor * tensor) -> void {
        if (tensor == nullptr || ids.count(tensor) != 0) {
            return;
        }
        self(self, tensor->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            self(self, tensor->src[i]);
        }
        ids.emplace(tensor, static_cast<ValueId>(ordered.size()));
        ordered.push_back(tensor);
    };
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        visit(visit, cgraph->leafs[i]);
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        visit(visit, cgraph->nodes[i]);
    }

    std::unordered_map<const ggml_tensor *, StorageId> storage_ids;
    for (const ggml_tensor * tensor : ordered) {
        const ggml_tensor * root = GraphImplementation::storage_root(tensor);
        if (root == nullptr || storage_ids.count(root) != 0) {
            continue;
        }
        Storage storage;
        storage.id = static_cast<StorageId>(graph.storages.size());
        storage.root = ids.at(root);
        storage.size = ggml_nbytes(root);
        storage.external = root->op == GGML_OP_NONE;
        storage.weight = GraphImplementation::buffer_is_weight(root);
        storage_ids.emplace(root, storage.id);
        graph.storages.push_back(storage);
    }

    graph.values.resize(ordered.size());
    std::vector<uint32_t> versions(graph.storages.size(), 0);
    std::unordered_map<const ggml_tensor *, int> node_ordinals;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        node_ordinals.emplace(cgraph->nodes[i], i);
    }

    for (const ggml_tensor * tensor : ordered) {
        Value value;
        value.id = ids.at(tensor);
        value.type = tensor->type;
        value.op = tensor->op;
        value.flags = tensor->flags;
        value.name = tensor->name;
        value.view_source = tensor->view_src != nullptr ? ids.at(tensor->view_src) : kInvalidId;
        const ggml_tensor * root = GraphImplementation::storage_root(tensor);
        value.access.storage = storage_ids.at(root);
        value.access.version = versions[value.access.storage];
        value.access.offset = tensor->view_src != nullptr ? tensor->view_offs : 0;
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), value.access.shape.begin());
        std::copy(std::begin(tensor->nb), std::end(tensor->nb), value.access.strides.begin());
        if ((tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            value.boundary = BoundaryKind::Output;
            graph.roots.push_back(value.id);
        } else if ((tensor->flags & GGML_TENSOR_FLAG_INPUT) != 0) {
            value.boundary = BoundaryKind::Input;
        } else if (graph.storages[value.access.storage].weight) {
            value.boundary = BoundaryKind::Weight;
        }

        auto ordinal = node_ordinals.find(tensor);
        if (ordinal != node_ordinals.end()) {
            Operation operation;
            operation.id = static_cast<OperationId>(graph.operations.size());
            operation.op = tensor->op;
            operation.output = value.id;
            operation.original_ordinal = ordinal->second;
            std::memcpy(operation.raw_params.data(), tensor->op_params, operation.raw_params.size());
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (tensor->src[i] != nullptr) {
                    operation.inputs.push_back(ids.at(tensor->src[i]));
                    const Value & input = graph.values[ids.at(tensor->src[i])];
                    Effect read;
                    read.kind = EffectKind::Read;
                    read.storage = input.access.storage;
                    read.before_version = versions[read.storage];
                    read.after_version = read.before_version;
                    read.offset = input.access.offset;
                    read.size = GraphImplementation::access_span(input.access, input.type);
                    read.exact = true;
                    operation.effects.push_back(read);
                }
            }

            const bool aliases_storage = tensor->view_src != nullptr && !GraphImplementation::is_layout_op(tensor->op);
            if (aliases_storage) {
                Effect write;
                write.kind = EffectKind::Write;
                write.storage = value.access.storage;
                write.before_version = versions[write.storage];
                write.after_version = ++versions[write.storage];
                write.offset = value.access.offset;
                write.size = tensor->op == GGML_OP_SET_ROWS ? graph.storages[write.storage].size : GraphImplementation::access_span(value.access, value.type);
                write.exact = tensor->op != GGML_OP_SET_ROWS;
                operation.effects.push_back(write);
                value.access.version = write.after_version;
                Storage & storage = graph.storages[write.storage];
                storage.mutable_state |= storage.external;
                storage.final_version = write.after_version;
                if (storage.mutable_state && value.boundary != BoundaryKind::Output) {
                    value.boundary = BoundaryKind::MutableState;
                }
            }
            value.producer = operation.id;
            graph.operations.push_back(std::move(operation));
        }
        graph.values[value.id] = std::move(value);
    }

    for (Storage & storage : graph.storages) {
        storage.final_version = versions[storage.id];
        if (storage.mutable_state) {
            Value & root = graph.values[storage.root];
            if (root.boundary == BoundaryKind::Internal) {
                root.boundary = BoundaryKind::MutableState;
            }
        }
    }
    if (graph.roots.empty() && !graph.operations.empty()) {
        graph.roots.push_back(graph.operations.back().output);
    }
    GraphImplementation::validate_graph(graph);
    graph.fingerprint = GraphImplementation::fingerprint_graph(graph);
    return graph;
}

ImportedGraph ImportedGraph::import(const ggml_cgraph * cgraph) {
    ImportedGraph result;
    if (cgraph == nullptr) {
        result.graph.errors.emplace_back("null cgraph");
        return result;
    }
    // Scheduler reserve and pre-placement graphs may list unused leaf tensors.
    // Execution plans are defined only by values reachable from executable
    // nodes, which also makes raw and post-split imports canonical.
    ggml_cgraph execution_graph = *cgraph;
    execution_graph.n_leafs = 0;
    result.graph = Graph::import(&execution_graph);
    if (!result.graph.valid()) return result;

    std::unordered_map<const ggml_tensor *, ValueId> ids;
    auto visit = [&](auto && self, const ggml_tensor * tensor) -> void {
        if (tensor == nullptr || ids.count(tensor) != 0) {
            return;
        }
        self(self, tensor->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            self(self, tensor->src[i]);
        }
        ids.emplace(tensor, static_cast<ValueId>(result.value_tensors.size()));
        result.value_tensors.push_back(tensor);
    };
    for (int i = 0; i < cgraph->n_nodes; ++i) visit(visit, cgraph->nodes[i]);

    if (result.value_tensors.size() != result.graph.values.size()) {
        result.graph.errors.emplace_back("runtime binding count does not match normalized values");
        return result;
    }
    result.storage_roots.reserve(result.graph.storages.size());
    for (const Storage & storage : result.graph.storages) {
        if (storage.root >= result.value_tensors.size()) {
            result.graph.errors.emplace_back("runtime storage root has no tensor binding");
            return result;
        }
        result.storage_roots.push_back(result.value_tensors[storage.root]);
    }
    return result;
}

const char * Value::boundary_kind_name(BoundaryKind kind) {
    switch (kind) {
        case BoundaryKind::Internal: return "internal";
        case BoundaryKind::Input: return "input";
        case BoundaryKind::Weight: return "weight";
        case BoundaryKind::MutableState: return "mutable_state";
        case BoundaryKind::Output: return "output";
    }
    return "unknown";
}

std::string Graph::serialize_json(const Graph & graph) {
    std::ostringstream out;
    out << "{\"version\":1,\"fingerprint\":\"" << graph.fingerprint << "\",\"storages\":[";
    for (size_t i = 0; i < graph.storages.size(); ++i) {
        const Storage & storage = graph.storages[i];
        if (i != 0) out << ',';
        out << "{\"id\":" << storage.id << ",\"root\":" << storage.root << ",\"size\":" << storage.size
            << ",\"external\":" << (storage.external ? "true" : "false")
            << ",\"weight\":" << (storage.weight ? "true" : "false")
            << ",\"mutable\":" << (storage.mutable_state ? "true" : "false")
            << ",\"version\":" << storage.final_version << '}';
    }
    out << "],\"values\":[";
    for (size_t i = 0; i < graph.values.size(); ++i) {
        const Value & value = graph.values[i];
        if (i != 0) out << ',';
        out << "{\"id\":" << value.id << ",\"type\":\"" << ggml_type_name(value.type)
            << "\",\"op\":\"" << ggml_op_name(value.op) << "\",\"storage\":" << value.access.storage
            << ",\"version\":" << value.access.version << ",\"offset\":" << value.access.offset
            << ",\"flags\":" << value.flags << ",\"producer\":" << value.producer
            << ",\"view_source\":" << value.view_source
            << ",\"boundary\":\"" << Value::boundary_kind_name(value.boundary) << "\",\"shape\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (d != 0) out << ',';
            out << value.access.shape[d];
        }
        out << "],\"strides\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (d != 0) out << ',';
            out << value.access.strides[d];
        }
        out << "],\"name\":\"" << GraphImplementation::escape_json(value.name) << "\"}";
    }
    out << "],\"operations\":[";
    for (size_t i = 0; i < graph.operations.size(); ++i) {
        const Operation & operation = graph.operations[i];
        if (i != 0) out << ',';
        out << "{\"id\":" << operation.id << ",\"op\":\"" << ggml_op_name(operation.op)
            << "\",\"output\":" << operation.output << ",\"ordinal\":" << operation.original_ordinal
            << ",\"params\":\"" << GraphImplementation::bytes_as_hex(operation.raw_params.data(), operation.raw_params.size())
            << "\",\"inputs\":[";
        for (size_t j = 0; j < operation.inputs.size(); ++j) {
            if (j != 0) out << ',';
            out << operation.inputs[j];
        }
        out << "],\"effects\":[";
        for (size_t j = 0; j < operation.effects.size(); ++j) {
            const Effect & effect = operation.effects[j];
            if (j != 0) out << ',';
            out << "{\"kind\":\"" << (effect.kind == EffectKind::Read ? "read" : "write")
                << "\",\"storage\":" << effect.storage << ",\"before\":" << effect.before_version
                << ",\"after\":" << effect.after_version << ",\"offset\":" << effect.offset
                << ",\"size\":" << effect.size << ",\"exact\":" << (effect.exact ? "true" : "false") << '}';
        }
        out << "]}";
    }
    out << "],\"roots\":[";
    for (size_t i = 0; i < graph.roots.size(); ++i) {
        if (i != 0) out << ',';
        out << graph.roots[i];
    }
    out << "],\"errors\":[";
    for (size_t i = 0; i < graph.errors.size(); ++i) {
        if (i != 0) out << ',';
        out << '\"' << GraphImplementation::escape_json(graph.errors[i]) << '\"';
    }
    out << "]}";
    return out.str();
}

} // namespace ggml::hrx
