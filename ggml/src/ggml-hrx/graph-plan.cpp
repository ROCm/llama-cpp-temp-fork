#include "graph-plan.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ggml::hrx {

class graph_plan_impl {
  public:
    static bool is_layout_op(enum ggml_op op) {
        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    }

    static operation_storage_semantics storage_semantics(enum ggml_op op) {
        switch (op) {
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                return operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_ALIAS;
            case GGML_OP_SET_ROWS:
                return operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_MUTATE;
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
            case GGML_OP_RMS_NORM:
            case GGML_OP_ROPE:
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SUM_ROWS:
                return operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_ALLOCATE;
            default:
                return operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_UNKNOWN;
        }
    }

    static const ggml_tensor * storage_root(const ggml_tensor * tensor) {
        while (tensor != nullptr && tensor->view_src != nullptr) {
            tensor = tensor->view_src;
        }
        return tensor;
    }

    static size_t access_span(const access_path & access, enum ggml_type type) {
        const size_t blocks = (static_cast<size_t>(access.shape[0]) + ggml_blck_size(type) - 1) / ggml_blck_size(type);
        size_t       result = blocks * access.strides[0];
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

    static std::string escape_json(const std::string & value) {
        std::ostringstream stream;
        for (unsigned char ch : value) {
            switch (ch) {
                case '\\':
                    stream << "\\\\";
                    break;
                case '"':
                    stream << "\\\"";
                    break;
                case '\n':
                    stream << "\\n";
                    break;
                case '\r':
                    stream << "\\r";
                    break;
                case '\t':
                    stream << "\\t";
                    break;
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
        std::string           result(size * 2, '0');
        for (size_t i = 0; i < size; ++i) {
            result[2 * i]     = digits[data[i] >> 4];
            result[2 * i + 1] = digits[data[i] & 0xf];
        }
        return result;
    }

    static std::string format_payload(const region_payload & payload);

    static void validate_graph(graph_plan & graph) {
        for (size_t i = 0; i < graph.storages_.size(); ++i) {
            const graph_storage & storage = graph.storages_[i];
            if (storage.id != i || storage.root >= graph.values_.size()) {
                graph.diagnostics_.errors.emplace_back("invalid storage identity");
            }
        }
        for (size_t i = 0; i < graph.values_.size(); ++i) {
            const graph_value & value = graph.values_[i];
            if (value.id != i || value.access.storage >= graph.storages_.size()) {
                graph.diagnostics_.errors.emplace_back("invalid value identity or storage");
                continue;
            }
            const size_t          span    = access_span(value.access, value.type);
            const graph_storage & storage = graph.storages_[value.access.storage];
            if (value.access.offset > storage.size || span > storage.size - value.access.offset) {
                graph.diagnostics_.errors.emplace_back("value access exceeds its storage");
            }
            if (value.view_source != ID_INVALID &&
                (value.view_source >= graph.values_.size() ||
                 graph.values_[value.view_source].access.storage != value.access.storage)) {
                graph.diagnostics_.errors.emplace_back("view does not preserve storage identity");
            }
        }
        for (size_t i = 0; i < graph.operations_.size(); ++i) {
            const graph_op & operation = graph.operations_[i];
            if (operation.id != i || operation.output >= graph.values_.size() ||
                graph.values_[operation.output].producer != operation.id) {
                graph.diagnostics_.errors.emplace_back("invalid operation identity or output");
                continue;
            }
            for (value_id input : operation.inputs) {
                if (input >= graph.values_.size() ||
                    (graph.values_[input].producer != ID_INVALID && graph.values_[input].producer >= operation.id)) {
                    graph.diagnostics_.errors.emplace_back("operation input is not topologically available");
                }
            }
            for (const graph_effect & effect : operation.effects) {
                if (effect.storage >= graph.storages_.size() ||
                    effect.before_version > graph.storages_[effect.storage].final_version ||
                    effect.after_version > graph.storages_[effect.storage].final_version) {
                    graph.diagnostics_.errors.emplace_back("invalid operation storage effect");
                }
            }
            if (operation.storage_semantics == operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_ALIAS &&
                graph.values_[operation.output].view_source == ID_INVALID) {
                graph.diagnostics_.errors.emplace_back("alias operation has no source-storage reference");
            }
            if (operation.storage_semantics == operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_MUTATE &&
                std::none_of(operation.effects.begin(), operation.effects.end(), [](const graph_effect & effect) {
                    return effect.kind == effect_kind::EFFECT_KIND_WRITE;
                })) {
                graph.diagnostics_.errors.emplace_back("mutating operation has no write effect");
            }
            if (operation.op == GGML_OP_SET_ROWS) {
                if (operation.inputs.size() != 3 || graph.values_[operation.output].access.storage !=
                                                        graph.values_[operation.inputs[2]].access.storage) {
                    graph.diagnostics_.errors.emplace_back("SET_ROWS destination storage is not legacy operand 2");
                }
            }
        }
    }
};

graph_plan graph_plan::deserialize_json(const std::string & text) {
    graph_plan graph;
    try {
        const nlohmann::json root = nlohmann::json::parse(text);
        if (root.at("version").get<int>() != 1) {
            graph.diagnostics_.errors.emplace_back("unsupported graph fixture version");
            return graph;
        }
        auto parse_type = [](const std::string & name) {
            for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                if (name == ggml_type_name(static_cast<enum ggml_type>(i))) {
                    return static_cast<enum ggml_type>(i);
                }
            }
            return GGML_TYPE_COUNT;
        };
        auto parse_op = [](const std::string & name) {
            for (int i = 0; i < GGML_OP_COUNT; ++i) {
                if (name == ggml_op_name(static_cast<enum ggml_op>(i))) {
                    return static_cast<enum ggml_op>(i);
                }
            }
            return GGML_OP_COUNT;
        };
        auto parse_boundary = [](const std::string & name) {
            for (boundary_kind kind : { boundary_kind::BOUNDARY_KIND_INTERNAL, boundary_kind::BOUNDARY_KIND_INPUT,
                                        boundary_kind::BOUNDARY_KIND_WEIGHT, boundary_kind::BOUNDARY_KIND_MUTABLE_STATE,
                                        boundary_kind::BOUNDARY_KIND_OUTPUT }) {
                if (name == boundary_kind_name(kind)) {
                    return kind;
                }
            }
            return boundary_kind::BOUNDARY_KIND_INTERNAL;
        };
        for (const nlohmann::json & item : root.at("storages")) {
            graph_storage storage;
            storage.id            = item.at("id").get<storage_id>();
            storage.root          = item.at("root").get<value_id>();
            storage.size          = item.at("size").get<size_t>();
            storage.external      = item.at("external").get<bool>();
            storage.weight        = item.at("weight").get<bool>();
            storage.mutable_state = item.at("mutable").get<bool>();
            storage.final_version = item.at("version").get<uint32_t>();
            graph.storages_.push_back(storage);
        }
        for (const nlohmann::json & item : root.at("values")) {
            graph_value value;
            value.id = item.at("id").get<value_id>();
            if (item.contains("source_kind")) {
                const std::string source_kind = item.at("source_kind").get<std::string>();
                value.source.kind =
                    source_kind == "leaf" ?
                        source_tensor_kind::SOURCE_TENSOR_KIND_LEAF :
                        (source_kind == "node" ?
                             source_tensor_kind::SOURCE_TENSOR_KIND_NODE :
                             (source_kind == "node_input" ? source_tensor_kind::SOURCE_TENSOR_KIND_NODE_INPUT :
                                                            source_tensor_kind::SOURCE_TENSOR_KIND_NONE));
                value.source.ordinal = item.at("source_ordinal").get<uint32_t>();
                value.source.slot    = item.value("source_slot", ID_INVALID);
            }
            value.type           = parse_type(item.at("type").get<std::string>());
            value.op             = parse_op(item.at("op").get<std::string>());
            value.access.storage = item.at("storage").get<storage_id>();
            value.access.version = item.at("version").get<uint32_t>();
            value.access.offset  = item.at("offset").get<size_t>();
            value.flags          = item.at("flags").get<int32_t>();
            value.producer       = item.at("producer").get<op_id>();
            value.view_source    = item.at("view_source").get<value_id>();
            value.boundary       = parse_boundary(item.at("boundary").get<std::string>());
            value.name           = item.at("name").get<std::string>();
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                value.access.shape[i]   = item.at("shape").at(i).get<int64_t>();
                value.access.strides[i] = item.at("strides").at(i).get<size_t>();
            }
            graph.values_.push_back(std::move(value));
        }
        for (const nlohmann::json & item : root.at("operations")) {
            graph_op operation;
            operation.id                = item.at("id").get<op_id>();
            operation.op                = parse_op(item.at("op").get<std::string>());
            operation.storage_semantics = graph_plan_impl::storage_semantics(operation.op);
            operation.output            = item.at("output").get<value_id>();
            operation.source.kind       = source_tensor_kind::SOURCE_TENSOR_KIND_NODE;
            operation.source.ordinal    = item.at("ordinal").get<uint32_t>();
            operation.original_ordinal  = static_cast<int>(operation.source.ordinal);
            operation.inputs            = item.at("inputs").get<std::vector<value_id>>();
            const std::string params    = item.at("params").get<std::string>();
            if (params.size() != operation.raw_params.size() * 2) {
                throw std::runtime_error("invalid op parameter encoding");
            }
            auto nibble = [](char ch) -> uint8_t {
                if (ch >= '0' && ch <= '9') {
                    return ch - '0';
                }
                if (ch >= 'a' && ch <= 'f') {
                    return ch - 'a' + 10;
                }
                throw std::runtime_error("invalid hexadecimal digit");
            };
            for (size_t i = 0; i < operation.raw_params.size(); ++i) {
                operation.raw_params[i] =
                    static_cast<uint8_t>((nibble(params[2 * i]) << 4) | nibble(params[2 * i + 1]));
            }
            for (const nlohmann::json & effect_item : item.at("effects")) {
                graph_effect effect;
                effect.kind    = effect_item.at("kind").get<std::string>() == "read" ? effect_kind::EFFECT_KIND_READ :
                                                                                       effect_kind::EFFECT_KIND_WRITE;
                effect.storage = effect_item.at("storage").get<storage_id>();
                effect.before_version = effect_item.at("before").get<uint32_t>();
                effect.after_version  = effect_item.at("after").get<uint32_t>();
                effect.offset         = effect_item.at("offset").get<size_t>();
                effect.size           = effect_item.at("size").get<size_t>();
                effect.exact          = effect_item.at("exact").get<bool>();
                operation.effects.push_back(effect);
            }
            graph.operations_.push_back(std::move(operation));
        }
        graph.roots_              = root.at("roots").get<std::vector<value_id>>();
        graph.diagnostics_.errors = root.at("errors").get<std::vector<std::string>>();
        graph_plan_impl::validate_graph(graph);
        graph.build_index();
        graph.initialize_atoms();
    } catch (const std::exception & error) {
        graph.diagnostics_.errors.emplace_back(std::string("invalid graph fixture: ") + error.what());
    }
    return graph;
}

std::string graph_plan_impl::format_payload(const region_payload & payload) {
    std::ostringstream out;
    if (const auto * value = std::get_if<atom_region>(&payload)) {
        out << "operation=" << value->operation;
    } else if (const auto * value = std::get_if<embedding_region>(&payload)) {
        out << "token_ids=" << value->token_ids << " weight=" << value->weight << " hidden=" << value->hidden;
    } else if (const auto * value = std::get_if<attention_prepare_region>(&payload)) {
        out << "hidden=" << value->hidden << " scale=" << value->scale << " prepared=" << value->prepared;
    } else if (const auto * value = std::get_if<attention_qkv_region>(&payload)) {
        out << "prepared=" << value->prepared << " query=" << value->query << " key_cache=" << value->key_cache
            << " value_cache=" << value->value_cache << " projections=" << value->query_projection << ','
            << value->key_projection << ',' << value->value_projection;
    } else if (const auto * value = std::get_if<attention_region>(&payload)) {
        out << "flash=" << value->flash << " query=" << value->query << " key_cache=" << value->key_cache
            << " value_cache=" << value->value_cache << " mask=" << value->mask << " result=" << value->result;
    } else if (const auto * value = std::get_if<attention_output_region>(&payload)) {
        out << "attention_result=" << value->attention_result << " residual_hidden=" << value->residual_hidden
            << " prepared_ffn=" << value->prepared_ffn << " selection_ids=" << value->selection_ids;
    } else if (const auto * value = std::get_if<router_selection_region>(&payload)) {
        out << "prepared=" << value->prepared << " route_ids=" << value->route_ids
            << " route_weights=" << value->route_weights;
    } else if (const auto * value = std::get_if<expert_gate_up_region>(&payload)) {
        out << "prepared=" << value->prepared << " route_ids=" << value->route_ids
            << " activation=" << value->activation;
    } else if (const auto * value = std::get_if<expert_down_region>(&payload)) {
        out << "activation=" << value->activation << " route_weights=" << value->route_weights
            << " hidden_output=" << value->hidden_output << " route_count=" << value->route_count;
    } else if (const auto * value = std::get_if<endpoint_region>(&payload)) {
        out << "hidden=" << value->hidden << " logits=" << value->logits;
    } else if (const auto * value = std::get_if<fusion_region>(&payload)) {
        out << "fusion_kind=" << value->kind;
    }
    return out.str();
}

graph_plan graph_plan::import(const ggml_cgraph * cgraph) {
    graph_plan graph;
    if (cgraph == nullptr) {
        graph.diagnostics_.errors.emplace_back("null cgraph");
        return graph;
    }

    std::vector<const ggml_tensor *>                  ordered;
    std::unordered_map<const ggml_tensor *, value_id> ids;
    auto                                              visit = [&](auto && self, const ggml_tensor * tensor) -> void {
        if (tensor == nullptr || ids.count(tensor) != 0) {
            return;
        }
        self(self, tensor->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            self(self, tensor->src[i]);
        }
        ids.emplace(tensor, static_cast<value_id>(ordered.size()));
        ordered.push_back(tensor);
    };
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        visit(visit, cgraph->leafs[i]);
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        visit(visit, cgraph->nodes[i]);
    }

    std::unordered_map<const ggml_tensor *, storage_id> storage_ids;
    for (const ggml_tensor * tensor : ordered) {
        const ggml_tensor * root = graph_plan_impl::storage_root(tensor);
        if (root == nullptr || storage_ids.count(root) != 0) {
            continue;
        }
        graph_storage storage;
        storage.id       = static_cast<storage_id>(graph.storages_.size());
        storage.root     = ids.at(root);
        storage.size     = ggml_nbytes(root);
        storage.external = root->op == GGML_OP_NONE;
        storage.weight   = graph_plan_impl::buffer_is_weight(root);
        storage_ids.emplace(root, storage.id);
        graph.storages_.push_back(storage);
    }

    graph.values_.resize(ordered.size());
    std::vector<uint32_t>                                        versions(graph.storages_.size(), 0);
    std::unordered_map<const ggml_tensor *, int>                 node_ordinals;
    std::unordered_map<const ggml_tensor *, int>                 leaf_ordinals;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> input_slots;
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        leaf_ordinals.emplace(cgraph->leafs[i], i);
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        node_ordinals.emplace(cgraph->nodes[i], i);
        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
            if (cgraph->nodes[i]->src[slot] != nullptr) {
                input_slots.try_emplace(cgraph->nodes[i]->src[slot], i, slot);
            }
        }
    }

    for (const ggml_tensor * tensor : ordered) {
        graph_value value;
        value.id          = ids.at(tensor);
        value.type        = tensor->type;
        value.op          = tensor->op;
        value.flags       = tensor->flags;
        value.name        = tensor->name;
        auto leaf_ordinal = leaf_ordinals.find(tensor);
        if (leaf_ordinal != leaf_ordinals.end()) {
            value.source.kind    = source_tensor_kind::SOURCE_TENSOR_KIND_LEAF;
            value.source.ordinal = static_cast<uint32_t>(leaf_ordinal->second);
        } else {
            const auto input_slot = input_slots.find(tensor);
            if (input_slot != input_slots.end()) {
                value.source.kind    = source_tensor_kind::SOURCE_TENSOR_KIND_NODE_INPUT;
                value.source.ordinal = static_cast<uint32_t>(input_slot->second.first);
                value.source.slot    = static_cast<uint32_t>(input_slot->second.second);
            }
        }
        value.view_source        = tensor->view_src != nullptr ? ids.at(tensor->view_src) : ID_INVALID;
        const ggml_tensor * root = graph_plan_impl::storage_root(tensor);
        value.access.storage     = storage_ids.at(root);
        value.access.version     = versions[value.access.storage];
        value.access.offset      = tensor->view_src != nullptr ? tensor->view_offs : 0;
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), value.access.shape.begin());
        std::copy(std::begin(tensor->nb), std::end(tensor->nb), value.access.strides.begin());
        if ((tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            value.boundary = boundary_kind::BOUNDARY_KIND_OUTPUT;
            graph.roots_.push_back(value.id);
        } else if ((tensor->flags & GGML_TENSOR_FLAG_INPUT) != 0) {
            value.boundary = boundary_kind::BOUNDARY_KIND_INPUT;
        } else if (graph.storages_[value.access.storage].weight) {
            value.boundary = boundary_kind::BOUNDARY_KIND_WEIGHT;
        }

        auto ordinal = node_ordinals.find(tensor);
        if (ordinal != node_ordinals.end()) {
            graph_op operation;
            operation.id                = static_cast<op_id>(graph.operations_.size());
            operation.op                = tensor->op;
            operation.storage_semantics = graph_plan_impl::storage_semantics(operation.op);
            if (operation.storage_semantics == operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_UNKNOWN) {
                graph.diagnostics_.warnings.push_back(std::string("operation semantics are unknown for ") +
                                                      ggml_op_name(operation.op));
            }
            operation.output           = value.id;
            operation.source.kind      = source_tensor_kind::SOURCE_TENSOR_KIND_NODE;
            operation.source.ordinal   = static_cast<uint32_t>(ordinal->second);
            operation.original_ordinal = ordinal->second;
            value.source               = operation.source;
            std::memcpy(operation.raw_params.data(), tensor->op_params, operation.raw_params.size());
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (tensor->src[i] != nullptr) {
                    operation.inputs.push_back(ids.at(tensor->src[i]));
                    const graph_value & input = graph.values_[ids.at(tensor->src[i])];
                    graph_effect        read;
                    read.kind           = effect_kind::EFFECT_KIND_READ;
                    read.storage        = input.access.storage;
                    read.before_version = versions[read.storage];
                    read.after_version  = read.before_version;
                    read.offset         = input.access.offset;
                    read.size           = graph_plan_impl::access_span(input.access, input.type);
                    read.exact          = true;
                    operation.effects.push_back(read);
                }
            }

            const bool aliases_storage = tensor->view_src != nullptr && !graph_plan_impl::is_layout_op(tensor->op);
            if (aliases_storage) {
                graph_effect write;
                write.kind           = effect_kind::EFFECT_KIND_WRITE;
                write.storage        = value.access.storage;
                write.before_version = versions[write.storage];
                write.after_version  = ++versions[write.storage];
                write.offset         = value.access.offset;
                write.size           = tensor->op == GGML_OP_SET_ROWS ? graph.storages_[write.storage].size :
                                                                        graph_plan_impl::access_span(value.access, value.type);
                write.exact          = tensor->op != GGML_OP_SET_ROWS;
                operation.effects.push_back(write);
                value.access.version    = write.after_version;
                graph_storage & storage = graph.storages_[write.storage];
                storage.mutable_state |= storage.external;
                storage.final_version = write.after_version;
                if (storage.mutable_state && value.boundary != boundary_kind::BOUNDARY_KIND_OUTPUT) {
                    value.boundary = boundary_kind::BOUNDARY_KIND_MUTABLE_STATE;
                }
            }
            value.producer = operation.id;
            graph.operations_.push_back(std::move(operation));
        }
        graph.values_[value.id] = std::move(value);
    }

    for (graph_storage & storage : graph.storages_) {
        storage.final_version = versions[storage.id];
        if (storage.mutable_state) {
            graph_value & root = graph.values_[storage.root];
            if (root.boundary == boundary_kind::BOUNDARY_KIND_INTERNAL) {
                root.boundary = boundary_kind::BOUNDARY_KIND_MUTABLE_STATE;
            }
        }
    }
    if (graph.roots_.empty() && !graph.operations_.empty()) {
        graph.roots_.push_back(graph.operations_.back().output);
    }
    graph_plan_impl::validate_graph(graph);
    graph.build_index();
    graph.initialize_atoms();
    return graph;
}

graph_import graph_import::import(const ggml_cgraph * cgraph) {
    graph_import result;
    if (cgraph == nullptr) {
        result.plan.diagnostics_.errors.emplace_back("null cgraph");
        return result;
    }
    // Scheduler reserve and pre-placement graphs may list unused leaf tensors.
    // Execution plans are defined only by values reachable from executable
    // nodes, which also makes raw and post-split imports canonical.
    ggml_cgraph execution_graph = *cgraph;
    execution_graph.n_leafs     = 0;
    result.plan                 = graph_plan::import(&execution_graph);
    if (!result.plan.valid()) {
        return result;
    }

    std::unordered_map<const ggml_tensor *, value_id> ids;
    auto                                              visit = [&](auto && self, const ggml_tensor * tensor) -> void {
        if (tensor == nullptr || ids.count(tensor) != 0) {
            return;
        }
        self(self, tensor->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            self(self, tensor->src[i]);
        }
        ids.emplace(tensor, static_cast<value_id>(result.value_tensors.size()));
        result.value_tensors.push_back(tensor);
    };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        visit(visit, cgraph->nodes[i]);
    }

    if (result.value_tensors.size() != result.plan.values_.size()) {
        result.plan.diagnostics_.errors.emplace_back("runtime binding count does not match normalized values");
        return result;
    }
    result.storage_roots.reserve(result.plan.storages_.size());
    for (const graph_storage & storage : result.plan.storages_) {
        if (storage.root >= result.value_tensors.size()) {
            result.plan.diagnostics_.errors.emplace_back("runtime storage root has no tensor binding");
            return result;
        }
        result.storage_roots.push_back(result.value_tensors[storage.root]);
    }
    return result;
}

const char * boundary_kind_name(boundary_kind kind) {
    switch (kind) {
        case boundary_kind::BOUNDARY_KIND_INTERNAL:
            return "internal";
        case boundary_kind::BOUNDARY_KIND_INPUT:
            return "input";
        case boundary_kind::BOUNDARY_KIND_WEIGHT:
            return "weight";
        case boundary_kind::BOUNDARY_KIND_MUTABLE_STATE:
            return "mutable_state";
        case boundary_kind::BOUNDARY_KIND_OUTPUT:
            return "output";
    }
    return "unknown";
}

const char * region_kind_name(region_kind kind) {
    switch (kind) {
        case region_kind::REGION_KIND_ATOM:
            return "atom";
        case region_kind::REGION_KIND_EMBEDDING:
            return "embedding";
        case region_kind::REGION_KIND_ATTENTION_PREPARE:
            return "attention_prepare";
        case region_kind::REGION_KIND_ATTENTION_QKV:
            return "attention_qkv";
        case region_kind::REGION_KIND_ATTENTION:
            return "attention";
        case region_kind::REGION_KIND_ATTENTION_OUTPUT:
            return "attention_output";
        case region_kind::REGION_KIND_ROUTER_SELECTION:
            return "router_selection";
        case region_kind::REGION_KIND_EXPERT_GATE_UP:
            return "expert_gate_up";
        case region_kind::REGION_KIND_EXPERT_DOWN:
            return "expert_down";
        case region_kind::REGION_KIND_ENDPOINT:
            return "endpoint";
        case region_kind::REGION_KIND_FUSION:
            return "fusion";
    }
    return "unknown";
}

const char * recipe_parameter_kind_name(recipe_parameter_kind kind) {
    switch (kind) {
        case recipe_parameter_kind::RECIPE_PARAMETER_KIND_GRAPH_SHAPING:
            return "graph_shaping";
        case recipe_parameter_kind::RECIPE_PARAMETER_KIND_COMPILE_SPECIALIZATION:
            return "compile_specialization";
        case recipe_parameter_kind::RECIPE_PARAMETER_KIND_COMMAND_RECORDING:
            return "command_recording";
        case recipe_parameter_kind::RECIPE_PARAMETER_KIND_LAUNCH_SCALAR:
            return "launch_scalar";
    }
    return "unknown";
}

void graph_plan::add_error(std::string error) {
    diagnostics_.errors.push_back(std::move(error));
}

void graph_plan::build_index() {
    consumers_.assign(values_.size(), {});
    predecessors_.assign(operations_.size(), {});
    successors_.assign(operations_.size(), {});
    for (auto & operations : operations_by_kind_) {
        operations.clear();
    }
    storage_writers_.clear();

    auto add_edge = [&](op_id from, op_id to) {
        if (from == to || from >= successors_.size() || to >= predecessors_.size()) {
            return;
        }
        successors_[from].push_back(to);
        predecessors_[to].push_back(from);
    };
    for (const graph_op & operation : operations_) {
        if (operation.op < GGML_OP_COUNT) {
            operations_by_kind_[operation.op].push_back(operation.id);
        }
        for (value_id input : operation.inputs) {
            if (input >= values_.size()) {
                continue;
            }
            consumers_[input].push_back(operation.id);
            if (values_[input].producer != ID_INVALID) {
                add_edge(values_[input].producer, operation.id);
            }
        }
        for (const graph_effect & effect : operation.effects) {
            if (effect.kind == effect_kind::EFFECT_KIND_WRITE &&
                !storage_writers_.emplace(std::make_pair(effect.storage, effect.after_version), operation.id).second) {
                add_error("multiple operations write the same storage version");
            }
        }
    }
    for (const graph_op & operation : operations_) {
        for (const graph_effect & effect : operation.effects) {
            if (effect.before_version == 0) {
                continue;
            }
            const auto writer = storage_writers_.find({ effect.storage, effect.before_version });
            if (writer != storage_writers_.end()) {
                add_edge(writer->second, operation.id);
            }
        }
    }
    auto canonicalize = [](std::vector<op_id> & ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    };
    for (auto & ids : consumers_) {
        canonicalize(ids);
    }
    for (auto & ids : predecessors_) {
        canonicalize(ids);
    }
    for (auto & ids : successors_) {
        canonicalize(ids);
    }
    for (auto & ids : operations_by_kind_) {
        std::reverse(ids.begin(), ids.end());
    }
}

void graph_plan::initialize_atoms() {
    regions_.clear();
    active_regions_.clear();
    source_op_owner_.assign(operations_.size(), ID_INVALID);
    regions_.reserve(operations_.size());
    active_regions_.reserve(operations_.size());
    for (const graph_op & operation : operations_) {
        graph_region region;
        region.id                      = static_cast<region_id>(regions_.size());
        region.kind                    = region_kind::REGION_KIND_ATOM;
        region.payload                 = atom_region{ operation.id };
        region.source_ops              = { operation.id };
        region.boundary                = calculate_boundary(region.source_ops);
        region.effects                 = operation.effects;
        source_op_owner_[operation.id] = region.id;
        active_regions_.push_back(region.id);
        regions_.push_back(std::move(region));
    }
}

const std::vector<op_id> & graph_plan::consumers(value_id value) const {
    static const std::vector<op_id> empty;
    return value < consumers_.size() ? consumers_[value] : empty;
}

const std::vector<op_id> & graph_plan::predecessors(op_id operation) const {
    static const std::vector<op_id> empty;
    return operation < predecessors_.size() ? predecessors_[operation] : empty;
}

const std::vector<op_id> & graph_plan::successors(op_id operation) const {
    static const std::vector<op_id> empty;
    return operation < successors_.size() ? successors_[operation] : empty;
}

const std::vector<op_id> & graph_plan::operations_by_kind(enum ggml_op kind) const {
    static const std::vector<op_id> empty;
    return kind < GGML_OP_COUNT ? operations_by_kind_[kind] : empty;
}

op_id graph_plan::storage_writer(storage_id storage, uint32_t version) const {
    const auto writer = storage_writers_.find({ storage, version });
    return writer == storage_writers_.end() ? ID_INVALID : writer->second;
}

region_boundary graph_plan::calculate_boundary(const std::vector<op_id> & operations) const {
    std::set<op_id>          covered(operations.begin(), operations.end());
    std::set<value_id>       inputs;
    std::set<value_id>       outputs;
    const std::set<value_id> roots(roots_.begin(), roots_.end());
    for (op_id operation_id : operations) {
        if (operation_id >= operations_.size()) {
            continue;
        }
        const graph_op & operation = operations_[operation_id];
        for (value_id input : operation.inputs) {
            if (input >= values_.size()) {
                continue;
            }
            const op_id producer = values_[input].producer;
            if (producer == ID_INVALID || covered.count(producer) == 0) {
                inputs.insert(input);
            }
        }
        bool escapes = roots.count(operation.output) != 0;
        for (op_id consumer : consumers(operation.output)) {
            escapes = escapes || covered.count(consumer) == 0;
        }
        for (const graph_effect & effect : operation.effects) {
            escapes = escapes || effect.kind == effect_kind::EFFECT_KIND_WRITE;
        }
        if (escapes) {
            outputs.insert(operation.output);
        }
    }
    return {
        { inputs.begin(),  inputs.end()  },
        { outputs.begin(), outputs.end() }
    };
}

bool graph_plan::verify(bool expensive) const {
    if (!valid() || source_op_owner_.size() != operations_.size()) {
        return false;
    }
    std::vector<uint32_t> ownership(operations_.size(), 0);
    for (region_id active : active_regions_) {
        if (active >= regions_.size() || regions_[active].parent != ID_INVALID) {
            return false;
        }
        for (op_id operation : regions_[active].source_ops) {
            if (operation >= ownership.size()) {
                return false;
            }
            ++ownership[operation];
        }
    }
    if (std::any_of(ownership.begin(), ownership.end(), [](uint32_t count) { return count != 1; })) {
        return false;
    }
    if (!expensive) {
        return true;
    }
    for (const graph_region & region : regions_) {
        std::vector<op_id> sorted = region.source_ops;
        std::sort(sorted.begin(), sorted.end());
        std::set<std::tuple<effect_kind, storage_id, uint32_t, uint32_t, size_t, size_t, bool>> expected_effects;
        std::set<std::tuple<effect_kind, storage_id, uint32_t, uint32_t, size_t, size_t, bool>> actual_effects;
        for (op_id operation : region.source_ops) {
            for (const graph_effect & effect : operations_[operation].effects) {
                expected_effects.emplace(effect.kind, effect.storage, effect.before_version, effect.after_version,
                                         effect.offset, effect.size, effect.exact);
            }
        }
        for (const graph_effect & effect : region.effects) {
            actual_effects.emplace(effect.kind, effect.storage, effect.before_version, effect.after_version,
                                   effect.offset, effect.size, effect.exact);
        }
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() ||
            calculate_boundary(region.source_ops).inputs != region.boundary.inputs ||
            calculate_boundary(region.source_ops).outputs != region.boundary.outputs ||
            expected_effects != actual_effects) {
            return false;
        }
    }
    std::vector<uint32_t> group_membership(regions_.size(), 0);
    for (size_t i = 0; i < groups_.size(); ++i) {
        const semantic_group & group = groups_[i];
        if (group.id != i || group.members.empty() ||
            group.previous != (i == 0 ? ID_INVALID : static_cast<group_id>(i - 1)) ||
            group.next != (i + 1 == groups_.size() ? ID_INVALID : static_cast<group_id>(i + 1))) {
            return false;
        }
        for (region_id member : group.members) {
            if (member >= regions_.size() || regions_[member].kind == region_kind::REGION_KIND_ATOM ||
                ++group_membership[member] != 1) {
                return false;
            }
        }
    }
    return true;
}

std::string graph_plan::format() const {
    std::ostringstream out;
    out << "graph-plan phase=" << static_cast<unsigned>(phase_) << " operations=" << operations_.size()
        << " values=" << values_.size() << " storages=" << storages_.size() << " regions=" << regions_.size()
        << " active=" << active_regions_.size() << " groups=" << groups_.size()
        << " probes=" << diagnostics_.matcher_probes << '\n';
    for (region_id id : active_regions_) {
        const graph_region & region = regions_[id];
        out << "region " << region.id << " kind=" << region_kind_name(region.kind)
            << " operations=" << region.source_ops.size() << " recipe=" << region.selected_recipe
            << " inputs=" << region.boundary.inputs.size() << " outputs=" << region.boundary.outputs.size()
            << " effects=" << region.effects.size() << " capture={" << graph_plan_impl::format_payload(region.payload)
            << "}\n";
    }
    for (const semantic_group & group : groups_) {
        out << "group " << group.id << " kind=routed_block members=";
        for (region_id member : group.members) {
            out << member << ',';
        }
        out << " previous=" << group.previous << " next=" << group.next << '\n';
    }
    for (const auto & fact : facts_) {
        out << "fact " << fact.first << '=' << fact.second << '\n';
    }
    for (const selected_recipe & recipe : selected_recipes_) {
        out << "recipe " << recipe.id << " capability=" << schedule_capability_name(recipe.capability) << " regions=";
        for (region_id region : recipe.regions) {
            out << region << ',';
        }
        out << '\n';
        for (const recipe_parameter & parameter : recipe.parameters) {
            out << "  parameter " << parameter.name << '=' << parameter.value
                << " kind=" << recipe_parameter_kind_name(parameter.kind) << '\n';
        }
    }
    for (const std::string & warning : diagnostics_.warnings) {
        out << "warning=" << warning << '\n';
    }
    for (const std::string & error : diagnostics_.errors) {
        out << "error=" << error << '\n';
    }
    return out.str();
}

std::string graph_plan::serialize_json() const {
    std::ostringstream out;
    out << "{\"version\":1,\"storages\":[";
    for (size_t i = 0; i < storages_.size(); ++i) {
        const graph_storage & storage = storages_[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"id\":" << storage.id << ",\"root\":" << storage.root << ",\"size\":" << storage.size
            << ",\"external\":" << (storage.external ? "true" : "false")
            << ",\"weight\":" << (storage.weight ? "true" : "false")
            << ",\"mutable\":" << (storage.mutable_state ? "true" : "false") << ",\"version\":" << storage.final_version
            << '}';
    }
    out << "],\"values\":[";
    for (size_t i = 0; i < values_.size(); ++i) {
        const graph_value & value = values_[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"id\":" << value.id << ",\"type\":\"" << ggml_type_name(value.type) << "\",\"op\":\""
            << ggml_op_name(value.op) << "\",\"storage\":" << value.access.storage << ",\"source_kind\":\""
            << (value.source.kind == source_tensor_kind::SOURCE_TENSOR_KIND_LEAF ?
                    "leaf" :
                    (value.source.kind == source_tensor_kind::SOURCE_TENSOR_KIND_NODE ?
                         "node" :
                         (value.source.kind == source_tensor_kind::SOURCE_TENSOR_KIND_NODE_INPUT ? "node_input" :
                                                                                                   "none")))
            << "\",\"source_ordinal\":" << value.source.ordinal << ",\"source_slot\":" << value.source.slot
            << ",\"version\":" << value.access.version << ",\"offset\":" << value.access.offset
            << ",\"flags\":" << value.flags << ",\"producer\":" << value.producer
            << ",\"view_source\":" << value.view_source << ",\"boundary\":\"" << boundary_kind_name(value.boundary)
            << "\",\"shape\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (d != 0) {
                out << ',';
            }
            out << value.access.shape[d];
        }
        out << "],\"strides\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (d != 0) {
                out << ',';
            }
            out << value.access.strides[d];
        }
        out << "],\"name\":\"" << graph_plan_impl::escape_json(value.name) << "\"}";
    }
    out << "],\"operations\":[";
    for (size_t i = 0; i < operations_.size(); ++i) {
        const graph_op & operation = operations_[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"id\":" << operation.id << ",\"op\":\"" << ggml_op_name(operation.op)
            << "\",\"output\":" << operation.output << ",\"ordinal\":" << operation.source.ordinal << ",\"params\":\""
            << graph_plan_impl::bytes_as_hex(operation.raw_params.data(), operation.raw_params.size())
            << "\",\"inputs\":[";
        for (size_t j = 0; j < operation.inputs.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << operation.inputs[j];
        }
        out << "],\"effects\":[";
        for (size_t j = 0; j < operation.effects.size(); ++j) {
            const graph_effect & effect = operation.effects[j];
            if (j != 0) {
                out << ',';
            }
            out << "{\"kind\":\"" << (effect.kind == effect_kind::EFFECT_KIND_READ ? "read" : "write")
                << "\",\"storage\":" << effect.storage << ",\"before\":" << effect.before_version
                << ",\"after\":" << effect.after_version << ",\"offset\":" << effect.offset
                << ",\"size\":" << effect.size << ",\"exact\":" << (effect.exact ? "true" : "false") << '}';
        }
        out << "]}";
    }
    out << "],\"roots\":[";
    for (size_t i = 0; i < roots_.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << roots_[i];
    }
    out << "],\"plan\":{\"phase\":" << static_cast<unsigned>(phase_) << ",\"regions\":[";
    for (size_t i = 0; i < regions_.size(); ++i) {
        const graph_region & region = regions_[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"id\":" << region.id << ",\"kind\":\"" << region_kind_name(region.kind)
            << "\",\"parent\":" << region.parent << ",\"recipe\":" << region.selected_recipe << ",\"capture\":\""
            << graph_plan_impl::escape_json(graph_plan_impl::format_payload(region.payload)) << "\",\"operations\":[";
        for (size_t j = 0; j < region.source_ops.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << region.source_ops[j];
        }
        out << "]}";
    }
    out << "],\"active_regions\":[";
    for (size_t i = 0; i < active_regions_.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << active_regions_[i];
    }
    out << "],\"groups\":[";
    for (size_t i = 0; i < groups_.size(); ++i) {
        const semantic_group & group = groups_[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"id\":" << group.id << ",\"previous\":" << group.previous << ",\"next\":" << group.next
            << ",\"members\":[";
        for (size_t j = 0; j < group.members.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << group.members[j];
        }
        out << "],\"facts\":{";
        size_t group_fact_ordinal = 0;
        for (const auto & fact : group.facts) {
            if (group_fact_ordinal++ != 0) {
                out << ',';
            }
            out << '\"' << graph_plan_impl::escape_json(fact.first) << "\":" << fact.second;
        }
        out << "}}";
    }
    out << "],\"facts\":{";
    size_t fact_ordinal = 0;
    for (const auto & fact : facts_) {
        if (fact_ordinal++ != 0) {
            out << ',';
        }
        out << '"' << graph_plan_impl::escape_json(fact.first) << "\":" << fact.second;
    }
    out << "},\"selected_recipes\":[";
    for (size_t i = 0; i < selected_recipes_.size(); ++i) {
        const selected_recipe & recipe = selected_recipes_[i];
        if (i != 0) {
            out << ',';
        }
        out << "{\"id\":" << recipe.id << ",\"capability\":" << static_cast<unsigned>(recipe.capability)
            << ",\"capability_name\":\"" << schedule_capability_name(recipe.capability) << "\",\"regions\":[";
        for (size_t j = 0; j < recipe.regions.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << recipe.regions[j];
        }
        out << "],\"facts\":{";
        size_t recipe_fact_ordinal = 0;
        for (const auto & fact : recipe.facts) {
            if (recipe_fact_ordinal++ != 0) {
                out << ',';
            }
            out << '\"' << graph_plan_impl::escape_json(fact.first) << "\":" << fact.second;
        }
        out << "},\"parameters\":[";
        for (size_t j = 0; j < recipe.parameters.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            const recipe_parameter & parameter = recipe.parameters[j];
            out << "{\"name\":\"" << graph_plan_impl::escape_json(parameter.name) << "\",\"value\":" << parameter.value
                << ",\"kind\":\"" << recipe_parameter_kind_name(parameter.kind) << "\"}";
        }
        out << "]}";
    }
    out << "]},\"errors\":[";
    for (size_t i = 0; i < diagnostics_.errors.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '\"' << graph_plan_impl::escape_json(diagnostics_.errors[i]) << '\"';
    }
    out << "]}";
    return out.str();
}

std::string graph_plan::dot() const {
    std::ostringstream out;
    out << "digraph graph_plan {\n";
    for (region_id id : active_regions_) {
        const graph_region & region = regions_[id];
        out << "  r" << id << " [label=\"" << region_kind_name(region.kind) << "\\nops=" << region.source_ops.size()
            << "\"];\n";
    }
    std::vector<region_id> owner(operations_.size(), ID_INVALID);
    for (region_id id : active_regions_) {
        for (op_id operation : regions_[id].source_ops) {
            owner[operation] = id;
        }
    }
    std::set<std::pair<region_id, region_id>> edges;
    for (op_id operation = 0; operation < successors_.size(); ++operation) {
        for (op_id successor : successors_[operation]) {
            if (owner[operation] != owner[successor]) {
                edges.emplace(owner[operation], owner[successor]);
            }
        }
    }
    for (const auto & edge : edges) {
        out << "  r" << edge.first << " -> r" << edge.second << ";\n";
    }
    out << "}\n";
    return out.str();
}

}  // namespace ggml::hrx
