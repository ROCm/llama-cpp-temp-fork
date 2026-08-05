#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_cgraph;

namespace ggml::hrx {

struct ImportedGraph;

using StorageId = uint32_t;
using ValueId = uint32_t;
using OperationId = uint32_t;

static constexpr uint32_t kInvalidId = UINT32_MAX;

enum class BoundaryKind : uint8_t {
    Internal,
    Input,
    Weight,
    MutableState,
    Output,
};

enum class EffectKind : uint8_t {
    Read,
    Write,
};

struct AccessPath {
    StorageId storage = kInvalidId;
    uint32_t version = 0;
    size_t offset = 0;
    std::array<int64_t, GGML_MAX_DIMS> shape = {};
    std::array<size_t, GGML_MAX_DIMS> strides = {};
};

struct Storage {
    StorageId id = kInvalidId;
    ValueId root = kInvalidId;
    size_t size = 0;
    bool external = false;
    bool weight = false;
    bool mutable_state = false;
    uint32_t final_version = 0;
};

struct Value {
    ValueId id = kInvalidId;
    enum ggml_type type = GGML_TYPE_COUNT;
    enum ggml_op op = GGML_OP_COUNT;
    int32_t flags = 0;
    AccessPath access;
    BoundaryKind boundary = BoundaryKind::Internal;
    OperationId producer = kInvalidId;
    ValueId view_source = kInvalidId;
    std::string name;

    static const char * boundary_kind_name(BoundaryKind kind);
};

struct Effect {
    EffectKind kind = EffectKind::Read;
    StorageId storage = kInvalidId;
    uint32_t before_version = 0;
    uint32_t after_version = 0;
    size_t offset = 0;
    size_t size = 0;
    bool exact = false;
};

struct Operation {
    OperationId id = kInvalidId;
    enum ggml_op op = GGML_OP_COUNT;
    std::vector<ValueId> inputs;
    ValueId output = kInvalidId;
    std::array<uint8_t, GGML_MAX_OP_PARAMS> raw_params = {};
    std::vector<Effect> effects;
    int original_ordinal = -1;
};

struct Graph {
    std::vector<Storage> storages;
    std::vector<Value> values;
    std::vector<Operation> operations;
    std::vector<ValueId> roots;
    std::vector<std::string> errors;
    std::string fingerprint;

    bool valid() const { return errors.empty(); }

    static Graph import(const ggml_cgraph * graph);
    static Graph deserialize_json(const std::string & json);
    static std::string serialize_json(const Graph & graph);
};

// Runtime tensor identity is deliberately kept out of Graph so that cached
// programs and serialized fixtures cannot retain graph-local pointers.
struct ImportedGraph {
    Graph graph;
    std::vector<const ggml_tensor *> value_tensors;
    std::vector<const ggml_tensor *> storage_roots;

    static ImportedGraph import(const ggml_cgraph * graph);
};

} // namespace ggml::hrx
