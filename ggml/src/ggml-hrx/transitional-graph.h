#pragma once

#include "graph-plan.h"

#include <string>
#include <vector>

namespace ggml::hrx {

// Mutable execution-only copy of graph_plan. The transitional command builder
// creates synthetic scratch values while lowering selected recipes. Keeping
// that state here prevents execution lowering from reaching back into the
// immutable recognition plan. Loom CommandProgram replaces this type.
struct transitional_graph {
    std::vector<graph_storage> storages;
    std::vector<graph_value>   values;
    std::vector<graph_op>      operations;
    std::vector<value_id>      roots;
    std::vector<std::string>   errors;
    std::string                fingerprint = "graph2-transition";

    bool valid() const { return errors.empty(); }

    static transitional_graph from_plan(const graph_plan & plan);
};

// Compatibility names are confined to the transitional lowering seam. New
// recognition and planning code uses the snake_case graph_plan vocabulary.
using StorageId    = storage_id;
using ValueId      = value_id;
using OperationId  = op_id;
using Storage      = graph_storage;
using Value        = graph_value;
using Operation    = graph_op;
using Effect       = graph_effect;
using AccessPath   = access_path;
using BoundaryKind = boundary_kind;
using EffectKind   = effect_kind;
using Graph        = transitional_graph;

static constexpr uint32_t kInvalidId = ID_INVALID;

}  // namespace ggml::hrx
