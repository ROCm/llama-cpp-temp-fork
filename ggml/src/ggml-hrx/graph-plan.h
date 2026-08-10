#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

struct ggml_cgraph;

namespace ggml::hrx {

using storage_id = uint32_t;
using value_id   = uint32_t;
using op_id      = uint32_t;
using region_id  = uint32_t;
using recipe_id  = uint32_t;
using group_id   = uint32_t;

static constexpr uint32_t ID_INVALID = UINT32_MAX;

enum class plan_phase : uint8_t {
    PLAN_PHASE_IMPORTED,
    PLAN_PHASE_RECOGNIZED,
    PLAN_PHASE_PLANNED,
};

enum class boundary_kind : uint8_t {
    BOUNDARY_KIND_INTERNAL,
    BOUNDARY_KIND_INPUT,
    BOUNDARY_KIND_WEIGHT,
    BOUNDARY_KIND_MUTABLE_STATE,
    BOUNDARY_KIND_OUTPUT,
    Internal     = BOUNDARY_KIND_INTERNAL,
    Input        = BOUNDARY_KIND_INPUT,
    Weight       = BOUNDARY_KIND_WEIGHT,
    MutableState = BOUNDARY_KIND_MUTABLE_STATE,
    Output       = BOUNDARY_KIND_OUTPUT,
};

enum class effect_kind : uint8_t {
    EFFECT_KIND_READ,
    EFFECT_KIND_WRITE,
    Read  = EFFECT_KIND_READ,
    Write = EFFECT_KIND_WRITE,
};

enum class operation_storage_semantics : uint8_t {
    OPERATION_STORAGE_SEMANTICS_ALLOCATE,
    OPERATION_STORAGE_SEMANTICS_ALIAS,
    OPERATION_STORAGE_SEMANTICS_MUTATE,
    OPERATION_STORAGE_SEMANTICS_UNKNOWN,
};

enum class source_tensor_kind : uint8_t {
    SOURCE_TENSOR_KIND_NONE,
    SOURCE_TENSOR_KIND_LEAF,
    SOURCE_TENSOR_KIND_NODE,
    SOURCE_TENSOR_KIND_NODE_INPUT,
};

struct source_tensor_ref {
    source_tensor_kind kind    = source_tensor_kind::SOURCE_TENSOR_KIND_NONE;
    uint32_t           ordinal = ID_INVALID;
    uint32_t           slot    = ID_INVALID;
};

struct access_path {
    storage_id                         storage = ID_INVALID;
    uint32_t                           version = 0;
    size_t                             offset  = 0;
    std::array<int64_t, GGML_MAX_DIMS> shape   = {};
    std::array<size_t, GGML_MAX_DIMS>  strides = {};
};

struct graph_storage {
    storage_id id            = ID_INVALID;
    value_id   root          = ID_INVALID;
    size_t     size          = 0;
    bool       external      = false;
    bool       weight        = false;
    bool       mutable_state = false;
    uint32_t   final_version = 0;
};

struct graph_value {
    value_id          id = ID_INVALID;
    source_tensor_ref source;
    enum ggml_type    type  = GGML_TYPE_COUNT;
    enum ggml_op      op    = GGML_OP_COUNT;
    int32_t           flags = 0;
    access_path       access;
    boundary_kind     boundary    = boundary_kind::BOUNDARY_KIND_INTERNAL;
    op_id             producer    = ID_INVALID;
    value_id          view_source = ID_INVALID;
    std::string       name;
};

struct graph_effect {
    effect_kind kind           = effect_kind::EFFECT_KIND_READ;
    storage_id  storage        = ID_INVALID;
    uint32_t    before_version = 0;
    uint32_t    after_version  = 0;
    size_t      offset         = 0;
    size_t      size           = 0;
    bool        exact          = false;
};

struct graph_op {
    op_id                       id = ID_INVALID;
    source_tensor_ref           source;
    enum ggml_op                op                = GGML_OP_COUNT;
    operation_storage_semantics storage_semantics = operation_storage_semantics::OPERATION_STORAGE_SEMANTICS_UNKNOWN;
    std::vector<value_id>       inputs;
    value_id                    output                 = ID_INVALID;
    std::array<uint8_t, GGML_MAX_OP_PARAMS> raw_params = {};
    std::vector<graph_effect>               effects;
    int                                     original_ordinal = -1;
};

struct region_boundary {
    std::vector<value_id> inputs;
    std::vector<value_id> outputs;
};

struct atom_region {
    op_id operation = ID_INVALID;
};

struct embedding_region {
    value_id token_ids = ID_INVALID;
    value_id weight    = ID_INVALID;
    value_id hidden    = ID_INVALID;
};

struct attention_prepare_region {
    value_id hidden   = ID_INVALID;
    value_id scale    = ID_INVALID;
    value_id prepared = ID_INVALID;
};

struct attention_qkv_region {
    value_id   prepared         = ID_INVALID;
    value_id   query            = ID_INVALID;
    storage_id key_cache        = ID_INVALID;
    storage_id value_cache      = ID_INVALID;
    op_id      query_projection = ID_INVALID;
    op_id      key_projection   = ID_INVALID;
    op_id      value_projection = ID_INVALID;
};

struct attention_region {
    op_id      flash       = ID_INVALID;
    value_id   query       = ID_INVALID;
    storage_id key_cache   = ID_INVALID;
    storage_id value_cache = ID_INVALID;
    value_id   mask        = ID_INVALID;
    value_id   result      = ID_INVALID;
};

struct attention_output_region {
    value_id attention_result = ID_INVALID;
    value_id residual_hidden  = ID_INVALID;
    value_id prepared_ffn     = ID_INVALID;
    value_id selection_ids    = ID_INVALID;
};

struct router_selection_region {
    value_id prepared      = ID_INVALID;
    value_id route_ids     = ID_INVALID;
    value_id route_weights = ID_INVALID;
};

struct expert_gate_up_region {
    value_id prepared   = ID_INVALID;
    value_id route_ids  = ID_INVALID;
    value_id activation = ID_INVALID;
};

struct expert_down_region {
    value_id activation    = ID_INVALID;
    value_id route_weights = ID_INVALID;
    value_id hidden_output = ID_INVALID;
    uint32_t route_count   = 0;
};

struct endpoint_region {
    value_id hidden = ID_INVALID;
    value_id logits = ID_INVALID;
};

struct fusion_region {
    uint32_t kind = 0;
};

enum class schedule_capability : uint8_t {
    SCHEDULE_CAPABILITY_PRIMITIVE,
    SCHEDULE_CAPABILITY_EMBEDDING,
    SCHEDULE_CAPABILITY_ATTENTION_PREPARE,
    SCHEDULE_CAPABILITY_ATTENTION_QKV,
    SCHEDULE_CAPABILITY_ATTENTION,
    SCHEDULE_CAPABILITY_ATTENTION_OUTPUT,
    SCHEDULE_CAPABILITY_ROUTER_SELECTION,
    SCHEDULE_CAPABILITY_EXPERT_GATE_UP,
    SCHEDULE_CAPABILITY_EXPERT_DOWN,
    SCHEDULE_CAPABILITY_EXPERT_DOWN_NEXT_PREPARE,
    SCHEDULE_CAPABILITY_ENDPOINT,
};

enum class recipe_parameter_kind : uint8_t {
    RECIPE_PARAMETER_KIND_GRAPH_SHAPING,
    RECIPE_PARAMETER_KIND_COMPILE_SPECIALIZATION,
    RECIPE_PARAMETER_KIND_COMMAND_RECORDING,
    RECIPE_PARAMETER_KIND_LAUNCH_SCALAR,
};

struct recipe_parameter {
    std::string           name;
    int64_t               value = 0;
    recipe_parameter_kind kind  = recipe_parameter_kind::RECIPE_PARAMETER_KIND_GRAPH_SHAPING;
};

struct selected_recipe {
    recipe_id                      id         = ID_INVALID;
    schedule_capability            capability = schedule_capability::SCHEDULE_CAPABILITY_PRIMITIVE;
    std::vector<region_id>         regions;
    std::map<std::string, int64_t> facts;
    std::vector<recipe_parameter>  parameters;
};

using region_payload = std::variant<atom_region,
                                    embedding_region,
                                    attention_prepare_region,
                                    attention_qkv_region,
                                    attention_region,
                                    attention_output_region,
                                    router_selection_region,
                                    expert_gate_up_region,
                                    expert_down_region,
                                    endpoint_region,
                                    fusion_region>;

enum class region_kind : uint8_t {
    REGION_KIND_ATOM,
    REGION_KIND_EMBEDDING,
    REGION_KIND_ATTENTION_PREPARE,
    REGION_KIND_ATTENTION_QKV,
    REGION_KIND_ATTENTION,
    REGION_KIND_ATTENTION_OUTPUT,
    REGION_KIND_ROUTER_SELECTION,
    REGION_KIND_EXPERT_GATE_UP,
    REGION_KIND_EXPERT_DOWN,
    REGION_KIND_ENDPOINT,
    REGION_KIND_FUSION,
};

struct graph_region {
    region_id                 id      = ID_INVALID;
    region_kind               kind    = region_kind::REGION_KIND_ATOM;
    region_payload            payload = atom_region{};
    std::vector<region_id>    children;
    std::vector<op_id>        source_ops;
    region_boundary           boundary;
    std::vector<graph_effect> effects;
    region_id                 parent          = ID_INVALID;
    recipe_id                 selected_recipe = ID_INVALID;
};

enum class semantic_group_kind : uint8_t {
    SEMANTIC_GROUP_KIND_ROUTED_BLOCK,
};

struct semantic_group {
    group_id                       id   = ID_INVALID;
    semantic_group_kind            kind = semantic_group_kind::SEMANTIC_GROUP_KIND_ROUTED_BLOCK;
    std::vector<region_id>         members;
    group_id                       previous = ID_INVALID;
    group_id                       next     = ID_INVALID;
    std::map<std::string, int64_t> facts;
};

struct plan_diagnostics {
    size_t                   matcher_probes = 0;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

class matcher;
class planner;
class graph_plan_impl;

class graph_plan {
  public:
    static graph_plan import(const ggml_cgraph * graph);
    static graph_plan deserialize_json(const std::string & json);

    bool valid() const { return diagnostics_.errors.empty(); }

    plan_phase phase() const { return phase_; }

    const std::vector<graph_storage> & storages() const { return storages_; }

    const std::vector<graph_value> & values() const { return values_; }

    const std::vector<graph_op> & operations() const { return operations_; }

    const std::vector<value_id> & roots() const { return roots_; }

    const std::vector<graph_region> & regions() const { return regions_; }

    const std::vector<region_id> & active_regions() const { return active_regions_; }

    const std::vector<semantic_group> & groups() const { return groups_; }

    const std::map<std::string, int64_t> & facts() const { return facts_; }

    const std::vector<selected_recipe> & selected_recipes() const { return selected_recipes_; }

    const plan_diagnostics & diagnostics() const { return diagnostics_; }

    const std::vector<op_id> & consumers(value_id value) const;
    const std::vector<op_id> & predecessors(op_id operation) const;
    const std::vector<op_id> & successors(op_id operation) const;
    const std::vector<op_id> & operations_by_kind(enum ggml_op kind) const;
    op_id                      storage_writer(storage_id storage, uint32_t version) const;
    region_boundary            calculate_boundary(const std::vector<op_id> & operations) const;

    bool        verify(bool expensive = false) const;
    std::string format() const;
    std::string serialize_json() const;
    std::string dot() const;

  private:
    void build_index();
    void initialize_atoms();
    void add_error(std::string error);

    plan_phase                 phase_ = plan_phase::PLAN_PHASE_IMPORTED;
    std::vector<graph_storage> storages_;
    std::vector<graph_value>   values_;
    std::vector<graph_op>      operations_;
    std::vector<value_id>      roots_;

    std::vector<std::vector<op_id>>                  consumers_;
    std::vector<std::vector<op_id>>                  predecessors_;
    std::vector<std::vector<op_id>>                  successors_;
    std::array<std::vector<op_id>, GGML_OP_COUNT>    operations_by_kind_;
    std::map<std::pair<storage_id, uint32_t>, op_id> storage_writers_;

    std::vector<graph_region>      regions_;
    std::vector<region_id>         active_regions_;
    std::vector<region_id>         source_op_owner_;
    std::vector<semantic_group>    groups_;
    std::map<std::string, int64_t> facts_;
    std::vector<selected_recipe>   selected_recipes_;
    plan_diagnostics               diagnostics_;

    friend class matcher;
    friend class planner;
    friend class graph_plan_impl;
    friend struct graph_import;
};

struct graph_import {
    graph_plan                       plan;
    std::vector<const ggml_tensor *> value_tensors;
    std::vector<const ggml_tensor *> storage_roots;

    static graph_import import(const ggml_cgraph * graph);
};

const char * boundary_kind_name(boundary_kind kind);
const char * region_kind_name(region_kind kind);
const char * recipe_parameter_kind_name(recipe_parameter_kind kind);
const char * schedule_capability_name(schedule_capability capability);

}  // namespace ggml::hrx
