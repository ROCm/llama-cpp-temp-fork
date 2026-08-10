#pragma once

#include "graph-plan.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class match_status : uint8_t {
    MATCH_STATUS_NO_MATCH,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_MALFORMED,
};

struct region_match {
    region_kind        kind    = region_kind::REGION_KIND_ATOM;
    region_payload     payload = atom_region{};
    op_id              anchor  = ID_INVALID;
    std::vector<op_id> operations;
};

struct match_result {
    match_status status = match_status::MATCH_STATUS_NO_MATCH;
    region_match match;
    std::string  diagnostic;

    static match_result no_match();
    static match_result matched(region_match match);
    static match_result malformed(std::string diagnostic, op_id anchor);
};

class match_cursor {
  public:
    match_cursor(const graph_plan & plan, size_t budget);

    const graph_op *    op(op_id id);
    const graph_value * value(value_id id);
    op_id               producer(value_id value, enum ggml_op kind = GGML_OP_COUNT);
    std::vector<op_id>  consumers(value_id value, enum ggml_op kind = GGML_OP_COUNT);
    op_id               unique_consumer(value_id value, enum ggml_op kind);
    value_id            output(op_id operation);
    bool                is_root(value_id value);
    std::vector<op_id>  chain(value_id value, const std::vector<enum ggml_op> & kinds);

    size_t probes() const { return probes_; }

    bool exhausted() const { return exhausted_; }

  private:
    bool probe();

    const graph_plan & plan_;
    size_t             budget_    = 0;
    size_t             probes_    = 0;
    bool               exhausted_ = false;
};

using pattern_callback = match_result (*)(match_cursor & cursor, op_id anchor);

struct semantic_group_match {
    semantic_group_kind            kind = semantic_group_kind::SEMANTIC_GROUP_KIND_ROUTED_BLOCK;
    std::vector<region_id>         members;
    std::map<std::string, int64_t> facts;
};

using semantic_group_callback = std::vector<semantic_group_match> (*)(const graph_plan & plan);

struct pattern_rule {
    const char *     name         = nullptr;
    enum ggml_op     anchor       = GGML_OP_COUNT;
    size_t           probe_budget = 0;
    pattern_callback callback     = nullptr;
};

class pattern_registry {
  public:
    void add(pattern_rule rule);
    void add(semantic_group_callback callback);

    const std::vector<pattern_rule> & rules() const { return rules_; }

    const std::vector<semantic_group_callback> & group_callbacks() const { return group_callbacks_; }

  private:
    std::vector<pattern_rule>            rules_;
    std::vector<semantic_group_callback> group_callbacks_;
};

class matcher {
  public:
    static bool recognize(graph_plan & plan, const pattern_registry & registry);

  private:
    static bool commit(graph_plan & plan, std::vector<region_match> matches);
    static bool commit_groups(graph_plan & plan, const pattern_registry & registry);
};

}  // namespace ggml::hrx
