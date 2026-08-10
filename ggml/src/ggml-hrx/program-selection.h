#pragma once

#include "graph-plan.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

enum class program_implementation : uint8_t {
    PROGRAM_IMPLEMENTATION_TRANSITIONAL,
    PROGRAM_IMPLEMENTATION_LOOM,
};

struct fixed_parameter_root {
    std::string             role;
    std::vector<storage_id> storages;
};

struct runtime_resource {
    std::string role;
    storage_id  storage = ID_INVALID;
    value_id    value   = ID_INVALID;
};

struct program_coverage {
    std::vector<region_id> regions;
    std::vector<op_id>     operations;
};

struct missing_program_step {
    region_id    region    = ID_INVALID;
    op_id        operation = ID_INVALID;
    enum ggml_op op        = GGML_OP_COUNT;
    std::string  reason;
};

struct program_selection {
    program_implementation             implementation = program_implementation::PROGRAM_IMPLEMENTATION_TRANSITIONAL;
    std::string                        library;
    std::vector<std::string>           roots;
    std::map<std::string, int64_t>     source_configuration;
    std::map<std::string, std::string> target_specialization;
    std::vector<fixed_parameter_root>  fixed_parameters;
    std::vector<runtime_resource>      runtime_resources;
    program_coverage                   coverage;
    std::vector<missing_program_step>  missing;
    std::vector<std::string>           errors;

    bool valid() const { return errors.empty() && missing.empty() && !roots.empty(); }

    std::string format() const;
    std::string serialize_json() const;

    static program_selection select(const graph_plan & plan, const std::string & target);
};

struct graph_execution_frame {
    std::shared_ptr<const graph_plan>        plan;
    std::shared_ptr<const program_selection> selection;
    std::vector<const ggml_tensor *>         values;
    std::vector<const ggml_tensor *>         storage_roots;
    uint64_t                                 uid = 0;
    std::vector<std::string>                 errors;

    bool valid() const {
        return errors.empty() && plan != nullptr && selection != nullptr && plan->valid() && selection->valid();
    }
};

struct graph_plan_cache_stats {
    uint64_t builds   = 0;
    uint64_t hits     = 0;
    uint64_t failures = 0;
};

class graph_plan_cache {
  public:
    graph_execution_frame  prepare(const ggml_cgraph * graph, const std::string & target);
    graph_plan_cache_stats stats() const;

  private:
    struct entry {
        std::string                              target;
        std::shared_ptr<const graph_plan>        plan;
        std::shared_ptr<const program_selection> selection;
        const ggml_cgraph *                      graph = nullptr;
        std::vector<const ggml_tensor *>         values;
        std::vector<const ggml_tensor *>         storage_roots;
    };

    mutable std::mutex                  mutex_;
    std::unordered_map<uint64_t, entry> entries_;
    graph_plan_cache_stats              stats_;
};

}  // namespace ggml::hrx
