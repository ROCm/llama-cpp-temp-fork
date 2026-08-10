#pragma once

#include "program-selection.h"
#include "transitional-graph.h"
#include "transitional-resource-access.h"
#include "transitional-schedule.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct ResourceUse {
    uint32_t       invocation     = 0;
    StorageId      storage        = kInvalidId;
    uint32_t       before_version = 0;
    uint32_t       after_version  = 0;
    ResourceAccess access         = ResourceAccess::Read;
};

struct ResourceContract {
    StorageId            storage          = kInvalidId;
    size_t               size             = 0;
    bool                 imported         = false;
    bool                 weight           = false;
    bool                 mutable_state    = false;
    bool                 exported         = false;
    bool                 elidable         = false;
    uint32_t             final_version    = 0;
    uint32_t             first_invocation = UINT32_MAX;
    uint32_t             last_invocation  = 0;
    std::vector<ValueId> aliases;
};

struct ResourceProgram {
    std::vector<ResourceContract> resources;
    std::vector<ResourceUse>      uses;
};

struct ProgramPlan {
    Graph                    graph;
    Schedule                 schedule;
    ResourceProgram          resources;
    std::string              semantic_witness;
    std::string              planner_identity;
    std::string              fusion_search_text;
    std::string              fusion_search_json;
    std::string              fusion_regions_dot;
    std::string              logical_program_text;
    std::string              logical_program_json;
    std::string              logical_program_dot;
    size_t                   atom_fallback_count = 0;
    std::string              target;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

class transitional_program_adapter {
  public:
    static bool bind_storage_roots(const graph_plan &                       source,
                                   const ProgramPlan &                      program,
                                   const std::vector<const ggml_tensor *> & source_roots,
                                   std::vector<const ggml_tensor *> &       program_roots,
                                   std::vector<std::string> &               errors);
};

bool               eager_capability_declared(enum ggml_op op);
ResourceProgram    build_resource_program(const Graph & graph, const Schedule & schedule);
VerificationResult verify_resource_program(const Graph &           graph,
                                           const Schedule &        schedule,
                                           const ResourceProgram & resources);
std::string        schedule_semantic_witness(const Graph & graph, const Schedule & schedule);
std::string        format_resource_program(const ResourceProgram & resources);
ProgramPlan        build_reactive_plan(const Graph & graph, const std::string & target);
ProgramPlan        build_transitional_program(const graph_plan &        plan,
                                              const program_selection & selection,
                                              const std::string &       target,
                                              uint64_t                  graph_uid);

}  // namespace ggml::hrx
