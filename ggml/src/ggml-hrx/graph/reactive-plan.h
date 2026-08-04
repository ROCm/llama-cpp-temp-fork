#pragma once

#include "graph-ir.h"
#include "resource-access.h"
#include "schedule.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

struct ResourceUse {
    uint32_t invocation = 0;
    StorageId storage = kInvalidId;
    uint32_t before_version = 0;
    uint32_t after_version = 0;
    ResourceAccess access = ResourceAccess::Read;

    static const char * access_name(ResourceAccess access);
};

struct ResourceContract {
    StorageId storage = kInvalidId;
    size_t size = 0;
    bool imported = false;
    bool weight = false;
    bool mutable_state = false;
    bool exported = false;
    bool elidable = false;
    uint32_t final_version = 0;
    uint32_t first_invocation = UINT32_MAX;
    uint32_t last_invocation = 0;
    std::vector<ValueId> aliases;
};

struct ResourceProgram {
    std::vector<ResourceContract> resources;
    std::vector<ResourceUse> uses;

    static ResourceProgram build(const Graph & graph, const Schedule & schedule);
    static VerificationResult verify(const Graph & graph, const Schedule & schedule,
                                     const ResourceProgram & resources);
    static std::string format(const ResourceProgram & resources);
};

struct ProgramPlan {
    Graph graph;
    Schedule schedule;
    ResourceProgram resources;
    std::string semantic_witness;
    std::string planner_identity;
    std::string fusion_search_text;
    std::string fusion_search_json;
    std::string fusion_regions_dot;
    bool legacy_oracle_equivalent = false;
    std::string target;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }

    static bool eager_capability_declared(enum ggml_op op);
    static ProgramPlan build(const Graph & graph, const std::string & target);
};

struct ExecutionFrame {
    std::shared_ptr<const ProgramPlan> plan;
    std::vector<const ggml_tensor *> values;
    std::vector<const ggml_tensor *> storage_roots;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty() && plan != nullptr && plan->valid(); }
};

struct PlanCacheStats {
    uint64_t builds = 0;
    uint64_t hits = 0;
    uint64_t semantic_collisions = 0;
    uint64_t failures = 0;
};

class ReactivePlanCache {
public:
    ExecutionFrame prepare(const ggml_cgraph * graph, const std::string & target);
    PlanCacheStats stats() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<const ProgramPlan>>> plans_;
    PlanCacheStats stats_;
};

} // namespace ggml::hrx
