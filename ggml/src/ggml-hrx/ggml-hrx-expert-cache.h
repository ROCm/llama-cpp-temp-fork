#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml::hrx {

using expert_cache_group_id = uint64_t;

enum class expert_cache_payload_layout : uint8_t {
    native,
    row_tile_block,
};

inline constexpr size_t  expert_cache_slot_map_offset     = 0;
inline constexpr size_t  expert_cache_slot_map_bytes      = 4096;
inline constexpr size_t  expert_cache_cohort_map_offset   = expert_cache_slot_map_bytes;
inline constexpr size_t  expert_cache_cohort_map_bytes    = 4096;
inline constexpr size_t  expert_cache_binding_header_size = expert_cache_slot_map_bytes + expert_cache_cohort_map_bytes;
inline constexpr size_t  expert_cache_slot_base_alignment = 4096;
inline constexpr size_t  expert_cache_slot_map_capacity   = expert_cache_slot_map_bytes / sizeof(int32_t);
inline constexpr size_t  expert_cache_cohort_map_capacity = expert_cache_cohort_map_bytes / sizeof(int32_t);
inline constexpr int32_t expert_cache_slot_unavailable    = -1;
inline constexpr int32_t expert_cache_cohort_unavailable  = -1;
inline constexpr int32_t expert_cache_cohort_resident     = 1;
inline constexpr int32_t expert_cache_cohort_missing      = 2;
static_assert(sizeof(int32_t) == 4, "expert cache binding ABI requires 32-bit slot-map entries");
static_assert(expert_cache_slot_map_capacity == expert_cache_cohort_map_capacity,
              "expert cache slot and cohort maps must cover the same expert IDs");
static_assert(expert_cache_binding_header_size % expert_cache_slot_base_alignment == 0,
              "expert cache slot base must satisfy its ABI alignment");

// Describes one stored part of every expert in a layer. The cache duplicates fd and gathers each part into its slot.
struct expert_cache_range {
    uint32_t plane              = 0;
    int      fd                 = -1;
    uint64_t file_offset        = 0;
    uint64_t expert_stride      = 0;
    size_t   byte_length        = 0;
    size_t   slot_offset        = 0;
    size_t   direct_slot_offset = 0;
    size_t   direct_read_length = 0;
    expert_cache_payload_layout payload_layout = expert_cache_payload_layout::native;
    uint32_t layout_row_count   = 0;
    uint32_t layout_block_count = 0;
    uint32_t layout_block_bytes = 0;
    uint32_t layout_tile_rows   = 0;
};

// Identifies the cache record attached to a zero-payload tensor.
struct expert_cache_source_attachment {
    expert_cache_group_id group = 0;
    uint32_t              layer = 0;
    uint32_t              plane = 0;
};

struct expert_cache_attachment_layout {
    size_t slot_base_offset        = expert_cache_binding_header_size;
    size_t slot_count              = 0;
    size_t slot_stride             = 0;
    size_t plane_offset            = 0;
    size_t plane_length            = 0;
    size_t device_slot_base_offset = 0;
    size_t device_slot_count       = 0;
    size_t host_slot_base_offset   = expert_cache_binding_header_size;
    size_t host_slot_count         = 0;
};

struct expert_cache_layer_source {
    uint32_t                        layer        = 0;
    uint32_t                        expert_count = 0;
    std::vector<expert_cache_range> ranges;
};

struct expert_cache_config {
    // The persistent pool holds the largest per-layer expert union. A soft per-layer quota protects spare capacity.
    expert_cache_group_id group                = 0;
    size_t                slot_count           = 0;
    // Optional host-visible suffix of the global slot range.
    // Earlier slots are device-local; the suffix uses mapped host-local, device-visible memory.
    size_t                host_slot_count      = 0;
    size_t                transient_slot_count = 0;  // suffix of slot_count
    size_t                slot_size            = 0;
    size_t                worker_count         = 1;
    size_t                io_alignment         = 4096;
    bool                  prefer_direct        = true;
};

class expert_cache_layer_lease;

class expert_cache {
  public:
    static std::unique_ptr<expert_cache> create(hrx_device_t                                   device,
                                                hrx_stream_t                                   transfer_stream,
                                                const expert_cache_config &                    config,
                                                const std::vector<expert_cache_layer_source> & layers,
                                                std::string &                                  error);

    ~expert_cache();

    expert_cache(const expert_cache &)             = delete;
    expert_cache & operator=(const expert_cache &) = delete;

    // Reserves experts and starts every plane read. The caller must wait for each plane before dispatch.
    std::unique_ptr<expert_cache_layer_lease> begin_layer(uint32_t         layer,
                                                          const uint32_t * expert_ids,
                                                          size_t           expert_id_count,
                                                          size_t           load_chunk_size,
                                                          std::string &    error);

    // One binding starts with the expert-to-slot and residency-cohort maps.
    // Cache slots follow the fixed header.
    hrx_buffer_t    binding_buffer(const char * storage_binding = nullptr) const;
    size_t          binding_size(const char * storage_binding = nullptr) const;

    expert_cache_group_id group_id() const;
    bool                  resolve_attachment(const expert_cache_source_attachment & attachment,
                                             expert_cache_attachment_layout &       layout,
                                             std::string &                          error) const;

  private:
    struct shared_state;

    explicit expert_cache(std::shared_ptr<shared_state> state);

    std::shared_ptr<shared_state> state_;

    friend class expert_cache_layer_lease;
};

class expert_cache_layer_lease {
  public:
    ~expert_cache_layer_lease();

    expert_cache_layer_lease(const expert_cache_layer_lease &)             = delete;
    expert_cache_layer_lease & operator=(const expert_cache_layer_lease &) = delete;

    bool     publish_header(hrx_stream_t graph_stream, std::string & error);
    bool     wait_for_planes(const uint32_t * planes, size_t plane_count, std::string & error);
    bool     wait_for_planes(uint32_t expert_begin, uint32_t expert_end,
                             const uint32_t * planes, size_t plane_count, std::string & error);

  private:
    expert_cache_layer_lease(std::shared_ptr<expert_cache::shared_state> state,
                             uint32_t                                    layer,
                             std::vector<size_t>                         slots,
                             std::vector<uint32_t>                       experts);

    std::shared_ptr<expert_cache::shared_state> state_;
    uint32_t                                    layer_ = 0;
    std::vector<size_t>                         slots_;
    std::vector<uint32_t>                       experts_;

    friend class expert_cache;
};

}  // namespace ggml::hrx
