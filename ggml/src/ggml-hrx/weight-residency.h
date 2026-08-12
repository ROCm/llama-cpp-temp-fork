#pragma once

#include "hrx-interop-utils.h"
#include "storage-transform.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ggml::hrx {

class transfer_manager;

struct weight_source {
    // Base of the backing host allocation; offset selects the resident range.
    const void * host_data       = nullptr;
    uint64_t     buffer_identity = 0;
    uint64_t     generation      = 0;
    size_t       capacity        = 0;
    size_t       offset          = 0;
    size_t       length          = 0;
    std::string  layout          = "ggml-native";
    const kernel_storage_transform * transform = nullptr;
};

struct weight_residency_stats {
    uint64_t hits             = 0;
    uint64_t misses           = 0;
    uint64_t layout_conflicts = 0;
    size_t   allocation_count = 0;
    size_t   resident_bytes   = 0;
};

class weight_residency_lease {
  public:
    weight_residency_lease();
    ~weight_residency_lease();
    weight_residency_lease(const weight_residency_lease &)                 = default;
    weight_residency_lease & operator=(const weight_residency_lease &)     = default;
    weight_residency_lease(weight_residency_lease &&) noexcept             = default;
    weight_residency_lease & operator=(weight_residency_lease &&) noexcept = default;

    bool                valid() const;
    hrx_buffer_t        buffer() const;
    size_t              length() const;
    const std::string & layout() const;

  private:
    struct entry;
    std::shared_ptr<entry> entry_;
    explicit weight_residency_lease(std::shared_ptr<entry> entry);
    friend class weight_residency_cache;
};

struct weight_residency_result {
    weight_residency_lease lease;
    error_result           error;

    bool valid() const { return !error && lease.valid(); }
};

// Caches exceptional host-backed weights; GGML device-local weights bind directly and entries persist until teardown.
class weight_residency_cache {
  public:
    explicit weight_residency_cache(hrx_device_t device);
    ~weight_residency_cache();
    weight_residency_cache(const weight_residency_cache &)             = delete;
    weight_residency_cache & operator=(const weight_residency_cache &) = delete;

    bool                    valid() const;
    const std::string &     initialization_error() const;
    weight_residency_result acquire(hrx_stream_t stream, transfer_manager & transfers, const weight_source & source);
    weight_residency_stats  stats() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

std::string format_weight_residency_stats(const weight_residency_stats & stats);

}  // namespace ggml::hrx
