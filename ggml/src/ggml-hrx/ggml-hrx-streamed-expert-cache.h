#pragma once

#include "ggml-backend.h"
#include "ggml-hrx-expert-cache.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

struct streamed_expert_attachment {
    hrx_buffer_ref_t                host_cache = {};
    hrx_buffer_ref_t                device_cache = {};
    expert_cache_attachment_layout layout;
};

// Model-owned tensor buffers keep groups alive.
// The device-local registry stores weak references only.
class streamed_expert_group {
  public:
    bool resolve(hrx_device_t                               device,
                 hrx_stream_t                               transfer_stream,
                 const ggml_backend_streamed_weight_source * source,
                 streamed_expert_attachment &                attachment,
                 std::string &                               error);

    std::unique_ptr<expert_cache_layer_lease> begin_layer(
        hrx_device_t device,
        hrx_stream_t transfer_stream,
        uint32_t layer,
        const uint32_t * expert_ids,
        size_t expert_id_count,
        size_t load_chunk_size,
        std::string & error);

  private:
    explicit streamed_expert_group(const void * identity);

    bool add_source(const ggml_backend_streamed_weight_source * source, std::string & error);
    bool ensure_cache(hrx_device_t device, hrx_stream_t transfer_stream, std::string & error);

    const void * identity_ = nullptr;
    mutable std::mutex mutex_;
    std::vector<const ggml_backend_streamed_weight_source *> sources_;
    std::unique_ptr<expert_cache> cache_;
    bool initialization_attempted_ = false;
    std::string initialization_error_;

    friend class streamed_expert_group_registry;
};

class streamed_expert_group_registry {
  public:
    std::shared_ptr<streamed_expert_group> attach(
        const ggml_backend_streamed_weight_source * source,
        std::string & error);

  private:
    std::mutex mutex_;
    std::unordered_map<const void *, std::weak_ptr<streamed_expert_group>> groups_;
};

}  // namespace ggml::hrx
