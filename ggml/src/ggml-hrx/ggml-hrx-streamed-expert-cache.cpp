#include "ggml-hrx-streamed-expert-cache.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static bool checked_mul_size(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool checked_add_size(size_t lhs, size_t rhs, size_t & result) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static bool align_up_size(size_t value, size_t alignment, size_t & result) {
    size_t rounded;
    if (!checked_add_size(value, alignment - 1, rounded)) {
        return false;
    }
    result = rounded & ~(alignment - 1);
    return true;
}

static bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static bool valid_cache_layout(const ggml_backend_streamed_weight_source * source) {
    if (source->cache_layout == GGML_BACKEND_STREAMED_WEIGHT_LAYOUT_NONE) {
        return source->cache_layout_row_count == 0 && source->cache_layout_block_count == 0 &&
               source->cache_layout_block_bytes == 0 && source->cache_layout_tile_rows == 0;
    }
    if (source->cache_layout != GGML_BACKEND_STREAMED_WEIGHT_LAYOUT_ROW_TILE_BLOCK ||
        source->cache_layout_row_count == 0 || source->cache_layout_block_count == 0 ||
        source->cache_layout_block_bytes == 0 || source->cache_layout_tile_rows == 0 ||
        source->cache_layout_row_count % source->cache_layout_tile_rows != 0) {
        return false;
    }
    size_t elements;
    size_t bytes;
    return checked_mul_size(source->cache_layout_row_count, source->cache_layout_block_count, elements) &&
           checked_mul_size(elements, source->cache_layout_block_bytes, bytes) && bytes == source->record_size;
}

struct layer_builder {
    uint32_t                                                        record_count = 0;
    uint16_t                                                        plane_count  = 0;
    std::map<uint16_t, const ggml_backend_streamed_weight_source *> planes;
};

static constexpr size_t streamed_expert_cache_slot_multiplier        = 15;
static constexpr size_t streamed_expert_cache_slot_limit             = 3840;
static constexpr size_t streamed_expert_cache_expanded_slot_limit    = 11008;
static constexpr size_t streamed_expert_cache_transient_slots        = 32;
static constexpr size_t streamed_expert_cache_device_reserve         = size_t{ 9 } << 30;
static constexpr size_t streamed_expert_cache_device_minimum_reserve = size_t{ 8 } << 30;
static constexpr size_t streamed_expert_cache_host_arena_budget      = size_t{ 19 } << 30;
static constexpr size_t streamed_expert_cache_host_arena_limit       = size_t{ 20 } << 30;
static constexpr size_t streamed_expert_cache_two_tier_device_floor  = size_t{ 90 } << 30;

}  // namespace

streamed_expert_group::streamed_expert_group(const void * identity) : identity_(identity) {}

bool streamed_expert_group::add_source(const ggml_backend_streamed_weight_source * source, std::string & error) {
    if (!source || !source->group_identity || source->group_identity != identity_) {
        error = "streamed expert source has the wrong group identity";
        return false;
    }
    if (source->fd < 0 || source->length == 0 || source->record_count == 0 || source->record_size == 0 ||
        source->plane_count == 0 || source->plane_index >= source->plane_count || source->offset > source->file_size ||
        source->length > source->file_size - source->offset || !valid_cache_layout(source)) {
        error = "streamed expert source descriptor is invalid";
        return false;
    }
    size_t expected_length = 0;
    if (!checked_mul_size(source->record_count, source->record_size, expected_length) ||
        expected_length != source->length) {
        error = "streamed expert source is not a contiguous record plane";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (initialization_attempted_) {
        error = "streamed expert source was attached after cache initialization";
        return false;
    }
    if (std::find(sources_.begin(), sources_.end(), source) != sources_.end()) {
        error = "streamed expert source was attached twice";
        return false;
    }
    for (const auto * existing : sources_) {
        if (existing->layer_index == source->layer_index && existing->plane_index == source->plane_index) {
            error = "streamed expert layer plane was attached twice";
            return false;
        }
    }
    sources_.push_back(source);
    return true;
}

bool streamed_expert_group::ensure_cache(hrx_device_t device, hrx_stream_t transfer_stream, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_) {
        return true;
    }
    if (initialization_attempted_) {
        error = initialization_error_;
        return false;
    }
    initialization_attempted_ = true;

    if (!device || !transfer_stream || !identity_ || sources_.empty()) {
        initialization_error_ = "streamed expert group is incomplete";
        error                 = initialization_error_;
        return false;
    }

    std::map<uint32_t, layer_builder> layers;
    size_t                            logical_slot_size  = 0;
    size_t                            io_alignment       = expert_cache_slot_base_alignment;
    uint32_t                          group_record_count = 0;
    uint16_t                          group_plane_count  = 0;
    for (const auto * source : sources_) {
        if (!source || source->group_identity != identity_ || source->record_stride == 0 ||
            source->destination_offset > source->record_stride ||
            source->record_size > source->record_stride - source->destination_offset ||
            !is_power_of_two(source->read_alignment)) {
            initialization_error_ = "streamed expert group contains an unfinished source descriptor";
            error                 = initialization_error_;
            return false;
        }
        if (group_record_count == 0) {
            group_record_count = source->record_count;
            group_plane_count  = source->plane_count;
        } else if (source->record_count != group_record_count || source->plane_count != group_plane_count) {
            initialization_error_ = "streamed expert group has inconsistent record or plane counts";
            error                 = initialization_error_;
            return false;
        }

        auto & layer = layers[source->layer_index];
        if (layer.record_count == 0) {
            layer.record_count = source->record_count;
            layer.plane_count  = source->plane_count;
        }
        if (layer.record_count != source->record_count || layer.plane_count != source->plane_count ||
            !layer.planes.emplace(source->plane_index, source).second) {
            initialization_error_ = "streamed expert layer has inconsistent or duplicate planes";
            error                 = initialization_error_;
            return false;
        }
        logical_slot_size = std::max(logical_slot_size, source->record_stride);
        io_alignment      = std::max(io_alignment, source->read_alignment);
    }

    std::vector<expert_cache_layer_source> cache_layers;
    cache_layers.reserve(layers.size());
    size_t slot_size         = logical_slot_size;
    bool   has_direct_layout = false;
    for (const auto & entry : layers) {
        const layer_builder & builder = entry.second;
        if (builder.planes.size() != builder.plane_count) {
            initialization_error_ = "streamed expert layer is missing a plane";
            error                 = initialization_error_;
            return false;
        }
        expert_cache_layer_source layer;
        layer.layer        = entry.first;
        layer.expert_count = builder.record_count;
        layer.ranges.reserve(builder.planes.size());
        const bool direct_layout   = std::all_of(builder.planes.begin(), builder.planes.end(), [&](const auto & plane) {
            return plane.second->record_size % io_alignment == 0;
        });
        size_t     physical_offset = 0;
        for (const auto & plane : builder.planes) {
            const auto * source             = plane.second;
            size_t       slot_offset        = source->destination_offset;
            size_t       direct_slot_offset = 0;
            size_t       direct_read_length = 0;
            if (direct_layout) {
                const size_t payload_prefix = static_cast<size_t>(source->offset & (io_alignment - 1));
                size_t       direct_extent;
                if (!checked_add_size(payload_prefix, source->record_size, direct_extent) ||
                    !align_up_size(direct_extent, io_alignment, direct_read_length) ||
                    !checked_add_size(physical_offset, payload_prefix, slot_offset)) {
                    initialization_error_ = "streamed expert direct-I/O slot layout overflow";
                    error                 = initialization_error_;
                    return false;
                }
                direct_slot_offset = physical_offset;
                if (!checked_add_size(physical_offset, direct_read_length, physical_offset)) {
                    initialization_error_ = "streamed expert direct-I/O slot layout overflow";
                    error                 = initialization_error_;
                    return false;
                }
            }
            layer.ranges.push_back({
                /* .plane         = */ source->plane_index,
                /* .fd            = */ source->fd,
                /* .file_offset   = */ source->offset,
                /* .expert_stride = */ source->record_size,
                /* .byte_length   = */ source->record_size,
                /* .slot_offset   = */ slot_offset,
                /* .direct_slot_offset = */ direct_slot_offset,
                /* .direct_read_length = */ direct_read_length,
                /* .payload_layout = */ source->cache_layout == GGML_BACKEND_STREAMED_WEIGHT_LAYOUT_ROW_TILE_BLOCK ?
                    expert_cache_payload_layout::row_tile_block : expert_cache_payload_layout::native,
                /* .layout_row_count = */ source->cache_layout_row_count,
                /* .layout_block_count = */ source->cache_layout_block_count,
                /* .layout_block_bytes = */ source->cache_layout_block_bytes,
                /* .layout_tile_rows = */ source->cache_layout_tile_rows,
            });
        }
        if (direct_layout) {
            has_direct_layout = true;
            slot_size         = std::max(slot_size, physical_offset);
        }
        cache_layers.push_back(std::move(layer));
    }
    if (has_direct_layout && !align_up_size(slot_size, io_alignment, slot_size)) {
        initialization_error_ = "streamed expert direct-I/O slot stride overflow";
        error                 = initialization_error_;
        return false;
    }

    expert_cache_config config;
    config.group = static_cast<expert_cache_group_id>(reinterpret_cast<uintptr_t>(identity_));
    size_t expanded_slot_count;
    size_t total_record_count;
    if (!checked_mul_size(group_record_count, streamed_expert_cache_slot_multiplier, expanded_slot_count) ||
        !checked_mul_size(group_record_count, cache_layers.size(), total_record_count)) {
        initialization_error_ = "streamed expert cache slot count overflow";
        error                 = initialization_error_;
        return false;
    }
    uint64_t     total_memory = 0;
    hrx_status_t memory_status =
        hrx_device_get_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total_memory, sizeof(total_memory));
    if (!hrx_status_is_ok(memory_status)) {
        char * message = nullptr;
        size_t length  = 0;
        hrx_status_to_string(memory_status, &message, &length);
        initialization_error_ = "streamed expert cache memory query: " +
                                (message ? std::string(message, length) : std::string("unknown HRX error"));
        hrx_status_free_message(message);
        hrx_status_ignore(memory_status);
        error = initialization_error_;
        return false;
    }
    if (total_memory > std::numeric_limits<size_t>::max()) {
        initialization_error_ = "streamed expert cache total memory exceeds size_t";
        error                 = initialization_error_;
        return false;
    }
    const size_t total_bytes         = static_cast<size_t>(total_memory);
    size_t       device_arena_budget = 0;
    if (total_bytes > streamed_expert_cache_device_reserve + expert_cache_binding_header_size) {
        device_arena_budget = total_bytes - streamed_expert_cache_device_reserve - expert_cache_binding_header_size;
    }
    size_t physical_device_arena_limit = 0;
    if (total_bytes > streamed_expert_cache_device_minimum_reserve + expert_cache_binding_header_size) {
        physical_device_arena_limit =
            total_bytes - streamed_expert_cache_device_minimum_reserve - expert_cache_binding_header_size;
    }
    const size_t device_payload_slot_limit  = device_arena_budget / logical_slot_size;
    const size_t device_physical_slot_limit = physical_device_arena_limit / slot_size;
    const size_t device_memory_slot_limit   = std::min(device_payload_slot_limit, device_physical_slot_limit);
    const bool   use_expanded_device_budget = device_memory_slot_limit > streamed_expert_cache_slot_limit;
    const size_t memory_slot_limit          = device_memory_slot_limit;
    if (memory_slot_limit < group_record_count) {
        initialization_error_ = "streamed expert cache memory budget cannot hold one layer expert union";
        error                 = initialization_error_;
        return false;
    }

    config.slot_count = group_record_count;
    if (cache_layers.size() > 1) {
        const size_t policy_slot_limit = use_expanded_device_budget ?
                                             streamed_expert_cache_expanded_slot_limit :
                                             std::min(expanded_slot_count, streamed_expert_cache_slot_limit);
        config.slot_count              = std::min({
            total_record_count,
            policy_slot_limit,
            memory_slot_limit,
        });
        if (total_bytes >= streamed_expert_cache_two_tier_device_floor && config.slot_count < total_record_count &&
            config.slot_count < policy_slot_limit) {
            const size_t host_payload_slot_limit  = streamed_expert_cache_host_arena_budget / logical_slot_size;
            const size_t host_physical_slot_limit = streamed_expert_cache_host_arena_limit / slot_size;
            const size_t host_memory_slot_limit   = std::min(host_payload_slot_limit, host_physical_slot_limit);
            config.host_slot_count                = std::min({
                host_memory_slot_limit,
                total_record_count - config.slot_count,
                policy_slot_limit - config.slot_count,
            });
            config.slot_count += config.host_slot_count;
        }
    }
    config.transient_slot_count =
        std::min(streamed_expert_cache_transient_slots, config.slot_count - group_record_count);
    config.slot_size     = slot_size;
    config.worker_count  = std::min<size_t>(4, std::max<uint32_t>(1, group_record_count));
    config.io_alignment  = io_alignment;
    config.prefer_direct = true;

    std::string cache_error;
    cache_ = expert_cache::create(device, transfer_stream, config, cache_layers, cache_error);
    if (!cache_) {
        initialization_error_ = std::move(cache_error);
        error                 = initialization_error_;
        return false;
    }
    return true;
}

bool streamed_expert_group::resolve(hrx_device_t                               device,
                                    hrx_stream_t                               transfer_stream,
                                    const ggml_backend_streamed_weight_source * source,
                                    streamed_expert_attachment &                attachment,
                                    std::string &                               error) {
    if (!source || source->group_identity != identity_ || !ensure_cache(device, transfer_stream, error)) {
        if (error.empty()) {
            error = "streamed expert source belongs to another group";
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cache_->resolve_attachment({
                                        /* .group = */ cache_->group_id(),
                                        /* .layer = */ source->layer_index,
                                        /* .plane = */ source->plane_index,
                                    }, attachment.layout, error)) {
        return false;
    }
    attachment.host_cache = {
        /* .buffer = */ cache_->binding_buffer("host_cache"),
        /* .offset = */ 0,
        /* .length = */ cache_->binding_size("host_cache"),
    };
    attachment.device_cache = {
        /* .buffer = */ cache_->binding_buffer("device_cache"),
        /* .offset = */ 0,
        /* .length = */ cache_->binding_size("device_cache"),
    };
    if (!attachment.host_cache.buffer || attachment.host_cache.length == 0 ||
        !attachment.device_cache.buffer || attachment.device_cache.length == 0) {
        error = "streamed expert cache has incomplete storage bindings";
        return false;
    }
    return true;
}

std::unique_ptr<expert_cache_layer_lease> streamed_expert_group::begin_layer(hrx_device_t     device,
                                                                             hrx_stream_t     transfer_stream,
                                                                             uint32_t         layer,
                                                                             const uint32_t * expert_ids,
                                                                             size_t           expert_id_count,
                                                                             size_t           load_chunk_size,
                                                                             std::string &    error) {
    if (!ensure_cache(device, transfer_stream, error)) {
        return nullptr;
    }
    return cache_->begin_layer(
        layer, expert_ids, expert_id_count, load_chunk_size, error);
}

std::shared_ptr<streamed_expert_group> streamed_expert_group_registry::attach(
    const ggml_backend_streamed_weight_source * source,
    std::string &                               error) {
    if (!source || !source->group_identity) {
        error = "streamed expert source has no group identity";
        return nullptr;
    }

    std::shared_ptr<streamed_expert_group> group;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &                      weak = groups_[source->group_identity];
        group                            = weak.lock();
        if (!group) {
            try {
                group = std::shared_ptr<streamed_expert_group>(new streamed_expert_group(source->group_identity));
            } catch (const std::bad_alloc &) {
                error = "failed to allocate streamed expert group";
                groups_.erase(source->group_identity);
                return nullptr;
            }
            weak = group;
        }
    }
    if (!group->add_source(source, error)) {
        return nullptr;
    }
    return group;
}

}  // namespace ggml::hrx
