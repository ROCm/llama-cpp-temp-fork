#include "ggml-hrx-expert-cache.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#    include <linux/io_uring.h>
#    include <sys/syscall.h>
#endif

namespace ggml::hrx {
namespace {

static bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static bool checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

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

static bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static uint64_t align_down(uint64_t value, size_t alignment) {
    return value & ~(static_cast<uint64_t>(alignment) - 1);
}

static bool align_up(uint64_t value, size_t alignment, uint64_t & result) {
    uint64_t rounded;
    if (!checked_add_u64(value, alignment - 1, rounded)) {
        return false;
    }
    result = rounded & ~(static_cast<uint64_t>(alignment) - 1);
    return true;
}

static uint64_t composite_key(uint32_t layer, uint32_t expert) {
    return (static_cast<uint64_t>(layer) << 32) | expert;
}

static std::string errno_message(const char * operation, int code) {
    return std::string(operation) + ": " + std::strerror(code);
}

static bool hrx_ok(hrx_status_t status, const char * operation, std::string & error) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_to_string(status, &message, &length);
    error = std::string(operation) + ": " + (message ? std::string(message, length) : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

class stable_buffer {
  public:
    ~stable_buffer() {
        if (buffer_) {
            if (mapped_) {
                hrx_status_ignore(hrx_buffer_unmap(buffer_));
            }
            hrx_buffer_release(buffer_);
        }
    }

    stable_buffer(const stable_buffer &)             = delete;
    stable_buffer & operator=(const stable_buffer &) = delete;

    static std::unique_ptr<stable_buffer> allocate_hrx_device(hrx_device_t device, size_t size, std::string & error) {
        if (!device || size == 0) {
            error = "expert cache: invalid HRX allocation request";
            return nullptr;
        }
        auto                      result = std::unique_ptr<stable_buffer>(new stable_buffer(size));
        const hrx_buffer_params_t params = {
            /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        };
        if (!hrx_ok(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, size, &result->buffer_),
                    "expert cache device-local allocation", error)) {
            return nullptr;
        }
        return result;
    }

    static std::unique_ptr<stable_buffer> allocate_hrx_staging(hrx_device_t device, size_t size, std::string & error) {
        if (!device || size == 0) {
            error = "expert cache: invalid HRX staging allocation request";
            return nullptr;
        }
        auto                      result = std::unique_ptr<stable_buffer>(new stable_buffer(size));
        const hrx_buffer_params_t params = {
            /* .type = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED |
                HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
            /* .queue_affinity = */ 0,
        };
        if (!hrx_ok(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, size, &result->buffer_),
                    "expert cache staging allocation", error)) {
            return nullptr;
        }
        void * mapped = nullptr;
        if (!hrx_ok(hrx_buffer_map(result->buffer_, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &mapped),
                    "expert cache staging map", error)) {
            hrx_buffer_release(result->buffer_);
            result->buffer_ = nullptr;
            return nullptr;
        }
        result->mapped_ = static_cast<uint8_t *>(mapped);
        return result;
    }

    uint8_t * data() const { return mapped_; }

    size_t size() const { return size_; }

    hrx_buffer_t buffer() const { return buffer_; }

  private:
    explicit stable_buffer(size_t size) : size_(size) {}

    hrx_buffer_t buffer_ = nullptr;
    uint8_t *    mapped_ = nullptr;
    size_t       size_   = 0;
};

class io_uring_reader {
  public:
    io_uring_reader() {
#if defined(__linux__) && defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
        io_uring_params params = {};
        fd_                    = static_cast<int>(syscall(__NR_io_uring_setup, 8u, &params));
        if (fd_ < 0) {
            return;
        }

        sq_ring_size_ = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
        cq_ring_size_ = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
        if ((params.features & IORING_FEAT_SINGLE_MMAP) != 0) {
            sq_ring_size_ = std::max(sq_ring_size_, cq_ring_size_);
            cq_ring_size_ = sq_ring_size_;
        }

        sq_ring_ =
            mmap(nullptr, sq_ring_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
        if (sq_ring_ == MAP_FAILED) {
            sq_ring_ = nullptr;
            reset();
            return;
        }
        if ((params.features & IORING_FEAT_SINGLE_MMAP) != 0) {
            cq_ring_ = sq_ring_;
        } else {
            cq_ring_ = mmap(nullptr, cq_ring_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_,
                            IORING_OFF_CQ_RING);
            if (cq_ring_ == MAP_FAILED) {
                cq_ring_ = nullptr;
                reset();
                return;
            }
        }

        sqes_size_ = params.sq_entries * sizeof(io_uring_sqe);
        sqes_      = static_cast<io_uring_sqe *>(
            mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) {
            sqes_ = nullptr;
            reset();
            return;
        }

        auto * sq   = static_cast<uint8_t *>(sq_ring_);
        auto * cq   = static_cast<uint8_t *>(cq_ring_);
        sq_head_    = reinterpret_cast<uint32_t *>(sq + params.sq_off.head);
        sq_tail_    = reinterpret_cast<uint32_t *>(sq + params.sq_off.tail);
        sq_mask_    = reinterpret_cast<uint32_t *>(sq + params.sq_off.ring_mask);
        sq_entries_ = reinterpret_cast<uint32_t *>(sq + params.sq_off.ring_entries);
        sq_array_   = reinterpret_cast<uint32_t *>(sq + params.sq_off.array);
        cq_head_    = reinterpret_cast<uint32_t *>(cq + params.cq_off.head);
        cq_tail_    = reinterpret_cast<uint32_t *>(cq + params.cq_off.tail);
        cq_mask_    = reinterpret_cast<uint32_t *>(cq + params.cq_off.ring_mask);
        cqes_       = reinterpret_cast<io_uring_cqe *>(cq + params.cq_off.cqes);
#endif
    }

    ~io_uring_reader() { reset(); }

    io_uring_reader(const io_uring_reader &)             = delete;
    io_uring_reader & operator=(const io_uring_reader &) = delete;

    bool available() const { return fd_ >= 0; }

    int read_full(int fd, void * buffer, size_t size, uint64_t offset) {
#if defined(__linux__) && defined(__NR_io_uring_enter)
        size_t done = 0;
        while (done < size) {
            const uint32_t head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
            const uint32_t tail = __atomic_load_n(sq_tail_, __ATOMIC_RELAXED);
            if (tail - head >= *sq_entries_) {
                return EAGAIN;
            }
            const uint32_t index = tail & *sq_mask_;
            io_uring_sqe & sqe   = sqes_[index];
            std::memset(&sqe, 0, sizeof(sqe));
            const size_t request_size =
                std::min(size - done, static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
            sqe.opcode       = IORING_OP_READ;
            sqe.fd           = fd;
            sqe.off          = offset + done;
            sqe.addr         = reinterpret_cast<uint64_t>(static_cast<uint8_t *>(buffer) + done);
            sqe.len          = static_cast<uint32_t>(request_size);
            sqe.user_data    = 1;
            sq_array_[index] = index;
            __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);

            long submitted;
            do {
                submitted = syscall(__NR_io_uring_enter, fd_, 1u, 1u, IORING_ENTER_GETEVENTS, nullptr, 0u);
            } while (submitted < 0 && errno == EINTR);
            if (submitted < 0) {
                return errno;
            }
            uint32_t cq_head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
            uint32_t cq_tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
            if (cq_head == cq_tail) {
                long waited;
                do {
                    waited = syscall(__NR_io_uring_enter, fd_, 0u, 1u, IORING_ENTER_GETEVENTS, nullptr, 0u);
                } while (waited < 0 && errno == EINTR);
                if (waited < 0) {
                    return errno;
                }
                cq_tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
                if (cq_head == cq_tail) {
                    return EIO;
                }
            }
            const io_uring_cqe & cqe    = cqes_[cq_head & *cq_mask_];
            const int            result = cqe.res;
            __atomic_store_n(cq_head_, cq_head + 1, __ATOMIC_RELEASE);
            if (result < 0) {
                return -result;
            }
            if (result == 0) {
                return EIO;
            }
            done += static_cast<size_t>(result);
        }
        return 0;
#else
        (void) fd;
        (void) buffer;
        (void) size;
        (void) offset;
        return ENOSYS;
#endif
    }

  private:
    void reset() {
#if defined(__linux__)
        if (sqes_) {
            munmap(sqes_, sqes_size_);
        }
        if (cq_ring_ && cq_ring_ != sq_ring_) {
            munmap(cq_ring_, cq_ring_size_);
        }
        if (sq_ring_) {
            munmap(sq_ring_, sq_ring_size_);
        }
#endif
        if (fd_ >= 0) {
            close(fd_);
        }
        fd_         = -1;
        sq_ring_    = nullptr;
        cq_ring_    = nullptr;
        sqes_       = nullptr;
        sq_head_    = nullptr;
        sq_tail_    = nullptr;
        sq_mask_    = nullptr;
        sq_entries_ = nullptr;
        sq_array_   = nullptr;
        cq_head_    = nullptr;
        cq_tail_    = nullptr;
        cq_mask_    = nullptr;
        cqes_       = nullptr;
    }

    int    fd_           = -1;
    void * sq_ring_      = nullptr;
    void * cq_ring_      = nullptr;
    size_t sq_ring_size_ = 0;
    size_t cq_ring_size_ = 0;
    size_t sqes_size_    = 0;
#if defined(__linux__)
    io_uring_sqe * sqes_ = nullptr;
    io_uring_cqe * cqes_ = nullptr;
#else
    void * sqes_ = nullptr;
    void * cqes_ = nullptr;
#endif
    uint32_t * sq_head_    = nullptr;
    uint32_t * sq_tail_    = nullptr;
    uint32_t * sq_mask_    = nullptr;
    uint32_t * sq_entries_ = nullptr;
    uint32_t * sq_array_   = nullptr;
    uint32_t * cq_head_    = nullptr;
    uint32_t * cq_tail_    = nullptr;
    uint32_t * cq_mask_    = nullptr;
};

static int pread_full(int fd, void * buffer, size_t size, uint64_t offset) {
    size_t done = 0;
    while (done < size) {
        const ssize_t result =
            pread(fd, static_cast<uint8_t *>(buffer) + done, size - done, static_cast<off_t>(offset + done));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno;
        }
        if (result == 0) {
            return EIO;
        }
        done += static_cast<size_t>(result);
    }
    return 0;
}

class backing_file {
  public:
    ~backing_file() {
        if (direct_fd_ >= 0) {
            close(direct_fd_);
        }
        if (buffered_fd_ >= 0) {
            close(buffered_fd_);
        }
    }

    backing_file(const backing_file &)             = delete;
    backing_file & operator=(const backing_file &) = delete;

    static std::shared_ptr<backing_file> open_from_fd(int           fd,
                                                      bool          prefer_direct,
                                                      dev_t         device,
                                                      ino_t         inode,
                                                      uint64_t      size,
                                                      std::string & error) {
        const int buffered = fcntl(fd, F_DUPFD_CLOEXEC, 0);
        if (buffered < 0) {
            error = errno_message("expert cache duplicate fd", errno);
            return nullptr;
        }
        int direct = -1;
#if defined(__linux__)
        if (prefer_direct) {
            const std::string path = "/proc/self/fd/" + std::to_string(fd);
            direct                 = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        }
#else
        (void) prefer_direct;
#endif
        return std::shared_ptr<backing_file>(new backing_file(buffered, direct, device, inode, size));
    }

    bool same_file(dev_t device, ino_t inode) const { return device_ == device && inode_ == inode; }

    bool read(io_uring_reader & ring,
              size_t            alignment,
              uint64_t          offset,
              size_t            size,
              uint8_t *         staging,
              size_t            staging_capacity,
              size_t &          payload_offset,
              std::string &     error) {
        uint64_t end;
        if (!checked_add_u64(offset, size, end) || end > size_) {
            error = "expert cache read exceeds backing file";
            return false;
        }

        if (direct_enabled_.load(std::memory_order_acquire)) {
            const uint64_t direct_offset = align_down(offset, alignment);
            uint64_t       direct_end;
            if (align_up(end, alignment, direct_end) && direct_end <= size_) {
                const uint64_t direct_size_u64 = direct_end - direct_offset;
                if (direct_size_u64 <= std::numeric_limits<size_t>::max()) {
                    const size_t direct_size = static_cast<size_t>(direct_size_u64);
                    if (direct_size > staging_capacity) {
                        error = "expert cache: aligned direct read exceeds staging slab";
                        return false;
                    }
                    int rc;
                    if (ring.available()) {
                        rc = ring.read_full(direct_fd_, staging, direct_size, direct_offset);
                    } else {
                        rc = pread_full(direct_fd_, staging, direct_size, direct_offset);
                    }
                    if (rc == 0) {
                        payload_offset = static_cast<size_t>(offset - direct_offset);
                        return true;
                    }
                    if (rc == EINVAL || rc == EIO || rc == EOPNOTSUPP || rc == ENOTSUP) {
                        direct_enabled_.store(false, std::memory_order_release);
                    } else {
                        error = errno_message("expert cache direct read", rc);
                        return false;
                    }
                }
            }
        }

        if (size > staging_capacity) {
            error = "expert cache: buffered read exceeds staging slab";
            return false;
        }
        const int rc = pread_full(buffered_fd_, staging, size, offset);
        if (rc != 0) {
            error = errno_message("expert cache buffered read", rc);
            return false;
        }
        payload_offset = 0;
        return true;
    }

    bool read_final(io_uring_reader & ring,
                    size_t            alignment,
                    uint64_t          offset,
                    size_t            size,
                    uint8_t *         destination,
                    size_t            direct_capacity,
                    size_t            payload_offset,
                    std::string &     error) {
        uint64_t end;
        if (!destination || !checked_add_u64(offset, size, end) || end > size_ || payload_offset > direct_capacity ||
            size > direct_capacity - payload_offset) {
            error = "expert cache direct-final read exceeds its source or slot region";
            return false;
        }

        const uint64_t direct_offset = align_down(offset, alignment);
        uint64_t       direct_end;
        if (offset - direct_offset != payload_offset || !align_up(end, alignment, direct_end) ||
            direct_end < direct_offset || direct_end - direct_offset != direct_capacity) {
            error = "expert cache direct-final layout does not match its source alignment";
            return false;
        }

        if (direct_enabled_.load(std::memory_order_acquire) && direct_end <= size_) {
            int rc;
            if (ring.available()) {
                rc = ring.read_full(direct_fd_, destination, direct_capacity, direct_offset);
            } else {
                rc = pread_full(direct_fd_, destination, direct_capacity, direct_offset);
            }
            if (rc == 0) {
                return true;
            }
            if (rc == EINVAL || rc == EIO || rc == EOPNOTSUPP || rc == ENOTSUP) {
                direct_enabled_.store(false, std::memory_order_release);
            } else {
                error = errno_message("expert cache direct-final read", rc);
                return false;
            }
        }

        const int rc = pread_full(buffered_fd_, destination + payload_offset, size, offset);
        if (rc != 0) {
            error = errno_message("expert cache buffered direct-final read", rc);
            return false;
        }
        return true;
    }

  private:
    backing_file(int buffered_fd, int direct_fd, dev_t device, ino_t inode, uint64_t size) :
        buffered_fd_(buffered_fd),
        direct_fd_(direct_fd),
        direct_enabled_(direct_fd >= 0),
        device_(device),
        inode_(inode),
        size_(size) {}

    int               buffered_fd_ = -1;
    int               direct_fd_   = -1;
    std::atomic<bool> direct_enabled_;
    dev_t             device_ = 0;
    ino_t             inode_  = 0;
    uint64_t          size_   = 0;
};

struct source_part {
    std::shared_ptr<backing_file> file;
    uint32_t                      plane              = 0;
    uint64_t                      file_offset        = 0;
    uint64_t                      expert_stride      = 0;
    size_t                        byte_length        = 0;
    size_t                        slot_offset        = 0;
    size_t                        direct_slot_offset = 0;
    size_t                        direct_read_length = 0;
    expert_cache_payload_layout   payload_layout     = expert_cache_payload_layout::native;
    uint32_t                      layout_row_count   = 0;
    uint32_t                      layout_block_count = 0;
    uint32_t                      layout_block_bytes = 0;
    uint32_t                      layout_tile_rows   = 0;
};

static bool payload_layout_tile_bytes(const source_part & part, size_t & tile_bytes) {
    size_t records;
    return part.payload_layout == expert_cache_payload_layout::row_tile_block && part.layout_tile_rows != 0 &&
           checked_mul_size(part.layout_tile_rows, part.layout_block_count, records) &&
           checked_mul_size(records, part.layout_block_bytes, tile_bytes);
}

static bool transform_payload_in_place(const source_part & part,
                                       uint8_t *           payload,
                                       size_t              payload_length,
                                       std::vector<uint8_t> & scratch,
                                       std::string &       error) {
    if (part.payload_layout == expert_cache_payload_layout::native) {
        return true;
    }
    size_t tile_bytes;
    if (!payload || payload_length != part.byte_length || !payload_layout_tile_bytes(part, tile_bytes) ||
        scratch.size() < tile_bytes || part.layout_row_count % part.layout_tile_rows != 0) {
        error = "expert cache: invalid row-tile payload transform";
        return false;
    }

    const size_t tile_count  = part.layout_row_count / part.layout_tile_rows;
    const size_t block_count = part.layout_block_count;
    const size_t block_bytes = part.layout_block_bytes;
    const size_t tile_rows   = part.layout_tile_rows;
    for (size_t tile = 0; tile < tile_count; ++tile) {
        uint8_t * tile_payload = payload + tile * tile_bytes;
        std::memcpy(scratch.data(), tile_payload, tile_bytes);
        if (block_bytes == 17) {
            for (size_t block = 0; block < block_count; ++block) {
                for (size_t row = 0; row < tile_rows; ++row) {
                    std::memcpy(tile_payload + (block * tile_rows + row) * 17,
                                scratch.data() + (row * block_count + block) * 17, 17);
                }
            }
        } else {
            for (size_t block = 0; block < block_count; ++block) {
                for (size_t row = 0; row < tile_rows; ++row) {
                    std::memcpy(tile_payload + (block * tile_rows + row) * block_bytes,
                                scratch.data() + (row * block_count + block) * block_bytes, block_bytes);
                }
            }
        }
    }
    return true;
}

struct layer_definition {
    uint32_t                 layer        = 0;
    uint32_t                 expert_count = 0;
    std::vector<source_part> parts;
};

enum class slot_state {
    empty,
    loading,
    resident,
    failed,
};

enum class slot_part_state {
    pending,
    resident,
    failed,
};

struct cache_slot {
    uint64_t                     key         = 0;
    uint64_t                     generation  = 0;
    uint64_t                     last_use    = 0;
    uint32_t                     lease_count = 0;
    bool                         occupied    = false;
    slot_state                   state       = slot_state::empty;
    std::vector<slot_part_state> parts;
    std::string                  error;
};

struct load_task {
    size_t   layer_index = 0;
    uint32_t expert      = 0;
    uint64_t key         = 0;
    size_t   slot        = 0;
    uint64_t generation  = 0;
    size_t   part_index  = 0;
};

struct load_result {
    bool        ok             = false;
    size_t      payload_offset = 0;
    std::string error;
};

enum class staging_bank_state {
    free,
    filling,
    ready,
    copying,
};

struct staging_bank {
    std::unique_ptr<stable_buffer> buffer;
    staging_bank_state             state = staging_bank_state::free;
    load_task                      task;
    load_result                    result;
};

struct ready_upload {
    size_t worker = 0;
    size_t bank   = 0;
};

struct batch_reservation {
    std::vector<size_t>  slots;
    std::vector<uint8_t> resident;
};

}  // namespace

struct expert_cache::shared_state {
    explicit shared_state(expert_cache_config value) : config(value) {}

    ~shared_state() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        work_cv.notify_all();
        state_cv.notify_all();
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        if (transfer_stream) {
            hrx_stream_release(transfer_stream);
        }
    }

    bool initialize(hrx_device_t                                   device,
                    hrx_stream_t                                   input_transfer_stream,
                    const std::vector<expert_cache_layer_source> & input_layers,
                    std::string &                                  error) {
        if (config.slot_count == 0 || config.slot_size == 0 || config.worker_count == 0 || config.worker_count > 64) {
            error = "expert cache: slot_count, slot_size, and worker_count must be non-zero";
            return false;
        }
        if (!is_power_of_two(config.io_alignment) || config.io_alignment < sizeof(void *)) {
            error = "expert cache: io_alignment must be a power of two at least sizeof(void *)";
            return false;
        }
        if (config.slot_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            error = "expert cache: slot_count exceeds the slot-map index type";
            return false;
        }
        if (config.host_slot_count > config.slot_count || config.host_slot_count == config.slot_count) {
            error = "expert cache: host-visible slots must be a proper suffix of the global slot pool";
            return false;
        }
        if (config.transient_slot_count > config.slot_count) {
            error = "expert cache: transient slots exceed the global slot pool";
            return false;
        }
        if (input_layers.empty()) {
            error = "expert cache: no layer sources";
            return false;
        }
        if (!device || !input_transfer_stream) {
            error = "expert cache: production cache requires a device and transfer stream";
            return false;
        }
        transfer_stream = input_transfer_stream;
        hrx_stream_retain(transfer_stream);

        std::vector<std::shared_ptr<backing_file>> files;
        size_t                                     maximum_experts              = 0;
        size_t                                     maximum_part_length          = 0;
        size_t                                     maximum_transform_tile_bytes = 0;
        for (const expert_cache_layer_source & input_layer : input_layers) {
            if (input_layer.expert_count == 0 || input_layer.ranges.empty()) {
                error = "expert cache: every layer must have experts and source ranges";
                return false;
            }
            if (layer_indices.find(input_layer.layer) != layer_indices.end()) {
                error = "expert cache: duplicate layer " + std::to_string(input_layer.layer);
                return false;
            }

            layer_definition layer;
            layer.layer        = input_layer.layer;
            layer.expert_count = input_layer.expert_count;
            std::vector<std::pair<size_t, size_t>> destinations;
            std::vector<std::pair<size_t, size_t>> storage_destinations;
            std::vector<uint32_t>                  planes;
            for (const expert_cache_range & range : input_layer.ranges) {
                if (range.fd < 0 || range.byte_length == 0 || range.expert_stride < range.byte_length) {
                    error = "expert cache: invalid source range for layer " + std::to_string(input_layer.layer);
                    return false;
                }
                if (range.slot_offset > config.slot_size || range.byte_length > config.slot_size - range.slot_offset) {
                    error = "expert cache: source range exceeds slot for layer " + std::to_string(input_layer.layer);
                    return false;
                }
                size_t transform_tile_bytes = 0;
                if (range.payload_layout == expert_cache_payload_layout::native) {
                    if (range.layout_row_count != 0 || range.layout_block_count != 0 ||
                        range.layout_block_bytes != 0 || range.layout_tile_rows != 0) {
                        error = "expert cache: native payload has row-tile metadata for layer " +
                                std::to_string(input_layer.layer);
                        return false;
                    }
                } else if (range.payload_layout != expert_cache_payload_layout::row_tile_block ||
                           range.layout_row_count == 0 || range.layout_block_count == 0 ||
                           range.layout_block_bytes == 0 || range.layout_tile_rows == 0 ||
                           range.layout_row_count % range.layout_tile_rows != 0 ||
                           !checked_mul_size(range.layout_tile_rows, range.layout_block_count,
                                             transform_tile_bytes) ||
                           !checked_mul_size(transform_tile_bytes, range.layout_block_bytes,
                                             transform_tile_bytes)) {
                    error = "expert cache: invalid row-tile payload metadata for layer " +
                            std::to_string(input_layer.layer);
                    return false;
                } else {
                    size_t layout_records;
                    size_t layout_bytes;
                    if (!checked_mul_size(range.layout_row_count, range.layout_block_count, layout_records) ||
                        !checked_mul_size(layout_records, range.layout_block_bytes, layout_bytes) ||
                        layout_bytes != range.byte_length) {
                        error = "expert cache: row-tile payload size mismatch for layer " +
                                std::to_string(input_layer.layer);
                        return false;
                    }
                    maximum_transform_tile_bytes =
                        std::max(maximum_transform_tile_bytes, transform_tile_bytes);
                }
                for (const auto & destination : destinations) {
                    const size_t end = range.slot_offset + range.byte_length;
                    if (range.slot_offset < destination.second && destination.first < end) {
                        error = "expert cache: overlapping slot ranges for layer " + std::to_string(input_layer.layer);
                        return false;
                    }
                }
                const bool has_direct_layout = range.direct_read_length != 0;
                if (!has_direct_layout && range.direct_slot_offset != 0) {
                    error =
                        "expert cache: incomplete direct-final layout for layer " + std::to_string(input_layer.layer);
                    return false;
                }
                size_t storage_offset = range.slot_offset;
                size_t storage_length = range.byte_length;
                if (has_direct_layout) {
                    const size_t payload_offset = static_cast<size_t>(range.file_offset & (config.io_alignment - 1));
                    uint64_t     direct_extent;
                    uint64_t     expected_length;
                    size_t       expected_slot_offset;
                    if (config.slot_size % config.io_alignment != 0 ||
                        range.direct_slot_offset % config.io_alignment != 0 ||
                        range.expert_stride % config.io_alignment != 0 ||
                        !checked_add_size(range.direct_slot_offset, payload_offset, expected_slot_offset) ||
                        expected_slot_offset != range.slot_offset ||
                        !checked_add_u64(payload_offset, range.byte_length, direct_extent) ||
                        !align_up(direct_extent, config.io_alignment, expected_length) ||
                        expected_length != range.direct_read_length || range.direct_slot_offset > config.slot_size ||
                        range.direct_read_length > config.slot_size - range.direct_slot_offset) {
                        error =
                            "expert cache: invalid direct-final layout for layer " + std::to_string(input_layer.layer);
                        return false;
                    }
                    storage_offset = range.direct_slot_offset;
                    storage_length = range.direct_read_length;
                }
                for (const auto & destination : storage_destinations) {
                    const size_t end = storage_offset + storage_length;
                    if (storage_offset < destination.second && destination.first < end) {
                        error = "expert cache: overlapping direct-I/O slot regions for layer " +
                                std::to_string(input_layer.layer);
                        return false;
                    }
                }
                if (std::find(planes.begin(), planes.end(), range.plane) != planes.end()) {
                    error = "expert cache: duplicate plane for layer " + std::to_string(input_layer.layer);
                    return false;
                }
                planes.push_back(range.plane);
                destinations.emplace_back(range.slot_offset, range.slot_offset + range.byte_length);
                storage_destinations.emplace_back(storage_offset, storage_offset + storage_length);

                struct stat stat = {};
                if (fstat(range.fd, &stat) != 0 || stat.st_size < 0) {
                    error = errno_message("expert cache fstat", errno);
                    return false;
                }
                const uint64_t file_size = static_cast<uint64_t>(stat.st_size);
                uint64_t       last_delta;
                uint64_t       last_offset;
                uint64_t       range_end;
                if (!checked_mul_u64(input_layer.expert_count - 1, range.expert_stride, last_delta) ||
                    !checked_add_u64(range.file_offset, last_delta, last_offset) ||
                    !checked_add_u64(last_offset, range.byte_length, range_end) || range_end > file_size ||
                    range_end > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
                    error = "expert cache: source range exceeds file for layer " + std::to_string(input_layer.layer);
                    return false;
                }

                std::shared_ptr<backing_file> file;
                for (const auto & candidate : files) {
                    if (candidate->same_file(stat.st_dev, stat.st_ino)) {
                        file = candidate;
                        break;
                    }
                }
                if (!file) {
                    file = backing_file::open_from_fd(range.fd, config.prefer_direct, stat.st_dev, stat.st_ino,
                                                      file_size, error);
                    if (!file) {
                        return false;
                    }
                    files.push_back(file);
                }
                layer.parts.push_back({ file, range.plane, range.file_offset, range.expert_stride, range.byte_length,
                                        range.slot_offset, range.direct_slot_offset, range.direct_read_length,
                                        range.payload_layout, range.layout_row_count, range.layout_block_count,
                                        range.layout_block_bytes, range.layout_tile_rows });
                maximum_part_length = std::max(maximum_part_length, range.byte_length);
            }
            layer_indices.emplace(layer.layer, layers.size());
            maximum_experts = std::max(maximum_experts, static_cast<size_t>(layer.expert_count));
            layers.push_back(std::move(layer));
        }

        persistent_slot_count = config.slot_count - config.transient_slot_count;
        if (persistent_slot_count < maximum_experts) {
            error = "expert cache: global slot pool must hold the largest layer expert union";
            return false;
        }
        if (maximum_experts > expert_cache_slot_map_capacity) {
            error = "expert cache: layer expert count exceeds the fixed binding header";
            return false;
        }
        if (layers.size() > 1 && persistent_slot_count > maximum_experts) {
            layer_resident_quota = (persistent_slot_count - maximum_experts) / (layers.size() - 1);
        }

        device_slot_count_value = config.slot_count - config.host_slot_count;
        size_t arena_bytes;
        size_t device_arena_bytes;
        size_t host_arena_bytes;
        size_t binding_bytes;
        if (!checked_mul_size(config.slot_count, config.slot_size, arena_bytes) ||
            !checked_mul_size(device_slot_count_value, config.slot_size, device_arena_bytes) ||
            !checked_mul_size(config.host_slot_count, config.slot_size, host_arena_bytes) ||
            !checked_add_size(expert_cache_binding_header_size, config.host_slot_count ? host_arena_bytes : arena_bytes,
                              binding_bytes)) {
            error = "expert cache: stable buffer size overflow";
            return false;
        }
        if (config.host_slot_count) {
            binding = stable_buffer::allocate_hrx_staging(device, binding_bytes, error);
            if (binding) {
                device_binding = stable_buffer::allocate_hrx_device(device, device_arena_bytes, error);
            }
        } else {
            binding = stable_buffer::allocate_hrx_device(device, binding_bytes, error);
        }
        if (!binding || (config.host_slot_count && !device_binding)) {
            return false;
        }
        if (binding->data() && reinterpret_cast<uintptr_t>(binding->data()) % expert_cache_slot_base_alignment != 0) {
            error = "expert cache: binding does not satisfy the slot-base alignment contract";
            binding.reset();
            return false;
        }
        std::fill_n(slot_map_data(), expert_cache_slot_map_capacity, expert_cache_slot_unavailable);
        std::fill_n(cohort_map_data(), expert_cache_cohort_map_capacity, expert_cache_cohort_unavailable);
        slots.resize(config.slot_count);

        uint64_t staging_required_u64;
        if (!checked_add_u64(maximum_part_length, config.io_alignment - 1, staging_required_u64) ||
            !align_up(staging_required_u64, config.io_alignment, staging_required_u64) ||
            staging_required_u64 > std::numeric_limits<size_t>::max()) {
            error = "expert cache: staging slab size overflow";
            return false;
        }
        staging_slab_size = static_cast<size_t>(staging_required_u64);
        try {
            worker_contexts.reserve(config.worker_count);
            for (size_t worker_index = 0; worker_index < config.worker_count; ++worker_index) {
                auto worker = std::make_unique<worker_context>();
                for (auto & bank : worker->banks) {
                    bank.buffer = stable_buffer::allocate_hrx_staging(device, staging_slab_size, error);
                    if (!bank.buffer) {
                        return false;
                    }
                    if (reinterpret_cast<uintptr_t>(bank.buffer->data()) % config.io_alignment != 0) {
                        error = "expert cache: staging slab does not satisfy direct-I/O alignment";
                        return false;
                    }
                }
                worker->transform_scratch.resize(maximum_transform_tile_bytes);
                worker_contexts.push_back(std::move(worker));
            }
        } catch (const std::exception & exception) {
            error = std::string("expert cache: failed to allocate staging slabs: ") + exception.what();
            return false;
        }

        try {
            workers.reserve(config.worker_count);
            for (size_t index = 0; index < config.worker_count; ++index) {
                workers.emplace_back([this, index]() { worker_loop(index); });
            }
        } catch (const std::exception & exception) {
            error = std::string("expert cache: failed to start I/O worker: ") + exception.what();
            return false;
        }
        return true;
    }

    bool unique_ids(size_t                  layer_index,
                    const uint32_t *        ids,
                    size_t                  count,
                    std::vector<uint32_t> & unique,
                    std::vector<size_t> &   assignment_counts,
                    std::string &           error) const {
        if (!ids || count == 0) {
            error = "expert cache: an acquisition must contain at least one expert";
            return false;
        }
        const layer_definition & layer = layers[layer_index];
        std::vector<size_t>      counts(layer.expert_count, 0);
        unique.reserve(std::min(count, static_cast<size_t>(layer.expert_count)));
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] >= layer.expert_count) {
                error = "expert cache: expert " + std::to_string(ids[i]) + " is outside layer " +
                        std::to_string(layer.layer);
                return false;
            }
            if (counts[ids[i]] == 0) {
                unique.push_back(ids[i]);
            }
            ++counts[ids[i]];
        }
        assignment_counts.reserve(unique.size());
        for (uint32_t expert : unique) {
            assignment_counts.push_back(counts[expert]);
        }
        return true;
    }

    bool reserve_batch_locked(size_t                        layer_index,
                              const std::vector<uint32_t> & unique,
                              const std::vector<size_t> &   assignment_counts,
                              bool                          pin,
                              bool                          allow_transient,
                              size_t                        load_chunk_size,
                              batch_reservation &           reservation,
                              std::string &                 error) {
        if (assignment_counts.size() != unique.size()) {
            error = "expert cache: assignment counts do not match the expert union";
            return false;
        }

        const layer_definition & layer = layers[layer_index];
        std::vector<uint8_t>     protected_slots(persistent_slot_count, 0);
        std::vector<size_t>      missing;
        reservation.slots.assign(unique.size(), std::numeric_limits<size_t>::max());
        reservation.resident.assign(unique.size(), 0);

        for (size_t i = 0; i < unique.size(); ++i) {
            const uint32_t expert = unique[i];
            const uint64_t key    = composite_key(layer.layer, expert);
            const auto     entry  = entries.find(key);
            if (entry == entries.end()) {
                missing.push_back(i);
                continue;
            }
            cache_slot & slot = slots[entry->second];
            if (slot.state == slot_state::failed && slot.lease_count == 0) {
                entries.erase(entry);
                slot.occupied = false;
                slot.state    = slot_state::empty;
                slot.error.clear();
                missing.push_back(i);
                continue;
            }
            protected_slots[entry->second] = 1;
            reservation.slots[i]           = entry->second;
            if (pin) {
                ++slot.lease_count;
            }
            if (slot.state == slot_state::resident) {
                reservation.resident[i] = 1;
                slot.last_use = ++use_clock;
            }
        }

        // A fully admitted batch cannot require placement or eviction.
        // Skip rebuilding the persistent-slot victim list after cache warm-up.
        if (missing.empty()) {
            return true;
        }

        std::vector<size_t> candidates;
        candidates.reserve(persistent_slot_count);
        for (size_t i = 0; i < persistent_slot_count; ++i) {
            const cache_slot & slot = slots[i];
            if (!protected_slots[i] && slot.lease_count == 0 &&
                (slot.state == slot_state::empty || slot.state == slot_state::failed)) {
                candidates.push_back(i);
            }
        }
        const size_t free_persistent_slots = candidates.size();

        std::vector<size_t> resident_candidates;
        std::vector<size_t> resident_counts(layers.size(), 0);
        for (size_t i = 0; i < persistent_slot_count; ++i) {
            const cache_slot & slot = slots[i];
            if (!protected_slots[i] && slot.lease_count == 0 && slot.state == slot_state::resident) {
                resident_candidates.push_back(i);
                const auto resident_layer = layer_indices.find(static_cast<uint32_t>(slot.key >> 32));
                if (resident_layer != layer_indices.end()) {
                    ++resident_counts[resident_layer->second];
                }
            }
        }
        std::sort(resident_candidates.begin(), resident_candidates.end(),
                  [this](size_t lhs, size_t rhs) { return slots[lhs].last_use < slots[rhs].last_use; });
        std::vector<size_t> quota_candidates;
        quota_candidates.reserve(resident_candidates.size());
        for (size_t slot_index : resident_candidates) {
            const auto resident_layer = layer_indices.find(static_cast<uint32_t>(slots[slot_index].key >> 32));
            if (resident_layer == layer_indices.end() || resident_layer->second == layer_index ||
                resident_counts[resident_layer->second] > layer_resident_quota) {
                candidates.push_back(slot_index);
                if (resident_layer != layer_indices.end() && resident_layer->second != layer_index) {
                    --resident_counts[resident_layer->second];
                }
            } else {
                quota_candidates.push_back(slot_index);
            }
        }
        candidates.insert(candidates.end(), quota_candidates.begin(), quota_candidates.end());

        std::vector<size_t> persistent_missing;
        std::vector<size_t> transient_missing;
        persistent_missing.reserve(missing.size());
        transient_missing.reserve(missing.size());
        const bool has_repeated_assignments =
            std::any_of(assignment_counts.begin(), assignment_counts.end(), [](size_t count) { return count >= 2; });
        if (!allow_transient || config.transient_slot_count == 0 || !has_repeated_assignments) {
            persistent_missing = missing;
        } else {
            std::vector<size_t> repeated;
            std::vector<size_t> single_use;
            for (size_t missing_index : missing) {
                if (assignment_counts[missing_index] >= 2) {
                    repeated.push_back(missing_index);
                } else {
                    single_use.push_back(missing_index);
                }
            }
            persistent_missing.insert(persistent_missing.end(), repeated.begin(), repeated.end());
            const size_t free_for_single_use =
                free_persistent_slots > repeated.size() ? free_persistent_slots - repeated.size() : 0;
            const size_t admitted_single_use = std::min(free_for_single_use, single_use.size());
            persistent_missing.insert(persistent_missing.end(), single_use.begin(),
                                      single_use.begin() + admitted_single_use);
            const size_t transient_count =
                std::min(config.transient_slot_count, single_use.size() - admitted_single_use);
            transient_missing.insert(transient_missing.end(), single_use.begin() + admitted_single_use,
                                     single_use.begin() + admitted_single_use + transient_count);
            persistent_missing.insert(persistent_missing.end(),
                                      single_use.begin() + admitted_single_use + transient_count, single_use.end());
        }

        if (candidates.size() < persistent_missing.size()) {
            for (size_t slot_index : reservation.slots) {
                if (pin && slot_index != std::numeric_limits<size_t>::max()) {
                    --slots[slot_index].lease_count;
                }
            }
            reservation = {};
            error       = "expert cache: global slots are still leased by an unfinished layer";
            return false;
        }

        std::vector<load_task> new_tasks;
        new_tasks.reserve(missing.size());
        auto assign_missing = [&](size_t unique_index, size_t slot_index, bool persistent) {
            const uint32_t expert = unique[unique_index];
            const uint64_t key    = composite_key(layer.layer, expert);
            cache_slot &   slot   = slots[slot_index];
            if (slot.occupied) {
                entries.erase(slot.key);
            }
            slot.key      = key;
            slot.occupied = true;
            slot.state    = slot_state::loading;
            slot.parts.assign(layer.parts.size(), slot_part_state::pending);
            slot.error.clear();
            slot.last_use    = ++use_clock;
            slot.lease_count = pin ? 1u : 0u;
            ++slot.generation;
            if (persistent) {
                entries[key] = slot_index;
            }
            reservation.slots[unique_index] = slot_index;
            new_tasks.push_back({ layer_index, expert, key, slot_index, slot.generation, 0 });
        };
        for (size_t i = 0; i < persistent_missing.size(); ++i) {
            assign_missing(persistent_missing[i], candidates[i], true);
        }
        for (size_t i = 0; i < transient_missing.size(); ++i) {
            assign_missing(transient_missing[i], persistent_slot_count + i, false);
        }
        if (!missing.empty()) {
            // Zero preserves plane-major order.
            // A nonzero chunk lets the graph consume one expert interval while workers fill the next.
            if (load_chunk_size == 0) {
                for (size_t part_index = 0; part_index < layer.parts.size(); ++part_index) {
                    for (load_task task : new_tasks) {
                        task.part_index = part_index;
                        queue.push_back(task);
                        ++inflight;
                    }
                }
            } else {
                for (size_t expert_begin = 0; expert_begin < layer.expert_count;
                     expert_begin += load_chunk_size) {
                    const size_t expert_end = std::min<size_t>(
                        layer.expert_count, expert_begin + load_chunk_size);
                    for (size_t part_index = 0; part_index < layer.parts.size(); ++part_index) {
                        for (load_task task : new_tasks) {
                            if (task.expert < expert_begin || task.expert >= expert_end) {
                                continue;
                            }
                            task.part_index = part_index;
                            queue.push_back(task);
                            ++inflight;
                        }
                    }
                }
            }
            work_cv.notify_all();
        }
        return true;
    }

    bool is_host_final_task(const load_task & task) const {
        return config.host_slot_count != 0 && task.slot >= device_slot_count_value &&
               task.layer_index < layers.size() && task.part_index < layers[task.layer_index].parts.size() &&
               layers[task.layer_index].parts[task.part_index].direct_read_length != 0;
    }

    load_result load_staged_part(size_t worker_index, size_t bank_index, const load_task & task) {
        load_result              result;
        worker_context &         worker = *worker_contexts[worker_index];
        staging_bank &           bank   = worker.banks[bank_index];
        const layer_definition & layer  = layers[task.layer_index];
        if (task.part_index >= layer.parts.size()) {
            result.error = "expert cache: source plane index overflow";
            return result;
        }
        const source_part & part = layer.parts[task.part_index];
        uint64_t            delta;
        uint64_t            offset;
        if (!checked_mul_u64(task.expert, part.expert_stride, delta) ||
            !checked_add_u64(part.file_offset, delta, offset)) {
            result.error = "expert cache: source offset overflow";
            return result;
        }
        if (!part.file->read(worker.ring, config.io_alignment, offset, part.byte_length, bank.buffer->data(),
                             bank.buffer->size(), result.payload_offset, result.error)) {
            return result;
        }
        if (!transform_payload_in_place(part, bank.buffer->data() + result.payload_offset, part.byte_length,
                                        worker.transform_scratch, result.error)) {
            return result;
        }
        result.ok = true;
        return result;
    }

    load_result load_host_final_part(size_t worker_index, const load_task & task) {
        load_result              result;
        const layer_definition & layer = layers[task.layer_index];
        if (task.part_index >= layer.parts.size()) {
            result.error = "expert cache: source plane index overflow";
            return result;
        }
        const source_part & part      = layer.parts[task.part_index];
        const size_t        host_slot = task.slot - device_slot_count_value;
        size_t              slot_offset;
        size_t              destination_offset;
        if (!part.direct_read_length || !binding->data() ||
            !checked_mul_size(host_slot, config.slot_size, slot_offset) ||
            !checked_add_size(expert_cache_binding_header_size, slot_offset, destination_offset) ||
            !checked_add_size(destination_offset, part.direct_slot_offset, destination_offset) ||
            destination_offset > binding->size() || part.direct_read_length > binding->size() - destination_offset ||
            reinterpret_cast<uintptr_t>(binding->data() + destination_offset) % config.io_alignment != 0) {
            result.error = "expert cache: direct-final destination exceeds its mapped host slot";
            return result;
        }
        uint64_t delta;
        uint64_t offset;
        if (!checked_mul_u64(task.expert, part.expert_stride, delta) ||
            !checked_add_u64(part.file_offset, delta, offset)) {
            result.error = "expert cache: source offset overflow";
            return result;
        }
        const size_t payload_offset = part.slot_offset - part.direct_slot_offset;
        if (!part.file->read_final(worker_contexts[worker_index]->ring, config.io_alignment, offset, part.byte_length,
                                   binding->data() + destination_offset, part.direct_read_length, payload_offset,
                                   result.error)) {
            return result;
        }
        if (!transform_payload_in_place(part, binding->data() + destination_offset + payload_offset,
                                        part.byte_length, worker_contexts[worker_index]->transform_scratch,
                                        result.error)) {
            return result;
        }
        result.ok = true;
        return result;
    }

    void worker_loop(size_t worker_index) {
        for (;;) {
            load_task task;
            size_t    bank_index = staging_banks_per_worker;
            bool      host_final = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work_cv.wait(lock, [this, worker_index]() {
                    if (shutdown || queue.empty()) {
                        return shutdown;
                    }
                    const auto & banks     = worker_contexts[worker_index]->banks;
                    const bool   free_bank = std::any_of(banks.begin(), banks.end(), [](const staging_bank & bank) {
                        return bank.state == staging_bank_state::free;
                    });
                    if (free_bank || is_host_final_task(queue.front())) {
                        return true;
                    }
                    const size_t front_part = queue.front().part_index;
                    return std::any_of(queue.begin(), queue.end(), [&](const load_task & candidate) {
                        return candidate.part_index == front_part && is_host_final_task(candidate);
                    });
                });
                if (shutdown) {
                    return;
                }
                auto &     banks     = worker_contexts[worker_index]->banks;
                const auto free_bank = std::find_if(banks.begin(), banks.end(), [](const staging_bank & bank) {
                    return bank.state == staging_bank_state::free;
                });
                if (queue.empty()) {
                    continue;
                }
                auto task_it = queue.begin();
                if (free_bank == banks.end() && !is_host_final_task(*task_it)) {
                    const size_t front_part = task_it->part_index;
                    task_it = std::find_if(queue.begin(), queue.end(), [&](const load_task & candidate) {
                        return candidate.part_index == front_part && is_host_final_task(candidate);
                    });
                    if (task_it == queue.end()) {
                        continue;
                    }
                }
                task       = *task_it;
                host_final = is_host_final_task(task);
                queue.erase(task_it);
                if (!host_final) {
                    if (free_bank == banks.end()) {
                        continue;
                    }
                    bank_index       = static_cast<size_t>(free_bank - banks.begin());
                    free_bank->state = staging_bank_state::filling;
                    free_bank->task  = task;
                }
            }

            load_result result = host_final ? load_host_final_part(worker_index, task) :
                                              load_staged_part(worker_index, bank_index, task);
            {
                std::lock_guard<std::mutex> lock(mutex);
                cache_slot &                slot = slots[task.slot];
                const bool current = slot.occupied && slot.key == task.key && slot.generation == task.generation &&
                                     task.part_index < slot.parts.size();
                if (host_final) {
                    if (current && result.ok && slot.state != slot_state::failed) {
                        slot.parts[task.part_index] = slot_part_state::resident;
                        slot.last_use               = ++use_clock;
                        if (slot.state == slot_state::loading &&
                            std::all_of(slot.parts.begin(), slot.parts.end(),
                                        [](slot_part_state state) { return state == slot_part_state::resident; })) {
                            slot.state = slot_state::resident;
                        }
                    } else if (current && slot.state != slot_state::failed) {
                        slot.parts[task.part_index] = slot_part_state::failed;
                        slot.state                  = slot_state::failed;
                        slot.error                  = std::move(result.error);
                    }
                    --inflight;
                } else {
                    staging_bank & bank = worker_contexts[worker_index]->banks[bank_index];
                    if (current && result.ok) {
                        bank.result = std::move(result);
                        bank.state  = staging_bank_state::ready;
                        ready_queue.push_back({ worker_index, bank_index });
                    } else {
                        if (current && slot.state != slot_state::failed) {
                            slot.parts[task.part_index] = slot_part_state::failed;
                            slot.state                  = slot_state::failed;
                            slot.error                  = std::move(result.error);
                        }
                        bank.result = {};
                        bank.state  = staging_bank_state::free;
                        --inflight;
                    }
                }
            }
            work_cv.notify_all();
            state_cv.notify_all();
        }
    }

    bool publish_ready_batch(std::string & error) {
        std::vector<ready_upload> batch;
        {
            std::lock_guard<std::mutex> lock(mutex);
            std::array<uint8_t, 64>     selected_workers = {};
            const size_t                ready_count      = ready_queue.size();
            for (size_t ready_index = 0; ready_index < ready_count; ++ready_index) {
                const ready_upload upload = ready_queue.front();
                ready_queue.pop_front();
                staging_bank & bank    = worker_contexts[upload.worker]->banks[upload.bank];
                cache_slot &   slot    = slots[bank.task.slot];
                const bool     current = bank.state == staging_bank_state::ready && slot.occupied &&
                                     slot.key == bank.task.key && slot.generation == bank.task.generation &&
                                     bank.task.part_index < slot.parts.size() && slot.state != slot_state::failed;
                if (current) {
                    if (selected_workers[upload.worker]) {
                        ready_queue.push_back(upload);
                        continue;
                    }
                    selected_workers[upload.worker] = 1;
                    bank.state                      = staging_bank_state::copying;
                    batch.push_back(upload);
                } else {
                    bank.result = {};
                    bank.state  = staging_bank_state::free;
                    --inflight;
                }
            }
        }
        work_cv.notify_all();
        state_cv.notify_all();
        if (batch.empty()) {
            return true;
        }

        bool        ok              = true;
        size_t      device_enqueued = 0;
        std::string upload_error;
        for (const ready_upload & upload : batch) {
            staging_bank &           bank        = worker_contexts[upload.worker]->banks[upload.bank];
            const load_task &        task        = bank.task;
            const layer_definition & layer       = layers[task.layer_index];
            const source_part &      part        = layer.parts[task.part_index];
            const bool               device_slot = config.host_slot_count == 0 || task.slot < device_slot_count_value;
            const size_t             tier_slot   = device_slot ? task.slot : task.slot - device_slot_count_value;
            stable_buffer * destination = device_slot && config.host_slot_count ? device_binding.get() : binding.get();
            const size_t    slot_base   = device_slot && config.host_slot_count ? 0 : expert_cache_binding_header_size;
            size_t          destination_offset;
            size_t          slot_offset;
            if (!checked_mul_size(tier_slot, config.slot_size, slot_offset) ||
                !checked_add_size(slot_base, slot_offset, destination_offset) ||
                !checked_add_size(destination_offset, part.slot_offset, destination_offset) ||
                bank.result.payload_offset > bank.buffer->size() ||
                part.byte_length > bank.buffer->size() - bank.result.payload_offset || !destination ||
                destination_offset > destination->size() ||
                part.byte_length > destination->size() - destination_offset) {
                upload_error = "expert cache: staged upload exceeds binding bounds";
                ok           = false;
                break;
            }

            if (!device_slot) {
                if (!destination->data()) {
                    upload_error = "expert cache: host-visible payload destination is not mapped";
                    ok           = false;
                    break;
                }
                std::memcpy(destination->data() + destination_offset, bank.buffer->data() + bank.result.payload_offset,
                            part.byte_length);
            } else if (!hrx_ok(
                           hrx_stream_copy_buffer(transfer_stream, bank.buffer->buffer(), bank.result.payload_offset,
                                                  destination->buffer(), destination_offset, part.byte_length),
                           "expert cache payload upload", upload_error)) {
                ok = false;
                break;
            }
            device_enqueued += device_slot ? 1 : 0;
        }
        if (config.host_slot_count) {
            std::atomic_thread_fence(std::memory_order_release);
        }
        if (device_enqueued > 0 &&
            !hrx_ok(hrx_stream_synchronize(transfer_stream), "expert cache payload upload wait", upload_error)) {
            ok = false;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const ready_upload & upload : batch) {
                staging_bank &  bank    = worker_contexts[upload.worker]->banks[upload.bank];
                const load_task task    = bank.task;
                cache_slot &    slot    = slots[task.slot];
                const bool      current = slot.occupied && slot.key == task.key && slot.generation == task.generation &&
                                     task.part_index < slot.parts.size();
                if (ok && current && slot.state != slot_state::failed) {
                    slot.parts[task.part_index] = slot_part_state::resident;
                    slot.last_use               = ++use_clock;
                    if (slot.state == slot_state::loading &&
                        std::all_of(slot.parts.begin(), slot.parts.end(),
                                    [](slot_part_state state) { return state == slot_part_state::resident; })) {
                        slot.state = slot_state::resident;
                    }
                } else if (current && slot.state != slot_state::failed) {
                    slot.parts[task.part_index] = slot_part_state::failed;
                    slot.state                  = slot_state::failed;
                    slot.error = upload_error.empty() ? "expert cache: payload upload failed" : upload_error;
                }
                bank.result = {};
                bank.state  = staging_bank_state::free;
                --inflight;
            }
        }
        work_cv.notify_all();
        state_cv.notify_all();
        if (!ok) {
            error = upload_error.empty() ? "expert cache: payload upload failed" : upload_error;
        }
        return ok;
    }

    bool publish_header(hrx_stream_t graph_stream, std::string & error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!layer_scope_active) {
            error = "expert cache: no active layer header to publish";
            return false;
        }
        if (header_published) {
            return true;
        }
        if (!graph_stream) {
            error = "expert cache: header publication requires a graph stream";
            return false;
        }
        if (config.host_slot_count) {
            std::atomic_thread_fence(std::memory_order_release);
        } else {
            if (!hrx_ok(hrx_stream_update_buffer(graph_stream, header_data(), expert_cache_binding_header_size,
                                                 binding->buffer(), 0),
                        "expert cache header update", error)) {
                return false;
            }
        }
        header_published = true;
        return true;
    }

    bool wait_for_planes(uint32_t                    layer,
                         const std::vector<size_t> & leased_slots,
                         const std::vector<uint32_t> & leased_experts,
                         uint32_t                    expert_begin,
                         uint32_t                    expert_end,
                         const uint32_t *            planes,
                         size_t                      plane_count,
                         std::string &               error) {
        if (!planes || plane_count == 0) {
            error = "expert cache: a plane wait must name at least one plane";
            return false;
        }
        if (leased_slots.size() != leased_experts.size() || expert_begin >= expert_end) {
            error = "expert cache: invalid expert range for plane wait";
            return false;
        }
        const auto layer_it = layer_indices.find(layer);
        if (layer_it == layer_indices.end()) {
            error = "expert cache: unknown layer " + std::to_string(layer);
            return false;
        }
        const layer_definition & definition = layers[layer_it->second];
        std::vector<size_t>      part_indices;
        part_indices.reserve(plane_count);
        for (size_t i = 0; i < plane_count; ++i) {
            const auto part = std::find_if(definition.parts.begin(), definition.parts.end(),
                                           [&](const source_part & candidate) { return candidate.plane == planes[i]; });
            if (part == definition.parts.end()) {
                error =
                    "expert cache: unknown plane " + std::to_string(planes[i]) + " for layer " + std::to_string(layer);
                return false;
            }
            const size_t part_index = static_cast<size_t>(part - definition.parts.begin());
            if (std::find(part_indices.begin(), part_indices.end(), part_index) == part_indices.end()) {
                part_indices.push_back(part_index);
            }
        }

        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            if (!layer_scope_active || active_layer != layer) {
                error = "expert cache: layer lease is no longer active";
                return false;
            }
            if (!header_published) {
                error = "expert cache: layer header was not published before its payload wait";
                return false;
            }
            bool complete = true;
            for (size_t lease_index = 0; lease_index < leased_slots.size(); ++lease_index) {
                if (leased_experts[lease_index] < expert_begin || leased_experts[lease_index] >= expert_end) {
                    continue;
                }
                const size_t slot_index = leased_slots[lease_index];
                const cache_slot & slot = slots[slot_index];
                if (slot.state == slot_state::failed) {
                    error = slot.error.empty() ? "expert cache: expert plane load failed" : slot.error;
                    return false;
                }
                for (size_t part_index : part_indices) {
                    if (part_index >= slot.parts.size() || slot.parts[part_index] == slot_part_state::failed) {
                        error = slot.error.empty() ? "expert cache: expert plane load failed" : slot.error;
                        return false;
                    }
                    complete = complete && slot.parts[part_index] == slot_part_state::resident;
                }
            }
            if (complete) {
                return true;
            }
            if (!ready_queue.empty()) {
                lock.unlock();
                if (!publish_ready_batch(error)) {
                    return false;
                }
                continue;
            }
            state_cv.wait(lock);
        }
    }

    void release_layer(uint32_t layer, const std::vector<size_t> & leased_slots) {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            const bool                   transient_pending =
                std::any_of(leased_slots.begin(), leased_slots.end(), [&](size_t slot_index) {
                    return slot_index >= persistent_slot_count &&
                           std::find(slots[slot_index].parts.begin(), slots[slot_index].parts.end(),
                                     slot_part_state::pending) != slots[slot_index].parts.end();
                });
            if (!transient_pending) {
                break;
            }
            if (!ready_queue.empty()) {
                lock.unlock();
                std::string ignored;
                publish_ready_batch(ignored);
                continue;
            }
            state_cv.wait(lock);
        }

        std::unique_lock<std::mutex> lock(mutex);
        if (!layer_scope_active || active_layer != layer) {
            return;
        }
        for (size_t slot_index : leased_slots) {
            cache_slot & slot = slots[slot_index];
            if (slot.lease_count > 0) {
                --slot.lease_count;
            }
            if (slot_index < persistent_slot_count) {
                slot.last_use = ++use_clock;
            } else {
                slot.key      = 0;
                slot.occupied = false;
                slot.state    = slot_state::empty;
                slot.parts.clear();
                slot.error.clear();
            }
        }
        std::fill_n(slot_map_data(), expert_cache_slot_map_capacity, expert_cache_slot_unavailable);
        std::fill_n(cohort_map_data(), expert_cache_cohort_map_capacity, expert_cache_cohort_unavailable);
        layer_scope_active = false;
        header_published   = false;
        state_cv.notify_all();
    }

    struct worker_context {
        io_uring_reader             ring;
        std::array<staging_bank, 2> banks;
        std::vector<uint8_t>        transform_scratch;
    };

    static constexpr size_t staging_banks_per_worker = 2;

    uint8_t * header_data() const {
        return binding->data() ? binding->data() : const_cast<uint8_t *>(header_mirror.data());
    }

    int32_t * slot_map_data() const {
        return reinterpret_cast<int32_t *>(header_data() + expert_cache_slot_map_offset);
    }

    int32_t * cohort_map_data() const {
        return reinterpret_cast<int32_t *>(header_data() + expert_cache_cohort_map_offset);
    }

    uint8_t * slot_base_data() const {
        return binding->data() ? binding->data() + expert_cache_binding_header_size : nullptr;
    }

    expert_cache_config                                   config;
    std::vector<layer_definition>                         layers;
    std::unordered_map<uint32_t, size_t>                  layer_indices;
    std::unique_ptr<stable_buffer>                        binding;
    std::unique_ptr<stable_buffer>                        device_binding;
    std::array<uint8_t, expert_cache_binding_header_size> header_mirror     = {};
    hrx_stream_t                                          transfer_stream   = nullptr;
    size_t                                                staging_slab_size = 0;

    mutable std::mutex                           mutex;
    std::condition_variable                      state_cv;
    std::condition_variable                      work_cv;
    std::deque<load_task>                        queue;
    std::deque<ready_upload>                     ready_queue;
    std::vector<cache_slot>                      slots;
    std::unordered_map<uint64_t, size_t>         entries;
    std::vector<std::thread>                     workers;
    std::vector<std::unique_ptr<worker_context>> worker_contexts;
    size_t                                       inflight                = 0;
    uint64_t                                     use_clock               = 0;
    size_t                                       persistent_slot_count   = 0;
    size_t                                       device_slot_count_value = 0;
    size_t                                       layer_resident_quota    = 0;
    bool                                         shutdown           = false;
    bool                                         layer_scope_active = false;
    bool                                         header_published   = false;
    uint32_t                                     active_layer       = 0;
};

expert_cache::expert_cache(std::shared_ptr<shared_state> state) : state_(std::move(state)) {}

expert_cache::~expert_cache() = default;

std::unique_ptr<expert_cache> expert_cache::create(hrx_device_t                                   device,
                                                   hrx_stream_t                                   transfer_stream,
                                                   const expert_cache_config &                    config,
                                                   const std::vector<expert_cache_layer_source> & layers,
                                                   std::string &                                  error) {
    try {
        auto state = std::make_shared<shared_state>(config);
        if (!state->initialize(device, transfer_stream, layers, error)) {
            return nullptr;
        }
        return std::unique_ptr<expert_cache>(new expert_cache(std::move(state)));
    } catch (const std::exception & exception) {
        error = std::string("expert cache: initialization failed: ") + exception.what();
        return nullptr;
    }
}

std::unique_ptr<expert_cache_layer_lease> expert_cache::begin_layer(uint32_t         layer,
                                                                    const uint32_t * expert_ids,
                                                                    size_t           expert_id_count,
                                                                    size_t           load_chunk_size,
                                                                    std::string &    error) {
    error.clear();
    const auto layer_it = state_->layer_indices.find(layer);
    if (layer_it == state_->layer_indices.end()) {
        error = "expert cache: unknown layer " + std::to_string(layer);
        return nullptr;
    }
    std::vector<uint32_t> unique;
    std::vector<size_t>   assignment_counts;
    if (!state_->unique_ids(layer_it->second, expert_ids, expert_id_count, unique, assignment_counts, error)) {
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(state_->mutex);
    if (state_->layer_scope_active) {
        error = "expert cache: the previous layer lease must complete before another acquisition";
        return nullptr;
    }
    state_->layer_scope_active = true;
    state_->active_layer       = layer;
    state_->header_published   = false;

    batch_reservation reservation;
    if (!state_->reserve_batch_locked(
            layer_it->second, unique, assignment_counts, true, true,
            load_chunk_size, reservation, error)) {
        state_->layer_scope_active = false;
        return nullptr;
    }

    auto * slot_map         = state_->slot_map_data();
    auto * cohort_map       = state_->cohort_map_data();
    std::fill_n(slot_map, expert_cache_slot_map_capacity, expert_cache_slot_unavailable);
    std::fill_n(cohort_map, expert_cache_cohort_map_capacity, expert_cache_cohort_unavailable);
    for (size_t i = 0; i < unique.size(); ++i) {
        const uint32_t expert = unique[i];
        if (reservation.slots[i] == std::numeric_limits<size_t>::max()) {
            error = "expert cache: load reservation disappeared";
            break;
        }
        slot_map[expert]    = static_cast<int32_t>(reservation.slots[i]);
        const bool resident = reservation.resident[i] != 0;
        uint32_t cohort = resident ? static_cast<uint32_t>(expert_cache_cohort_resident) :
                                     static_cast<uint32_t>(expert_cache_cohort_missing);
        cohort_map[expert] = static_cast<int32_t>(cohort);
    }
    if (!error.empty()) {
        for (size_t slot_index : reservation.slots) {
            if (slot_index == std::numeric_limits<size_t>::max()) {
                continue;
            }
            cache_slot & slot = state_->slots[slot_index];
            if (slot.lease_count > 0) {
                --slot.lease_count;
            }
        }
        std::fill_n(slot_map, expert_cache_slot_map_capacity, expert_cache_slot_unavailable);
        std::fill_n(cohort_map, expert_cache_cohort_map_capacity, expert_cache_cohort_unavailable);
        state_->layer_scope_active = false;
        return nullptr;
    }
    return std::unique_ptr<expert_cache_layer_lease>(new expert_cache_layer_lease(
        state_, layer, std::move(reservation.slots), std::move(unique)));
}

hrx_buffer_t expert_cache::binding_buffer(const char * storage_binding) const {
    if (!state_->config.host_slot_count || !storage_binding || std::strcmp(storage_binding, "host_cache") == 0) {
        return state_->binding->buffer();
    }
    if (std::strcmp(storage_binding, "device_cache") == 0) {
        return state_->device_binding ? state_->device_binding->buffer() : nullptr;
    }
    return nullptr;
}

size_t expert_cache::binding_size(const char * storage_binding) const {
    if (!state_->config.host_slot_count || !storage_binding || std::strcmp(storage_binding, "host_cache") == 0) {
        return state_->binding->size();
    }
    if (std::strcmp(storage_binding, "device_cache") == 0) {
        return state_->device_binding ? state_->device_binding->size() : 0;
    }
    return 0;
}

expert_cache_group_id expert_cache::group_id() const {
    return state_->config.group;
}

bool expert_cache::resolve_attachment(const expert_cache_source_attachment & attachment,
                                      expert_cache_attachment_layout &       layout,
                                      std::string &                          error) const {
    if (attachment.group != state_->config.group) {
        error = "expert cache: source attachment belongs to another group";
        return false;
    }
    const auto layer = state_->layer_indices.find(attachment.layer);
    if (layer == state_->layer_indices.end()) {
        error = "expert cache: source attachment has an unknown layer";
        return false;
    }
    for (const source_part & part : state_->layers[layer->second].parts) {
        if (part.plane == attachment.plane) {
            layout.slot_base_offset        = expert_cache_binding_header_size;
            layout.slot_count              = state_->config.slot_count;
            layout.slot_stride             = state_->config.slot_size;
            layout.plane_offset            = part.slot_offset;
            layout.plane_length            = part.byte_length;
            layout.device_slot_base_offset = state_->config.host_slot_count ? 0 : expert_cache_binding_header_size;
            layout.device_slot_count       = state_->device_slot_count_value;
            layout.host_slot_base_offset   = expert_cache_binding_header_size;
            layout.host_slot_count         = state_->config.host_slot_count;
            return true;
        }
    }
    error = "expert cache: source attachment has an unknown plane";
    return false;
}

expert_cache_layer_lease::expert_cache_layer_lease(std::shared_ptr<expert_cache::shared_state> state,
                                                   uint32_t                                    layer,
                                                   std::vector<size_t>                         slots,
                                                   std::vector<uint32_t>                       experts) :
    state_(std::move(state)),
    layer_(layer),
    slots_(std::move(slots)),
    experts_(std::move(experts)) {}

expert_cache_layer_lease::~expert_cache_layer_lease() {
    if (state_) {
        state_->release_layer(layer_, slots_);
    }
}

bool expert_cache_layer_lease::wait_for_planes(const uint32_t * planes, size_t plane_count, std::string & error) {
    return wait_for_planes(0, 0xffffffffu, planes, plane_count, error);
}

bool expert_cache_layer_lease::wait_for_planes(uint32_t expert_begin, uint32_t expert_end,
                                               const uint32_t * planes, size_t plane_count,
                                               std::string & error) {
    if (!state_) {
        error = "expert cache: layer lease is no longer active";
        return false;
    }
    return state_->wait_for_planes(
        layer_, slots_, experts_, expert_begin, expert_end, planes, plane_count, error);
}

bool expert_cache_layer_lease::publish_header(hrx_stream_t graph_stream, std::string & error) {
    if (!state_) {
        error = "expert cache: layer lease is no longer active";
        return false;
    }
    return state_->publish_header(graph_stream, error);
}

}  // namespace ggml::hrx
