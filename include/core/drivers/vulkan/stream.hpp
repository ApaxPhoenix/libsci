#pragma once

#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/batch.hpp"
#include "core/drivers/vulkan/buffer.hpp"
#include "core/drivers/vulkan/region.hpp"
#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    struct Context;
    class Pipeline;

    // Records compute dispatches into a ring of command buffers and submits them asynchronously.
    // A submission costs a GPU round trip (a few hundred microseconds on Apple GPUs), so work is
    // batched: a batch goes out when someone waits, when it is full, or after a fixed number of
    // dispatches. Consecutive dispatches are ordered by barriers, so later work always sees earlier
    // results; the host only blocks in wait().
    class Stream {
    public:
        Stream(const Context& context, Allocator& allocator);
        ~Stream();

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        // Scratch memory that lives until the batch being recorded completes: small argument blocks
        // going in, small results (reduction partials) coming out.
        [[nodiscard]] Region reserve(size bytes);

        // `work` estimates the dispatch's cost (see device.cpp); a batch is submitted once it carries
        // enough work, since macOS aborts command buffers that hold the GPU for too long.
        void record(const Pipeline& pipeline, const void* constants, uint64 work, uint32 columns, uint32 rows = 1, uint32 layers = 1);

        // Submits the recorded batch without waiting for it.
        void flush();

        // Submits and blocks until every batch has finished, after which the host may touch any buffer.
        void wait();

        // Hands a buffer back to the allocator once the GPU can no longer be using it.
        void retire(const Buffer& buffer);

    private:
        static constexpr size depth = 4;

        void begin();

        // Waits for batches[index] to finish, then hands its retired buffers back to the allocator.
        void complete(size index);

        const Context& context;
        Allocator& allocator;
        VkCommandPool pool{VK_NULL_HANDLE};
        Batch batches[depth]{};
        size current{0};
    };

}
