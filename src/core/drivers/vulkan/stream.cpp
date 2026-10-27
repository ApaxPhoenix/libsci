#include "core/drivers/vulkan/stream.hpp"
#include "core/assert.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/drivers/vulkan/pipeline.hpp"
#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/batch.hpp"
#include "core/drivers/vulkan/buffer.hpp"
#include "core/drivers/vulkan/region.hpp"
#include "core/types.hpp"

#include <cstdint>

namespace sci::core::drivers::vulkan {

    // Scratch per batch; a batch is submitted early when it runs out.
    constexpr size capacity = size{256} << 10;

    // Dispatches per batch before it is submitted without being asked, so long chains of
    // operations keep the GPU busy while the host records the rest.
    constexpr uint32 batching = 128;

    // Estimated work per batch (on the order of 100 ms on an M1) before it is submitted.
    constexpr uint64 budget = uint64{1} << 30;

    Stream::Stream(const Context& context, Allocator& allocator)
        : context(context), allocator(allocator) {
        const VkCommandPoolCreateInfo command{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = context.family
        };
        vkCreateCommandPool(context.device, &command, nullptr, &pool);

        for (Batch& batch : batches) {
            const VkCommandBufferAllocateInfo allocation{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
            vkAllocateCommandBuffers(context.device, &allocation, &batch.commands);

            constexpr VkFenceCreateInfo signal{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            vkCreateFence(context.device, &signal, nullptr, &batch.fence);
            batch.scratch = allocator.compose(capacity);
        }
    }

    Stream::~Stream() {
        wait();
        for (Batch& batch : batches) {
            allocator.release(batch.scratch);
            if (batch.fence) vkDestroyFence(context.device, batch.fence, nullptr);
        }
        if (pool) vkDestroyCommandPool(context.device, pool, nullptr);
    }

    Region Stream::reserve(const size bytes) {
        ASSERT(bytes <= capacity);
        const size aligned = bytes + 255 & ~size{255};
        if (batches[current].open && batches[current].used + aligned > capacity) {
            flush();
        }
        if (!batches[current].open) begin();

        Batch& batch = batches[current];
        const Region region{.address = batch.scratch.address + batch.used, .host = static_cast<uint8*>(batch.scratch.host) + batch.used};
        batch.used += aligned;
        return region;
    }

    void Stream::record(const Pipeline& pipeline, const void* constants, const uint64 work, const uint32 columns, const uint32 rows, const uint32 layers) {
        if (!batches[current].open) begin();
        Batch& batch = batches[current];

        // Each dispatch may read anything an earlier dispatch (in this or a previous batch) wrote.
        batch.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(batch.commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        vkCmdPushConstants(batch.commands, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pipeline.bytes, constants);
        vkCmdDispatch(batch.commands, columns, rows, layers);
        batch.work += work;
        if (++batch.recorded >= batching || batch.work >= budget) flush();
    }

    void Stream::flush() {
        Batch& batch = batches[current];
        if (!batch.open) return;

        // Make shader writes visible to the host once the fence signals.
        batch.barrier(VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        vkEndCommandBuffer(batch.commands);

        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &batch.commands,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr
        };
        vkQueueSubmit(context.queue, 1, &submit, batch.fence);
        batch.open = false;
        batch.pending = true;
        current = (current + 1) % depth;
    }

    void Stream::wait() {
        flush();
        // A fence also covers everything submitted before it, so waiting in submission order is enough.
        for (size step = 0; step < depth; ++step) {
            const size index = (current + step) % depth;
            if (batches[index].pending) complete(index);
        }
    }

    void Stream::retire(const Buffer& buffer) {
        if (!buffer.handle) return;
        if (batches[current].open) {
            batches[current].retired.push_back(buffer);
            return;
        }
        // Otherwise attach it to the latest submission still in flight; a fence covers all earlier work.
        Batch& latest = batches[(current + depth - 1) % depth];
        if (latest.pending) {
            latest.retired.push_back(buffer);
        } else {
            allocator.release(buffer);
        }
    }

    void Stream::begin() {
        if (batches[current].pending) complete(current);

        Batch& batch = batches[current];
        vkResetCommandBuffer(batch.commands, 0);
        constexpr VkCommandBufferBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };
        vkBeginCommandBuffer(batch.commands, &begin);
        batch.open = true;
        batch.used = 0;
        batch.recorded = 0;
        batch.work = 0;
    }

    void Stream::complete(const size index) {
        Batch& batch = batches[index];
        vkWaitForFences(context.device, 1, &batch.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(context.device, 1, &batch.fence);
        batch.pending = false;
        for (const Buffer& buffer : batch.retired) allocator.release(buffer);
        batch.retired.clear();
    }

}
