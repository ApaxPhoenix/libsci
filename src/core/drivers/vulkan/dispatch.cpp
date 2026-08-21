#include "core/drivers/vulkan/dispatch.hpp"
#include "core/assert.hpp"
#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/drivers/vulkan/pipeline.hpp"

namespace sci::core::drivers::vulkan {

    dispatch::dispatch(const context& context)
        : context_(context) {
        const VkCommandPoolCreateInfo command{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = context_.family
        };
        vkCreateCommandPool(context_.device, &command, nullptr, &pool_);

        const VkCommandBufferAllocateInfo allocation{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        vkAllocateCommandBuffers(context_.device, &allocation, &buffer_);

        constexpr VkFenceCreateInfo signal{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        vkCreateFence(context_.device, &signal, nullptr, &fence_);

        constexpr VkDescriptorPoolSize size{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 16
        };
        const VkDescriptorPoolCreateInfo descriptor{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 16,
            .poolSizeCount = 1,
            .pPoolSizes = &size
        };
        vkCreateDescriptorPool(context_.device, &descriptor, nullptr, &group_);
    }

    dispatch::~dispatch() {
        if (group_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(context_.device, group_, nullptr);
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(context_.device, fence_, nullptr);
        if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(context_.device, pool_, nullptr);
    }

    void dispatch::wait() const noexcept {
        vkWaitForFences(context_.device, 1, &fence_, VK_TRUE, UINT64_MAX);
    }

    bool dispatch::ready() const noexcept {
        return vkGetFenceStatus(context_.device, fence_) == VK_SUCCESS;
    }

    void dispatch::compose(const pipeline& pipeline, const std::span<const buffer> buffers, const uint32 x, const uint32 y, const uint32 z) {
        ASSERT(buffers.size() <= 16);

        wait();
        vkResetFences(context_.device, 1, &fence_);
        vkResetDescriptorPool(context_.device, group_, 0);

        VkDescriptorSet set{VK_NULL_HANDLE};
        VkDescriptorSetLayout binding = pipeline.binding();
        const VkDescriptorSetAllocateInfo allocation{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = group_,
            .descriptorSetCount = 1,
            .pSetLayouts = &binding
        };
        if (vkAllocateDescriptorSets(context_.device, &allocation, &set) != VK_SUCCESS) return;

        const uint32 count = buffers.size() > 16 ? 16 : static_cast<uint32>(buffers.size());

        VkDescriptorBufferInfo info[16]{};
        VkWriteDescriptorSet write[16]{};
        for (uint32 index = 0; index < count; ++index) {
            info[index] = {
                .buffer = buffers[index].handle,
                .offset = buffers[index].offset,
                .range = buffers[index].bytes
            };
            write[index] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = index,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &info[index],
                .pTexelBufferView = nullptr
            };
        }
        vkUpdateDescriptorSets(context_.device, count, write, 0, nullptr);

        constexpr VkCommandBufferBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };
        vkBeginCommandBuffer(buffer_, &begin);

        vkCmdBindPipeline(buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.native());
        vkCmdBindDescriptorSets(buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.schema(), 0, 1, &set, 0, nullptr);
        vkCmdDispatch(buffer_, x, y, z);

        vkEndCommandBuffer(buffer_);

        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &buffer_,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr
        };
        vkQueueSubmit(context_.queue, 1, &submit, fence_);
    }

}