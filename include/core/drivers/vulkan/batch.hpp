#pragma once

#include "core/drivers/vulkan/buffer.hpp"
#include "core/types.hpp"
#include <vulkan/vulkan.h>

#include <vector>

namespace sci::core::drivers::vulkan {

    // One command buffer of a Stream, with its fence, scratch memory and the buffers waiting on it.
    struct Batch {
        VkCommandBuffer commands{VK_NULL_HANDLE};
        VkFence fence{VK_NULL_HANDLE};
        Buffer scratch{};
        size used{0};
        uint32 recorded{0};
        uint64 work{0};
        bool open{false};
        bool pending{false};
        std::vector<Buffer> retired;

        // Records a barrier that makes everything compute shaders have written so far visible to
        // `read` accesses in the `after` stage.
        void barrier(const VkPipelineStageFlags after, const VkAccessFlags read) const noexcept {
            const VkMemoryBarrier memory{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = read
            };
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, after, 0, 1, &memory, 0, nullptr, 0, nullptr);
        }
    };

}
