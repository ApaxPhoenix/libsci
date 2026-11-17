#pragma once

/// @file
/// @brief One command buffer of a Stream, with its fence, scratch memory and the buffers waiting on
///       it.

#include "core/drivers/vulkan/buffer.hpp"
#include "core/types.hpp"
#include <vulkan/vulkan.h>

#include <vector>

namespace sci::core::drivers::vulkan {

    /// One command buffer of a Stream, with its fence, scratch memory and the buffers waiting on it.
    struct Batch {
        /// @brief The command buffer.
        VkCommandBuffer commands{VK_NULL_HANDLE};
        /// @brief Signalled when it is done.
        VkFence fence{VK_NULL_HANDLE};
        /// @brief Its scratch memory.
        Buffer scratch{};
        /// @brief How much of the scratch memory is used.
        size used{0};
        /// @brief How many dispatches it holds.
        uint32 recorded{0};
        /// @brief How much work they carry.
        uint64 work{0};
        /// @brief Whether it is being recorded.
        bool open{false};
        /// @brief Whether it was submitted and may not be done.
        bool pending{false};
        /// @brief Buffers to hand back once it is done.
        std::vector<Buffer> retired;

        /// Records a barrier that makes everything compute shaders have written so far visible to
        /// `read` accesses in the `after` stage.
        /// @param after The stage that reads.
        /// @param read The accesses that read.
        void barrier(const VkPipelineStageFlags after, const VkAccessFlags read) const noexcept {
            const VkMemoryBarrier memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                         .pNext = nullptr,
                                         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                         .dstAccessMask = read};
            vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, after, 0, 1, &memory, 0, nullptr, 0,
                                 nullptr);
        }
    };

}
