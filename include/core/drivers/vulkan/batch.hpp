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
    };

}
