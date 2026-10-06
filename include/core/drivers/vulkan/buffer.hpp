#pragma once

#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    // A device buffer that is also mapped into the host address space. Shaders reach it through its
    // 64-bit device address; the host reads and writes it directly through `host`.
    struct Buffer {
        VkBuffer handle{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        void* host{nullptr};
        uint64 address{0};
        size bytes{0};
    };

}
