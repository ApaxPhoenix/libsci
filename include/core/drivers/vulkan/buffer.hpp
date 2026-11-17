#pragma once

/// @file
/// @brief A device buffer that is also mapped into the host address space.

#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    /// A device buffer that is also mapped into the host address space. Shaders reach it through its
    /// 64-bit device address; the host reads and writes it directly through `host`.
    struct Buffer {
        /// @brief The buffer.
        VkBuffer handle{VK_NULL_HANDLE};
        /// @brief Its memory.
        VkDeviceMemory memory{VK_NULL_HANDLE};
        /// @brief Where the host sees it.
        void *host{nullptr};
        /// @brief Its device address.
        uint64 address{0};
        /// @brief How many bytes.
        size bytes{0};
    };

}
