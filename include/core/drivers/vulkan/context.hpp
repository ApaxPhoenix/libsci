#pragma once

#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    struct context {
        VkInstance instance{VK_NULL_HANDLE};
        VkPhysicalDevice physical{VK_NULL_HANDLE};
        VkDevice device{VK_NULL_HANDLE};
        VkQueue queue{VK_NULL_HANDLE};
        uint32 family{0};
        uint32 index{0};

        [[nodiscard]] static context compose(uint32 ordinal = 0);
        void dispose() noexcept;
    };

}