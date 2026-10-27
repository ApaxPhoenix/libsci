#pragma once

#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    // Instance, device and compute queue, destroyed with the Context. Requires Vulkan 1.2 with buffer
    // device addresses, 64-bit integers and subgroup arithmetic, which the shaders rely on
    // (portability drivers such as MoltenVK included).
    struct Context {
        VkInstance instance{VK_NULL_HANDLE};
        VkPhysicalDevice physical{VK_NULL_HANDLE};
        VkDevice device{VK_NULL_HANDLE};
        VkQueue queue{VK_NULL_HANDLE};
        uint32 family{0};
        uint32 index{0};
        VkPhysicalDeviceProperties properties{};

        // Whether shaders may request IEEE infinities, NaN and signed zero (SPIR-V float controls).
        bool preserving{false};

        Context() = default;
        ~Context();

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;
        Context(Context&& context) noexcept;
        Context& operator=(Context&&) = delete;

        // Opens the ordinal-th capable GPU. Check valid() afterwards.
        [[nodiscard]] static Context compose(uint32 ordinal = 0);
        [[nodiscard]] bool valid() const noexcept { return device != VK_NULL_HANDLE; }
    };

}
