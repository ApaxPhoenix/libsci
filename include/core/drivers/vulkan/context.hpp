#pragma once

/// @file
/// @brief Instance, device and compute queue, destroyed with the Context.

#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    /// Instance, device and compute queue, destroyed with the Context. Requires Vulkan 1.2 with buffer
    /// device addresses, 64-bit integers and subgroup arithmetic, which the shaders rely on
    /// (portability drivers such as MoltenVK included).
    struct Context {
        /// @brief The Vulkan instance.
        VkInstance instance{VK_NULL_HANDLE};
        /// @brief The physical device.
        VkPhysicalDevice physical{VK_NULL_HANDLE};
        /// @brief The logical device.
        VkDevice device{VK_NULL_HANDLE};
        /// @brief The compute queue.
        VkQueue queue{VK_NULL_HANDLE};
        /// @brief Its family.
        uint32 family{0};
        /// @brief Which capable GPU it is.
        uint32 index{0};
        /// @brief The device's properties.
        VkPhysicalDeviceProperties properties{};

        /// Whether shaders may request IEEE infinities, NaN and signed zero (SPIR-V float controls).
        bool preserving{false};

        Context() = default;
        ~Context();

        Context(const Context &) = delete;
        Context &operator=(const Context &) = delete;
        /// @brief Takes another context's device, leaving it empty.
        /// @param context The context moved from.
        Context(Context &&context) noexcept;
        Context &operator=(Context &&) = delete;

        /// Opens the ordinal-th capable GPU. Check valid() afterwards.
        /// @param ordinal Which capable GPU.
        /// @return The context.
        [[nodiscard]] static Context compose(uint32 ordinal = 0);
        /// @brief Whether a device is open.
        /// @return Whether it is.
        [[nodiscard]] bool valid() const noexcept { return device != VK_NULL_HANDLE; }
    };

}
