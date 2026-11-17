#pragma once

/// @file
/// @brief A compute pipeline for one entry point of a SPIR-V module.

#include "core/types.hpp"
#include <span>
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    struct Context;

    /// A compute pipeline for one entry point of a SPIR-V module. Shaders take every argument
    /// (device pointers included) as push constants, so there are no descriptor sets to manage.
    class Pipeline {
    public:
        /// @brief A compute pipeline from SPIR-V.
        /// @param context The Vulkan device.
        /// @param code The SPIR-V.
        /// @param entry The entry point.
        /// @param bytes The push-constant block's size.
        Pipeline(const Context &context, std::span<const uint32> code, const char *entry, uint32 bytes);
        ~Pipeline();

        Pipeline(const Pipeline &) = delete;
        Pipeline &operator=(const Pipeline &) = delete;

        /// @brief Whether it was built.
        /// @return Whether it was.
        [[nodiscard]] bool valid() const noexcept { return handle != VK_NULL_HANDLE; }

        /// @brief The Vulkan device.
        const Context &context;
        /// @brief The shader module.
        VkShaderModule shader{VK_NULL_HANDLE};
        /// @brief The pipeline's layout.
        VkPipelineLayout layout{VK_NULL_HANDLE};
        /// @brief The pipeline.
        VkPipeline handle{VK_NULL_HANDLE};

        /// Size of the push-constant block the entry point takes.
        uint32 bytes{0};
    };

}
