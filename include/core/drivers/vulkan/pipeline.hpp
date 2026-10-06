#pragma once

#include "core/types.hpp"
#include <vulkan/vulkan.h>
#include <span>

namespace sci::core::drivers::vulkan {

    struct Context;

    // A compute pipeline for one entry point of a SPIR-V module. Shaders take every argument
    // (device pointers included) as push constants, so there are no descriptor sets to manage.
    class Pipeline {
    public:
        Pipeline(const Context& context, std::span<const uint32> code, const char* entry, uint32 bytes);
        ~Pipeline();

        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;

        [[nodiscard]] bool valid() const noexcept { return handle != VK_NULL_HANDLE; }
        [[nodiscard]] VkPipeline native() const noexcept { return handle; }
        [[nodiscard]] VkPipelineLayout schema() const noexcept { return layout; }

        // Size of the push-constant block the entry point takes.
        [[nodiscard]] uint32 constants() const noexcept { return bytes; }

    private:
        const Context& context;
        VkShaderModule shader{VK_NULL_HANDLE};
        VkPipelineLayout layout{VK_NULL_HANDLE};
        VkPipeline handle{VK_NULL_HANDLE};
        uint32 bytes{0};
    };

}
