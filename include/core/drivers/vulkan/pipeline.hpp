#pragma once

#include "core/types.hpp"
#include <vulkan/vulkan.h>
#include <span>

namespace sci::core::drivers::vulkan {

    struct context;

    class pipeline {
    public:
        explicit pipeline(const context& context);
        ~pipeline();

        bool compose(std::span<const uint32> code, uint32 count = 3);
        void dispose() noexcept;

        [[nodiscard]] VkPipeline native() const noexcept { return handle_; }
        [[nodiscard]] VkPipelineLayout schema() const noexcept { return layout_; }
        [[nodiscard]] VkDescriptorSetLayout binding() const noexcept { return descriptor_; }

    private:
        const context& context_;
        VkShaderModule shader_{VK_NULL_HANDLE};
        VkDescriptorSetLayout descriptor_{VK_NULL_HANDLE};
        VkPipelineLayout layout_{VK_NULL_HANDLE};
        VkPipeline handle_{VK_NULL_HANDLE};
    };

}