#pragma once

#include "core/types.hpp"
#include <vulkan/vulkan.h>
#include <span>

namespace sci::core::drivers::vulkan {

    struct context;
    struct buffer;
    class pipeline;

    class dispatch {
    public:
        explicit dispatch(const context& context);
        ~dispatch();

        void wait() const noexcept;
        [[nodiscard]] bool ready() const noexcept;
        void compose(const pipeline& pipeline, std::span<const buffer> buffers, uint32 x, uint32 y = 1, uint32 z = 1);

    private:
        const context& context_;
        VkCommandPool pool_{VK_NULL_HANDLE};
        VkCommandBuffer buffer_{VK_NULL_HANDLE};
        VkFence fence_{VK_NULL_HANDLE};
        VkDescriptorPool group_{VK_NULL_HANDLE};
    };

}