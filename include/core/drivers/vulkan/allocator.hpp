#pragma once

#include "core/memory/cursor.hpp"
#include "core/types.hpp"
#include <vulkan/vulkan.h>

namespace sci::core::drivers::vulkan {

    struct context;

    struct buffer {
        VkBuffer handle{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        void* host{nullptr};
        size offset{0};
        size bytes{0};
    };

    class allocator {
    public:
        explicit allocator(const context& context, size bytes = 128 * 1024 * 1024);
        ~allocator() noexcept;

        [[nodiscard]] buffer compose(size bytes) noexcept;
        static void dispose(buffer& object) noexcept;
        void reset() noexcept;

        template <typename type>
        [[nodiscard]] static memory::cursor<type> map(const buffer& object) noexcept {
            return memory::cursor<type>(static_cast<type*>(object.host));
        }

    private:
        [[nodiscard]] static uint32 select(VkPhysicalDevice physical, uint32 filter) noexcept;

        const context& context_;
        VkBuffer buffer_{VK_NULL_HANDLE};
        VkDeviceMemory memory_{VK_NULL_HANDLE};
        void* host_{nullptr};
        size capacity_{0};
        size offset_{0};
    };

}