#pragma once

/// @file
/// @brief One VkBuffer per array, recycled through a size-keyed cache so temporaries created in a
///       loop do not pay for vkAllocateMemory every time.

#include "core/drivers/vulkan/buffer.hpp"
#include "core/types.hpp"
#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>

namespace sci::core::drivers::vulkan {

    struct Context;

    /// One VkBuffer per array, recycled through a size-keyed cache so temporaries created in a loop
    /// do not pay for vkAllocateMemory every time.
    class Allocator {
    public:
        /// @brief An allocator on a device.
        /// @param context The Vulkan device.
        explicit Allocator(const Context &context);
        ~Allocator() noexcept;

        Allocator(const Allocator &) = delete;
        Allocator &operator=(const Allocator &) = delete;

        /// An empty buffer (null handle) when the device is out of memory.
        /// @param bytes How many bytes.
        /// @return The buffer.
        [[nodiscard]] Buffer compose(size bytes);

        /// The caller guarantees the GPU no longer uses the buffer (see Stream::retire).
        /// @param buffer The buffer.
        void release(const Buffer &buffer) noexcept;

        /// Device address of a host pointer into a live buffer, or 0 when it is not GPU memory.
        /// @param pointer The pointer.
        /// @return The address.
        [[nodiscard]] uint64 locate(const void *pointer) const noexcept;

    private:
        /// @brief A new buffer, mapped and with a device address.
        /// @param bytes How many bytes.
        /// @return The buffer, or an empty one.
        [[nodiscard]] Buffer create(size bytes);
        /// @brief Frees a buffer.
        /// @param buffer The buffer.
        void destroy(const Buffer &buffer) const noexcept;

        /// @brief The Vulkan device.
        const Context &context;
        /// @brief The memory type buffers come from.
        uint32 type{0};

        /// Live buffers keyed by host address, and released ones kept for reuse, keyed by size.
        std::map<std::uintptr_t, Buffer> live;
        /// @brief Released buffers kept for reuse, by size.
        std::multimap<size, Buffer> spare;
        /// @brief How many bytes those hold.
        size cached{0};
    };

}
