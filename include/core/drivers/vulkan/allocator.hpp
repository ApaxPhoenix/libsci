#pragma once

#include "core/drivers/vulkan/buffer.hpp"
#include "core/types.hpp"
#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>

namespace sci::core::drivers::vulkan {

    struct Context;

    // One VkBuffer per array, recycled through a size-keyed cache so temporaries created in a loop
    // do not pay for vkAllocateMemory every time.
    class Allocator {
    public:
        explicit Allocator(const Context& context);
        ~Allocator() noexcept;

        Allocator(const Allocator&) = delete;
        Allocator& operator=(const Allocator&) = delete;

        // An empty buffer (null handle) when the device is out of memory.
        [[nodiscard]] Buffer compose(size bytes);

        // The caller guarantees the GPU no longer uses the buffer (see Stream::retire).
        void release(const Buffer& buffer) noexcept;

        // Device address of a host pointer into a live buffer, or 0 when it is not GPU memory.
        [[nodiscard]] uint64 locate(const void* pointer) const noexcept;

    private:
        [[nodiscard]] Buffer create(size bytes);
        void destroy(const Buffer& buffer) const noexcept;

        const Context& context;
        uint32 type{0};

        // Live buffers keyed by host address, and released ones kept for reuse, keyed by size.
        std::map<std::uintptr_t, Buffer> live;
        std::multimap<size, Buffer> spare;
        size cached{0};
    };

}
