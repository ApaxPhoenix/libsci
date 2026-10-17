#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/types.hpp"

#include <cstdint>
#include <ranges>

namespace sci::core::drivers::vulkan {

    // Requests are rounded to this granularity so similar sizes share cached buffers.
    constexpr size granule = 4096;

    // Upper bound on memory parked in the cache.
    constexpr size budget = size{1} << 30;

    Allocator::Allocator(const Context& context)
        : context(context) {
        // The memory type: host-visible and coherent is mandatory; device-local on top of it
        // (unified memory) wins.
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(context.physical, &memory);
        constexpr VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32 best = UINT32_MAX;
        for (uint32 index = 0; index < memory.memoryTypeCount; ++index) {
            const VkMemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
            if ((flags & required) != required) continue;
            if (best == UINT32_MAX) best = index;
            if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                best = index;
                break;
            }
        }
        type = best == UINT32_MAX ? 0 : best;
    }

    Allocator::~Allocator() noexcept {
        for (const Buffer& buffer : std::views::values(live)) destroy(buffer);
        for (const Buffer& buffer : std::views::values(spare)) destroy(buffer);
    }

    Buffer Allocator::compose(const size bytes) {
        const size rounded = (bytes == 0 ? 1 : bytes) + granule - 1 & ~(granule - 1);

        // Reuse a cached buffer that is at most a quarter larger than needed.
        if (const auto match = spare.lower_bound(rounded); match != spare.end() && match->first <= rounded + rounded / 4) {
            const Buffer buffer = match->second;
            cached -= buffer.bytes;
            spare.erase(match);
            live.emplace(reinterpret_cast<std::uintptr_t>(buffer.host), buffer);
            return buffer;
        }

        Buffer buffer = create(rounded);
        if (!buffer.handle) {
            // Out of memory: drop the cache and retry once.
            for (const Buffer& cache : std::views::values(spare)) destroy(cache);
            spare.clear();
            cached = 0;
            buffer = create(rounded);
            if (!buffer.handle) return {};
        }
        live.emplace(reinterpret_cast<std::uintptr_t>(buffer.host), buffer);
        return buffer;
    }

    void Allocator::release(const Buffer& buffer) noexcept {
        if (!buffer.handle) return;
        live.erase(reinterpret_cast<std::uintptr_t>(buffer.host));
        if (cached + buffer.bytes <= budget) {
            spare.emplace(buffer.bytes, buffer);
            cached += buffer.bytes;
        } else {
            destroy(buffer);
        }
    }

    uint64 Allocator::locate(const void* pointer) const noexcept {
        const auto point = reinterpret_cast<std::uintptr_t>(pointer);
        auto match = live.upper_bound(point);
        if (match == live.begin()) return 0;
        --match;
        const Buffer& buffer = match->second;
        if (point >= match->first + buffer.bytes) return 0;
        return buffer.address + (point - match->first);
    }

    Buffer Allocator::create(const size bytes) {
        Buffer buffer{};
        buffer.bytes = bytes;

        const VkBufferCreateInfo description{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = bytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };
        if (vkCreateBuffer(context.device, &description, nullptr, &buffer.handle) != VK_SUCCESS) return {};

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context.device, buffer.handle, &requirements);

        constexpr VkMemoryAllocateFlagsInfo flags{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = nullptr,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            .deviceMask = 0
        };
        const VkMemoryAllocateInfo allocation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &flags,
            .allocationSize = requirements.size,
            .memoryTypeIndex = type
        };
        if (vkAllocateMemory(context.device, &allocation, nullptr, &buffer.memory) != VK_SUCCESS ||
            vkBindBufferMemory(context.device, buffer.handle, buffer.memory, 0) != VK_SUCCESS ||
            vkMapMemory(context.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.host) != VK_SUCCESS) {
            destroy(buffer);
            return {};
        }

        const VkBufferDeviceAddressInfo address{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = buffer.handle
        };
        buffer.address = vkGetBufferDeviceAddress(context.device, &address);
        return buffer;
    }

    void Allocator::destroy(const Buffer& buffer) const noexcept {
        if (buffer.host) vkUnmapMemory(context.device, buffer.memory);
        if (buffer.handle) vkDestroyBuffer(context.device, buffer.handle, nullptr);
        if (buffer.memory) vkFreeMemory(context.device, buffer.memory, nullptr);
    }

}
