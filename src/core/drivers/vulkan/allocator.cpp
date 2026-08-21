#include "core/drivers/vulkan/allocator.hpp"
#include "core/arch/alignment.hpp"
#include "core/drivers/vulkan/context.hpp"

namespace sci::core::drivers::vulkan {

    allocator::allocator(const context& context, const size bytes)
        : context_(context), capacity_(bytes) {
        const VkBufferCreateInfo buffer{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = capacity_,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };
        if (vkCreateBuffer(context_.device, &buffer, nullptr, &buffer_) != VK_SUCCESS) return;

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context_.device, buffer_, &requirements);

        const VkMemoryAllocateInfo allocation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = select(context_.physical, requirements.memoryTypeBits)
        };
        if (vkAllocateMemory(context_.device, &allocation, nullptr, &memory_) != VK_SUCCESS) return;

        vkBindBufferMemory(context_.device, buffer_, memory_, 0);
        vkMapMemory(context_.device, memory_, 0, capacity_, 0, &host_);
    }

    allocator::~allocator() noexcept {
        if (host_) vkUnmapMemory(context_.device, memory_);
        if (buffer_) vkDestroyBuffer(context_.device, buffer_, nullptr);
        if (memory_) vkFreeMemory(context_.device, memory_, nullptr);
    }

    buffer allocator::compose(const size bytes) noexcept {
        buffer object{};
        const size aligned = bytes + arch::gpu - 1 & ~(arch::gpu - 1);

        if (offset_ + aligned > capacity_) return object;

        object.handle = buffer_;
        object.memory = memory_;
        object.offset = offset_;
        object.bytes = aligned;
        object.host = static_cast<uint8*>(host_) + offset_;

        offset_ += aligned;
        return object;
    }

    void allocator::dispose(buffer& object) noexcept {
        object.host = nullptr;
        object.handle = VK_NULL_HANDLE;
        object.memory = VK_NULL_HANDLE;
        object.bytes = 0;
        object.offset = 0;
    }

    void allocator::reset() noexcept {
        offset_ = 0;
    }

    uint32 allocator::select(const VkPhysicalDevice physical, const uint32 filter) noexcept {
        VkPhysicalDeviceMemoryProperties buffer{};
        vkGetPhysicalDeviceMemoryProperties(physical, &buffer);
        constexpr VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32 index = 0; index < buffer.memoryTypeCount; ++index) {
            if (filter & 1 << index && (buffer.memoryTypes[index].propertyFlags & required) == required) {
                return index;
            }
        }
        return 0;
    }

}