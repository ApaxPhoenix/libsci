#include "core/drivers/vulkan/context.hpp"

namespace sci::core::drivers::vulkan {

    context context::compose(const uint32 ordinal) {
        context data{};
        data.index = ordinal;

        constexpr VkApplicationInfo app{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = nullptr,
            .applicationVersion = 0,
            .pEngineName = "libsci",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3
        };

        const VkInstanceCreateInfo instance{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pApplicationInfo = &app,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = nullptr
        };

        if (vkCreateInstance(&instance, nullptr, &data.instance) != VK_SUCCESS) return data;

        VkPhysicalDevice devices[16]{};
        uint32 count{16};
        if (vkEnumeratePhysicalDevices(data.instance, &count, devices) != VK_SUCCESS || count == 0) return data;
        if (ordinal >= count) return data;

        const VkPhysicalDevice candidate = devices[ordinal];

        VkQueueFamilyProperties families[16]{};
        uint32 slots{16};
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &slots, families);

        for (uint32 slot = 0; slot < slots; ++slot) {
            if (families[slot].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                data.physical = candidate;
                data.family = slot;
                break;
            }
        }

        if (data.physical == VK_NULL_HANDLE) return data;

        float priority{1.0f};
        const VkDeviceQueueCreateInfo config{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = data.family,
            .queueCount = 1,
            .pQueuePriorities = &priority
        };

        const VkDeviceCreateInfo setup{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &config,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = nullptr,
            .pEnabledFeatures = nullptr
        };

        if (vkCreateDevice(data.physical, &setup, nullptr, &data.device) == VK_SUCCESS) {
            vkGetDeviceQueue(data.device, data.family, 0, &data.queue);
        }
        return data;
    }

    void context::dispose() noexcept {
        if (device) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }

}