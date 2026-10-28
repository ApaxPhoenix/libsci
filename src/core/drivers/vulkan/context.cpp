#include "core/drivers/vulkan/context.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace sci::core::drivers::vulkan {

    // Query structs below are zeroed, then given their type tag and extension-chain pointer.
    Context Context::compose(const uint32 ordinal) {
        Context context{};
        context.index = ordinal;

        uint32 count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());

        // Portability drivers (MoltenVK on macOS) are only listed when the instance opts in.
        std::vector<const char*> extensions;
        VkInstanceCreateFlags flags = 0;
        if (std::ranges::any_of(available, [](const VkExtensionProperties& extension) {
                return std::strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
            })) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }

        constexpr VkApplicationInfo application{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = nullptr,
            .applicationVersion = 0,
            .pEngineName = "libsci",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_2
        };

        const VkInstanceCreateInfo instance{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .pApplicationInfo = &application,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        if (vkCreateInstance(&instance, nullptr, &context.instance) != VK_SUCCESS) return context;

        uint32 total = 0;
        vkEnumeratePhysicalDevices(context.instance, &total, nullptr);
        std::vector<VkPhysicalDevice> devices(total);
        vkEnumeratePhysicalDevices(context.instance, &total, devices.data());

        // Shaders need Vulkan 1.2, 64-bit device pointers and subgroup reductions in compute.
        std::vector<VkPhysicalDevice> usable;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_2) continue;

            VkPhysicalDeviceVulkan12Features modern{};
            modern.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &modern;
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            if (!modern.bufferDeviceAddress || !features.features.shaderInt64) continue;

            VkPhysicalDeviceSubgroupProperties subgroup{};
            subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            VkPhysicalDeviceProperties2 extended{};
            extended.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            extended.pNext = &subgroup;
            vkGetPhysicalDeviceProperties2(candidate, &extended);
            constexpr VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            if ((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (subgroup.supportedOperations & needed) == needed) {
                usable.push_back(candidate);
            }
        }
        if (ordinal >= usable.size()) return context;

        VkPhysicalDevice candidate = usable[ordinal];
        vkGetPhysicalDeviceProperties(candidate, &context.properties);

        uint32 slots = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &slots, nullptr);
        std::vector<VkQueueFamilyProperties> families(slots);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &slots, families.data());

        bool found = false;
        for (uint32 slot = 0; slot < slots; ++slot) {
            if (families[slot].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                context.family = slot;
                found = true;
                break;
            }
        }
        if (!found) return context;
        context.physical = candidate;

        uint32 known = 0;
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &known, nullptr);
        std::vector<VkExtensionProperties> supported(known);
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &known, supported.data());

        // Required whenever a portability driver exposes it.
        std::vector<const char*> enabled;
        if (std::ranges::any_of(supported, [](const VkExtensionProperties& extension) {
                return std::strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0;
            })) {
            enabled.push_back("VK_KHR_portability_subset");
        }

        VkPhysicalDeviceVulkan12Features modern{};
        modern.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        modern.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &modern;
        features.features.shaderInt64 = VK_TRUE;

        float priority{1.0f};
        const VkDeviceQueueCreateInfo queue{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = context.family,
            .queueCount = 1,
            .pQueuePriorities = &priority
        };

        const VkDeviceCreateInfo setup{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &features,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32>(enabled.size()),
            .ppEnabledExtensionNames = enabled.data(),
            .pEnabledFeatures = nullptr
        };

        VkPhysicalDeviceFloatControlsProperties controls{};
        controls.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        VkPhysicalDeviceProperties2 extended{};
        extended.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        extended.pNext = &controls;
        vkGetPhysicalDeviceProperties2(candidate, &extended);
        context.preserving = controls.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;

        if (vkCreateDevice(context.physical, &setup, nullptr, &context.device) == VK_SUCCESS) {
            vkGetDeviceQueue(context.device, context.family, 0, &context.queue);
        }
        return context;
    }

    Context::~Context() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    Context::Context(Context&& context) noexcept
        : instance(std::exchange(context.instance, VK_NULL_HANDLE)),
          physical(std::exchange(context.physical, VK_NULL_HANDLE)),
          device(std::exchange(context.device, VK_NULL_HANDLE)),
          queue(std::exchange(context.queue, VK_NULL_HANDLE)),
          family(context.family),
          index(context.index),
          properties(context.properties),
          preserving(context.preserving) {}

}
