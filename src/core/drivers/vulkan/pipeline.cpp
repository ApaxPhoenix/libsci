#include "core/drivers/vulkan/pipeline.hpp"
#include "core/drivers/vulkan/context.hpp"

namespace sci::core::drivers::vulkan {

    pipeline::pipeline(const context& context)
        : context_(context) {}

    pipeline::~pipeline() {
        dispose();
    }

    bool pipeline::compose(const std::span<const uint32> code, const uint32 count) {
        dispose();

        const VkShaderModuleCreateInfo module{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = code.size_bytes(),
            .pCode = code.data()
        };
        if (vkCreateShaderModule(context_.device, &module, nullptr, &shader_) != VK_SUCCESS) return false;

        VkDescriptorSetLayoutBinding bindings[16]{};
        const uint32 slots = (count > 16) ? 16 : count;
        for (uint32 index = 0; index < slots; ++index) {
            bindings[index] = {
                .binding = index,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr
            };
        }

        const VkDescriptorSetLayoutCreateInfo group{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = slots,
            .pBindings = bindings
        };
        if (vkCreateDescriptorSetLayout(context_.device, &group, nullptr, &descriptor_) != VK_SUCCESS) return false;

        const VkPipelineLayoutCreateInfo plan{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr
        };
        if (vkCreatePipelineLayout(context_.device, &plan, nullptr, &layout_) != VK_SUCCESS) return false;

        const VkComputePipelineCreateInfo compute{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_,
                .pName = "main",
                .pSpecializationInfo = nullptr
            },
            .layout = layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1
        };
        return vkCreateComputePipelines(context_.device, VK_NULL_HANDLE, 1, &compute, nullptr, &handle_) == VK_SUCCESS;
    }

    void pipeline::dispose() noexcept {
        if (handle_) { vkDestroyPipeline(context_.device, handle_, nullptr); handle_ = VK_NULL_HANDLE; }
        if (layout_) { vkDestroyPipelineLayout(context_.device, layout_, nullptr); layout_ = VK_NULL_HANDLE; }
        if (descriptor_) { vkDestroyDescriptorSetLayout(context_.device, descriptor_, nullptr); descriptor_ = VK_NULL_HANDLE; }
        if (shader_) { vkDestroyShaderModule(context_.device, shader_, nullptr); shader_ = VK_NULL_HANDLE; }
    }

}