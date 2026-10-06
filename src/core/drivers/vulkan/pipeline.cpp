#include "core/drivers/vulkan/pipeline.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <span>

#include <vector>

namespace sci::core::drivers::vulkan {

    // Adds the SignedZeroInfNanPreserve float control (capability plus one execution mode per
    // entry point). Without it MoltenVK compiles shaders with Metal fast math, which drops NaN and
    // infinity handling and loses accuracy in log, sin and division.
    static std::vector<uint32> preserve(const std::span<const uint32> code) {
        constexpr uint32 capability = 17;
        constexpr uint32 entry = 15;
        constexpr uint32 mode = 16;
        constexpr uint32 feature = 4466;
        constexpr uint32 setting = 4461;
        constexpr size header = 5;

        // Float controls are core from SPIR-V 1.4; older modules are left alone.
        if (code.size() < header || code[1] < 0x00010400) return {code.begin(), code.end()};

        std::vector<uint32> result(code.begin(), code.begin() + header);
        result.push_back(2u << 16 | capability);
        result.push_back(feature);

        std::vector<uint32> entries;
        bool placed = false;
        for (size position = header; position < code.size();) {
            const uint32 count = code[position] >> 16;
            const uint32 opcode = code[position] & 0xFFFF;
            if (count == 0 || position + count > code.size()) return {code.begin(), code.end()};

            if (opcode == entry) {
                entries.push_back(code[position + 2]);
            } else if (!entries.empty() && !placed) {
                for (const uint32 function : entries) {
                    result.insert(result.end(), {4u << 16 | mode, function, setting, 32u});
                }
                placed = true;
            }
            result.insert(result.end(), code.begin() + static_cast<std::ptrdiff_t>(position), code.begin() + static_cast<std::ptrdiff_t>(position + count));
            position += count;
        }
        return result;
    }

    Pipeline::Pipeline(const Context& context, const std::span<const uint32> code, const char* entry, const uint32 bytes)
        : context(context), bytes(bytes) {
        const std::vector<uint32> words = context.preserving ? preserve(code) : std::vector<uint32>(code.begin(), code.end());
        const VkShaderModuleCreateInfo module{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = words.size() * sizeof(uint32),
            .pCode = words.data()
        };
        if (vkCreateShaderModule(context.device, &module, nullptr, &shader) != VK_SUCCESS) return;

        const VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = bytes
        };
        const VkPipelineLayoutCreateInfo plan{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range
        };
        if (vkCreatePipelineLayout(context.device, &plan, nullptr, &layout) != VK_SUCCESS) return;

        const VkComputePipelineCreateInfo compute{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = entry,
                .pSpecializationInfo = nullptr
            },
            .layout = layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1
        };
        if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &compute, nullptr, &handle) != VK_SUCCESS) {
            handle = VK_NULL_HANDLE;
        }
    }

    Pipeline::~Pipeline() {
        if (handle) vkDestroyPipeline(context.device, handle, nullptr);
        if (layout) vkDestroyPipelineLayout(context.device, layout, nullptr);
        if (shader) vkDestroyShaderModule(context.device, shader, nullptr);
    }

}
