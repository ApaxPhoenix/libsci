#pragma once

#include "core/drivers/vulkan/pipeline.hpp"

#include <memory>

namespace sci::core::drivers::vulkan {

    // The shader the Jit compiled for one chunk: both entry points of shaders/common/entries.slang,
    // evaluate (writes the output array) and reduce (one partial per workgroup), taking the same
    // arguments as shaders/interpreter.slang.
    struct Kernel {
        std::unique_ptr<Pipeline> evaluate;
        std::unique_ptr<Pipeline> reduce;
    };

}
