#pragma once

/// @file
/// @brief The shader the Compiler compiled for one chunk.

#include "core/drivers/vulkan/pipeline.hpp"

#include <memory>

namespace sci::core::drivers::vulkan {

    /// The shader the Compiler compiled for one chunk: both entry points of shaders/common/entries.slang,
    /// evaluate (writes the output array) and reduce (one partial per workgroup), taking the same
    /// arguments as shaders/interpreter.slang.
    struct Kernel {
        /// @brief The pipeline evaluating the chunk.
        std::unique_ptr<Pipeline> evaluate;
        /// @brief The pipeline reducing it.
        std::unique_ptr<Pipeline> reduce;
    };

}
