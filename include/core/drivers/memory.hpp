#pragma once

/// @file
/// @brief One device allocation.

#include "core/drivers/target.hpp"
#include "core/drivers/vulkan/buffer.hpp"
#include "core/memory/allocation.hpp"
#include "core/types.hpp"

namespace sci::core::drivers {

    /// One device allocation: a mapped GPU buffer, or host memory for a CPU device. GPU memory is
    /// mapped into the host address space too, so `host` is always a usable pointer once pending GPU
    /// work has finished.
    struct Memory {
        /// @brief The memory, as the host sees it.
        void *host{nullptr};
        /// @brief The Vulkan buffer, for GPU memory.
        vulkan::Buffer buffer{};
        /// @brief The allocation, for host memory.
        memory::Allocation allocation{};
        /// @brief How many bytes.
        size bytes{0};
        /// @brief Where it lives.
        Target target{Target::processor};
    };

}
