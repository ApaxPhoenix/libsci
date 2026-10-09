#pragma once

#include "core/drivers/target.hpp"
#include "core/drivers/vulkan/buffer.hpp"
#include "core/memory/allocation.hpp"
#include "core/types.hpp"

namespace sci::core::drivers {

    // One device allocation: a mapped GPU buffer, or host memory for a CPU device. GPU memory is
    // mapped into the host address space too, so `host` is always a usable pointer once pending GPU
    // work has finished.
    struct Memory {
        void* host{nullptr};
        vulkan::Buffer buffer{};
        memory::Allocation allocation{};
        size bytes{0};
        Target target{Target::processor};
    };

}
