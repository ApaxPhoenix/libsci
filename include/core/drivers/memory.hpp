#pragma once

#include "core/drivers/target.hpp"
#include "core/drivers/vulkan/buffer.hpp"
#include "core/types.hpp"

namespace sci::core::drivers {

    // One device allocation. GPU memory is mapped into the host address space too, so `host` is
    // always a usable pointer once pending GPU work has finished.
    struct Memory {
        void* host{nullptr};
        vulkan::Buffer buffer{};
        size bytes{0};
        Target target{Target::cpu};
    };

}
