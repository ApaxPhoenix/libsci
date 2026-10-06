#pragma once

#include "core/drivers/cpu/kernels/table.hpp"

namespace sci::core::drivers::cpu::kernels {

    // NEON kernels: four floats or two doubles per register. Null when this build is not for ARM64.
    [[nodiscard]] const Table* neon() noexcept;

}
