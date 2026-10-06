#pragma once

#include "core/drivers/cpu/kernels/table.hpp"

namespace sci::core::drivers::cpu::kernels {

    // Plain scalar kernels: any CPU, and the reference the vector paths are tested against. Never null.
    [[nodiscard]] const Table* fallback() noexcept;

}
