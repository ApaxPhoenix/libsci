#pragma once

#include "core/drivers/cpu/kernels/table.hpp"

namespace sci::core::drivers::cpu::kernels {

    // AVX-512 kernels: sixteen floats or eight doubles per register. Null when this build has no AVX-512 code path.
    [[nodiscard]] const Table* avx512() noexcept;

}
