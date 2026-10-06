#pragma once

#include "core/drivers/cpu/kernels/table.hpp"

namespace sci::core::drivers::cpu::kernels {

    // AVX2 + FMA kernels: eight floats or four doubles per register. Null when this build has no AVX2 code path.
    [[nodiscard]] const Table* avx2() noexcept;

}
