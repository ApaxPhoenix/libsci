#pragma once

#include "core/drivers/cpu/kernels/routines.hpp"

namespace sci::core::drivers::cpu::kernels {

    // One per instruction set (fallback.hpp, neon.hpp, avx2.hpp, avx512.hpp), each built from the same
    // generic kernels through generic/compose.hpp.
    struct Table {
        const char* name{nullptr};
        Routines<float> float32{};
        Routines<double> float64{};
    };

}
