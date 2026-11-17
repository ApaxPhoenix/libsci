#pragma once

/// @file
/// @brief Every kernel for every instruction set, one table each.

#include "core/drivers/processor/kernels/routines.hpp"

namespace sci::core::drivers::processor::kernels {

    /// One per instruction set (fallback.hpp, neon.hpp, avx2.hpp, avx512.hpp), each built from the same
    /// generic kernels through generic/compose.hpp.
    struct Table {
        /// @brief The instruction set's name.
        const char *name{nullptr};
        /// @brief The float kernels.
        Routines<float> single{};
        /// @brief The double kernels.
        Routines<double> wide{};
    };

}
