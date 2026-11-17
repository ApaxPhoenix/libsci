#pragma once

/// @file
/// @brief Plain scalar kernels.

#include "core/drivers/processor/kernels/table.hpp"

namespace sci::core::drivers::processor::kernels {

    /// Plain scalar kernels: any CPU, and the reference the vector paths are tested against. Never null.
    extern const Table *const fallback;

}
