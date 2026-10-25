#pragma once

#include "core/drivers/processor/kernels/table.hpp"

namespace sci::core::drivers::processor::kernels {

    // AVX-512 kernels: sixteen floats or eight doubles per register. Null when this build has no AVX-512 code path.
    extern const Table* const avx512;

}
