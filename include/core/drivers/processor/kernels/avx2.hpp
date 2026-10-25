#pragma once

#include "core/drivers/processor/kernels/table.hpp"

namespace sci::core::drivers::processor::kernels {

    // AVX2 + FMA kernels: eight floats or four doubles per register. Null when this build has no AVX2 code path.
    extern const Table* const avx2;

}
