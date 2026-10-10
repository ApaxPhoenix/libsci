#pragma once

#include "core/types.hpp"

namespace sci::core::alignment {

    // Alignment of host scratch and arena memory: one vector register of the widest ISA (AVX-512).
    inline constexpr size processor = 64;

    // Largest cache line among supported CPUs (Apple M-series); array storage aligns to it.
    inline constexpr size line = 128;

}
