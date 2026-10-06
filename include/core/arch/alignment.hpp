#pragma once

#include <core/types.hpp>

namespace sci::core::arch {

    inline constexpr size cpu = 64;
    inline constexpr size gpu = 256;
    inline constexpr size atom = 128;
    // Largest cache line among supported CPUs (Apple M-series); array storage aligns to it.
    inline constexpr size line = 128;

}