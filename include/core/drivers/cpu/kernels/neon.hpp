#pragma once

#include "core/memory/cursor.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    struct neon {
        static void add(memory::cursor<float> left, memory::cursor<float> right, memory::cursor<float> yield, size count) noexcept;
    };

}