#pragma once

#include "core/bytecode/chunk.hpp"

#include <asmjit/core.h>

namespace sci::core::drivers::cpu::jit {

    // Emits `chunk` into `code` as an AVX2 loop (a Jit::Function, eight floats or four doubles per iteration).
    // False when this build is not for x86-64 or the chunk holds an instruction it cannot compile.
    [[nodiscard]] bool avx2(asmjit::CodeHolder& code, const Chunk& chunk);

}
