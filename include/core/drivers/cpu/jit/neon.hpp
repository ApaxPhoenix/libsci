#pragma once

#include "core/bytecode/chunk.hpp"

#include <asmjit/core.h>

namespace sci::core::drivers::cpu::jit {

    // Emits `chunk` into `code` as a NEON loop (a Jit::Function, four floats or two doubles per iteration).
    // False when this build is not for ARM64 or the chunk holds an instruction it cannot compile.
    [[nodiscard]] bool neon(asmjit::CodeHolder& code, const Chunk& chunk);

}
