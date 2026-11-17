#pragma once

/// @file
/// @brief Emits `chunk` into `code` as an AVX2 loop (a Compiler::Function, eight floats or four
///       doubles per iteration).

#include "core/bytecode/chunk.hpp"

#include <asmjit/core.h>

namespace sci::core::drivers::processor::compiler {

    /// Emits `chunk` into `code` as an AVX2 loop (a Compiler::Function, eight floats or four doubles per iteration).
    /// False when this build is not for x86-64 or the chunk holds an instruction it cannot compile.
    /// @param code Where the code goes.
    /// @param chunk The chunk.
    /// @return Whether it could be emitted.
    [[nodiscard]] bool avx2(asmjit::CodeHolder &code, const Chunk &chunk);

}
