#pragma once

#include "core/bytecode/chunk.hpp"
#include "core/bytecode/precision.hpp"
#include "core/drivers/processor/kernels/table.hpp"
#include "core/types.hpp"

#include <asmjit/core.h>

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace sci::core::drivers::processor {

    // Compiles chunks into machine code (with asmjit): one loop per chunk in which every value stays
    // in a vector register, so an expression costs one pass with no temporary blocks and no call per
    // instruction. Every instruction is the one the kernel table uses, so results are the kernel
    // tables' to the bit (apart from which NaN comes out when two NaNs meet).
    //
    // ARM64 with NEON (compiler/neon.hpp) and x86-64 with AVX2 (compiler/avx2.hpp). Chunks with the
    // exponential, logarithm, circular or hyperbolic functions, and every chunk on the other tables
    // (fallback, AVX-512), keep running on the kernels.
    // SCI_JIT=0 turns the compiler off and SCI_JIT=trace reports every compile.
    class Compiler {
    public:
        // Runs a compiled chunk over `count` elements, a multiple of width(). inputs[k] and output
        // point at the first element, constants holds the chunk's constants in the element type.
        using Function = void (*)(const void* const* inputs, const void* constants, void* output, size count);

        [[nodiscard]] static Compiler& global();

        // Elements one iteration of compiled code covers, or 0 when this table has no compiler.
        [[nodiscard]] static size width(const kernels::Table& table, Precision precision) noexcept;

        // The compiled chunk, or null when the kernels should run it: not supported, too small, or
        // not worth compiling yet. A chunk compiles the second time it is seen, or the first time when
        // it carries enough work (`count` elements) to pay for compiling.
        [[nodiscard]] Function compile(const Chunk& chunk, const kernels::Table& table, size count);

    private:
        Compiler();

        [[nodiscard]] Function build(const Chunk& chunk, const kernels::Table& table);

        // Created with the first compile: setting it up detects the CPU and reserves executable
        // memory, which small array operations should never pay for.
        std::optional<asmjit::JitRuntime> runtime;
        std::mutex guard;

        // Compiled chunks by signature (null when compiling failed), and how often each chunk not yet
        // compiled has been seen.
        std::unordered_map<std::string, Function> functions;
        std::unordered_map<std::string, uint32> sightings;

        bool enabled{true};
        bool tracing{false};
    };

}
