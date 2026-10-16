#pragma once

#include "core/bytecode/chunk.hpp"
#include "core/drivers/processor/compiler.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::processor {

    // A chunk the Compiler turned into machine code, ready to run: the function, the width of its
    // registers in elements, and the chunk's constants in the element type, as the code reads them.
    // Empty (false) when the chunk was not compiled; the kernels run it then.
    template <typename type>
    class Program {
    public:
        Program(const Compiler::Function function, const Chunk& chunk, const size width) noexcept
            : function(function), chunk(chunk), width(width) {
            if (!function) return;
            for (size index = 0; index < chunk.literals; ++index) {
                constants[index] = static_cast<type>(chunk.constants[index]);
            }
        }

        explicit operator bool() const noexcept { return function != nullptr; }

        // How many of `count` elements the compiled code covers: whole registers only.
        [[nodiscard]] size cover(const size count) const noexcept { return count / width * width; }

        // Runs over elements [first, first + count) of the inputs, writing to `output` (where element
        // `first` goes). `count` must be a multiple of the width.
        void run(type* output, const size first, const size count) const {
            const void* inputs[Chunk::capacity];
            for (size index = 0; index < chunk.arguments; ++index) {
                inputs[index] = static_cast<const type*>(chunk.inputs[index]) + first;
            }
            function(inputs, constants, output, count);
        }

    private:
        Compiler::Function function;
        const Chunk& chunk;
        size width;
        type constants[Chunk::capacity];
    };

}
