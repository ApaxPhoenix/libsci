#pragma once

/// @file
/// @brief A chunk the Compiler turned into machine code, ready to run.

#include "core/bytecode/chunk.hpp"
#include "core/drivers/processor/compiler.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::processor {

    /// A chunk the Compiler turned into machine code, ready to run: the function, the width of its
    /// registers in elements, and the chunk's constants in the element type, as the code reads them.
    /// Empty (false) when the chunk was not compiled; the kernels run it then.
    template <typename type> class Program {
    public:
        /// @brief A compiled chunk, with its constants in the element type.
        /// @param function The compiled code.
        /// @param chunk The chunk.
        /// @param width Elements one iteration covers.
        Program(const Compiler::Function function, const Chunk &chunk, const size width) noexcept
            : function(function), chunk(chunk), width(width) {
            if (!function) return;
            for (size index = 0; index < chunk.literals; ++index) {
                constants[index] = static_cast<type>(chunk.constants[index]);
            }
        }

        /// @brief Whether there is compiled code.
        /// @return Whether there is.
        explicit operator bool() const noexcept { return function != nullptr; }

        /// How many of `count` elements the compiled code covers: whole registers only.
        /// @param count How many elements.
        /// @return How many it covers.
        [[nodiscard]] size cover(const size count) const noexcept { return count / width * width; }

        /// Runs over elements [first, first + count) of the inputs, writing to `output` (where element
        /// `first` goes). `count` must be a multiple of the width.
        /// @param output The output's first element.
        /// @param first The first element.
        /// @param count How many.
        void run(type *output, const size first, const size count) const {
            const void *inputs[Chunk::capacity];
            for (size index = 0; index < chunk.arguments; ++index) {
                inputs[index] = static_cast<const type *>(chunk.inputs[index]) + first;
            }
            function(inputs, constants, output, count);
        }

    private:
        /// @brief The compiled code.
        Compiler::Function function;
        /// @brief The chunk.
        const Chunk &chunk;
        /// @brief Elements one iteration covers.
        size width;
        /// @brief The chunk's constants, in the element type.
        type constants[Chunk::capacity];
    };

}
