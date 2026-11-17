#pragma once

/// @file
/// @brief The recorded operations of one elementwise expression.

#include "core/bytecode/code.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/precision.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <bit>
#include <string>

// Bytecode for array expressions. Writing `a * b + c` does not compute anything yet: the expression
// is translated into a Chunk (a short list of instructions, like a chunk of Lua bytecode), and the
// chunk runs when the result is needed, in a single loop over the arrays with no temporary arrays.
// The CPU engine, the GPU interpreter (shaders/interpreter.slang) and the JITs (processor::Compiler,
// vulkan::Compiler) all run the same Chunk.

namespace sci::core {

    /// The recorded operations of one elementwise expression: a flat list of instructions over input
    /// arrays, constants and temporaries. Expression nodes record onto it (input, constant, emit),
    /// `a * b + c` becomes one fma, and output() or reduce() finishes it. Fixed capacity, so
    /// recording never allocates.
    class Chunk {
    public:
        /// @brief The most instructions, inputs or constants a chunk holds.
        static constexpr size capacity = 64;

        /// @brief An empty chunk.
        /// @param precision Whether its elements are float or double.
        explicit Chunk(const Precision precision) noexcept : precision(precision) {}

        /// Recording. Each call returns the value id of what it added.
        /// @param pointer The array's first element.
        /// @return The input's value id.
        [[nodiscard]] uint8 input(const void *pointer) noexcept {
            for (uint8 index = 0; index < arguments; ++index) {
                if (inputs[index] == pointer) return slot::input | index;
            }
            if (arguments == capacity) return fail();
            inputs[arguments] = pointer;
            return slot::input | arguments++;
        }

        /// @brief Records a constant, or finds the same one recorded already.
        /// @param value The constant.
        /// @return The constant's value id.
        [[nodiscard]] uint8 constant(const double value) noexcept {
            for (uint8 index = 0; index < literals; ++index) {
                if (std::bit_cast<uint64>(constants[index]) == std::bit_cast<uint64>(value))
                    return slot::literal | index;
            }
            if (literals == capacity) return fail();
            constants[literals] = value;
            return slot::literal | literals++;
        }

        /// Sources are consumed: expression trees use every temporary exactly once, so a temporary
        /// returns to the free list as soon as its consumer is recorded.
        /// @param code The operation.
        /// @param first Its first operand.
        /// @param second Its second, if it takes one.
        /// @param third Its third, if it takes one.
        /// @return The result's value id.
        [[nodiscard]] uint8 emit(const Code code, const uint8 first, const uint8 second = slot::none,
                                 const uint8 third = slot::none) noexcept {
            if (overflow) return slot::none;

            if (code == Code::add) {
                if (const uint8 merged = contract(second, first); merged != slot::none) return merged;
                if (const uint8 merged = contract(first, second); merged != slot::none) return merged;
            }

            if (length == capacity) return fail();

            release(first);
            release(second);
            release(third);

            // The lowest free temporary becomes the target.
            if (~busy == 0) return fail();
            const auto index = static_cast<uint8>(std::countr_one(busy));
            busy |= uint64{1} << index;
            temporaries = std::max<uint8>(temporaries, index + 1);
            const uint8 target = slot::temporary | index;

            instructions[length++] = Instruction{.code = code, .target = target, .sources = {first, second, third}};
            return target;
        }

        /// Finishes a chunk whose result is written to the output array.
        /// @param value The value written out.
        /// @return This chunk.
        const Chunk &output(const uint8 value) noexcept {
            if (overflow) return *this;
            if (length > 0 && instructions[length - 1].target == value && (value & slot::types) == slot::temporary) {
                instructions[length - 1].target = slot::output;
            } else if (length < capacity) {
                instructions[length++] = Instruction{
                        .code = Code::copy, .target = slot::output, .sources = {value, slot::none, slot::none}};
            } else {
                fail();
            }
            result = slot::output;
            return *this;
        }

        /// Finishes a chunk whose result feeds a reduction instead of being stored.
        /// @param value The value reduced.
        /// @return This chunk.
        const Chunk &reduce(const uint8 value) noexcept {
            result = value;
            return *this;
        }

        /// False once the expression outgrew the fixed capacity; callers then fall back to a loop.
        /// @return Whether it fits.
        [[nodiscard]] bool valid() const noexcept { return !overflow; }

        /// Rough per-element work: one unit per array streamed plus each instruction's cost. Decides
        /// when a chunk is worth spreading across threads or sending to the GPU.
        /// @return The cost.
        [[nodiscard]] uint32 cost() const noexcept {
            uint32 total = arguments + 1;
            for (uint8 index = 0; index < length; ++index) {
                if (const Code operation = instructions[index].code;
                    operation == Code::divide || operation == Code::root) {
                    total += 4;
                } else if (operation == Code::exponential || operation == Code::logarithm) {
                    total += 12;
                } else if (operation >= Code::sin && operation <= Code::tan) {
                    total += 16;
                } else if (operation >= Code::asin && operation <= Code::acoth) {
                    total += 24;
                } else {
                    total += 1;
                }
            }
            return total;
        }

        /// A sum whose last instruction multiplies into the result, sum(... * ...): the reduction can
        /// skip that multiply and contract its operands directly (a dot product), so the product
        /// never round-trips through memory.
        /// @param fold The fold.
        /// @return Whether it is.
        [[nodiscard]] bool contracted(const Fold fold) const noexcept {
            return fold == Fold::sum && length > 0 && instructions[length - 1].code == Code::multiply &&
                   instructions[length - 1].target == result;
        }

        /// The instructions and value counts without the input addresses or constant values: chunks
        /// with the same signature compile to the same machine code or shader (see the JITs).
        /// @return The signature.
        [[nodiscard]] std::string signature() const {
            std::string key;
            key.reserve(5 + 5 * size{length});
            key += static_cast<char>(precision);
            key += static_cast<char>(arguments);
            key += static_cast<char>(literals);
            key += static_cast<char>(temporaries);
            key += static_cast<char>(result);
            for (uint8 index = 0; index < length; ++index) {
                const auto &[code, target, sources] = instructions[index];
                key += static_cast<char>(code);
                key += static_cast<char>(target);
                for (const uint8 source : sources) {
                    key += static_cast<char>(source);
                }
            }
            return key;
        }

        /// The recorded chunk, read by the executors. Only the first length / arguments / literals
        /// entries are meaningful.
        Precision precision;
        /// @brief How many instructions.
        uint8 length{0};
        /// @brief How many inputs.
        uint8 arguments{0};
        /// @brief How many constants.
        uint8 literals{0};
        /// @brief How many temporaries it uses at once.
        uint8 temporaries{0};
        /// @brief The value that is the chunk's result.
        uint8 result{slot::none};
        /// @brief The instructions.
        Instruction instructions[capacity]{};
        /// @brief Each input's first element.
        const void *inputs[capacity]{};
        /// @brief The constants.
        double constants[capacity]{};

    private:
        /// Rewrites `product + addend` into one fma when the product was the last recorded instruction.
        /// @param product The product's value id.
        /// @param addend The addend's value id.
        /// @return The fused value's id, or slot::none when it cannot be fused.
        uint8 contract(const uint8 product, const uint8 addend) noexcept {
            if (length == 0 || (product & slot::types) != slot::temporary) return slot::none;
            Instruction &last = instructions[length - 1];
            if (last.code != Code::multiply || last.target != product) return slot::none;
            last.code = Code::fuse;
            last.sources[2] = addend;
            release(addend);
            return product;
        }

        /// @brief Frees a temporary once its consumer is recorded.
        /// @param value The value id.
        void release(const uint8 value) noexcept {
            if ((value & slot::types) == slot::temporary) {
                busy &= ~(uint64{1} << (value & ~slot::types));
            }
        }

        /// @brief Marks the chunk as too large.
        /// @return slot::none.
        uint8 fail() noexcept {
            overflow = true;
            return slot::none;
        }

        /// @brief Which temporaries are in use, a bit each.
        uint64 busy{0};
        /// @brief Whether the expression outgrew the chunk.
        bool overflow{false};
    };

}
