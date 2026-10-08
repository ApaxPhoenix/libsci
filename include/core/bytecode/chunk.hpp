#pragma once

#include "core/bytecode/dtype.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/code.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <bit>
#include <string>

// Bytecode for array expressions. Writing `a * b + c` does not compute anything yet: the expression
// is translated into a Chunk (a short list of instructions, like a chunk of Lua bytecode), and the
// chunk runs when the result is needed, in a single loop over the arrays with no temporary arrays.
// The CPU engine, the GPU interpreter (shaders/interpreter.slang) and the JITs (cpu::Jit,
// vulkan::Jit) all run the same Chunk.

namespace sci::core {

    // The recorded operations of one elementwise expression: a flat list of instructions over input
    // arrays, constants and temporaries. Expression nodes record onto it (input, constant, emit),
    // `a * b + c` becomes one fma, and output() or reduce() finishes it. Fixed capacity, so
    // recording never allocates.
    class Chunk {
    public:
        static constexpr size capacity = 64;

        explicit Chunk(const Dtype type) noexcept
            : type(type) {}

        // Recording. Each call returns the value id of what it added.
        [[nodiscard]] uint8 input(const void* pointer) noexcept {
            for (uint8 index = 0; index < arguments; ++index) {
                if (inputs[index] == pointer) return slot::input | index;
            }
            if (arguments == capacity) return fail();
            inputs[arguments] = pointer;
            return slot::input | arguments++;
        }

        [[nodiscard]] uint8 constant(const double value) noexcept {
            for (uint8 index = 0; index < literals; ++index) {
                if (std::bit_cast<uint64>(constants[index]) == std::bit_cast<uint64>(value)) return slot::literal | index;
            }
            if (literals == capacity) return fail();
            constants[literals] = value;
            return slot::literal | literals++;
        }

        // Sources are consumed: expression trees use every temporary exactly once, so a temporary
        // returns to the free list as soon as its consumer is recorded.
        [[nodiscard]] uint8 emit(const Code code, const uint8 first, const uint8 second = slot::none, const uint8 third = slot::none) noexcept {
            if (overflow) return slot::none;

            if (code == Code::add) {
                if (const uint8 merged = contract(second, first); merged != slot::none) return merged;
                if (const uint8 merged = contract(first, second); merged != slot::none) return merged;
            }

            if (length == capacity) return fail();

            release(first);
            release(second);
            release(third);

            const uint8 target = acquire();
            if (target == slot::none) return fail();

            this->code[length++] = Instruction{.code = code, .target = target, .sources = {first, second, third}};
            return target;
        }

        // Finishes a chunk whose result is written to the output array.
        const Chunk& output(const uint8 value) noexcept {
            if (overflow) return *this;
            if (length > 0 && code[length - 1].target == value && (value & slot::kinds) == slot::temporary) {
                code[length - 1].target = slot::output;
            } else if (length < capacity) {
                code[length++] = Instruction{.code = Code::copy, .target = slot::output, .sources = {value, slot::none, slot::none}};
            } else {
                fail();
            }
            result = slot::output;
            return *this;
        }

        // Finishes a chunk whose result feeds a reduction instead of being stored.
        const Chunk& reduce(const uint8 value) noexcept {
            result = value;
            return *this;
        }

        // False once the expression outgrew the fixed capacity; callers then fall back to a loop.
        [[nodiscard]] bool valid() const noexcept {
            return !overflow;
        }

        // Rough per-element work: one unit per array streamed plus each instruction's cost. Decides
        // when a chunk is worth spreading across threads or sending to the GPU.
        [[nodiscard]] uint32 cost() const noexcept {
            uint32 total = arguments + 1;
            for (uint8 index = 0; index < length; ++index) {
                switch (code[index].code) {
                    case Code::div:
                    case Code::sqrt: total += 4; break;
                    case Code::exp:
                    case Code::log: total += 12; break;
                    case Code::sin:
                    case Code::cos:
                    case Code::tan: total += 16; break;
                    case Code::asin:
                    case Code::acos:
                    case Code::atan:
                    case Code::sinh:
                    case Code::cosh:
                    case Code::tanh:
                    case Code::asinh:
                    case Code::acosh:
                    case Code::atanh: total += 24; break;
                    default: total += 1; break;
                }
            }
            return total;
        }

        // The instructions and value counts without the input addresses or constant values: chunks
        // with the same signature compile to the same machine code or shader (see the JITs).
        [[nodiscard]] std::string signature() const {
            std::string key;
            key.reserve(5 + 5 * size{length});
            key += static_cast<char>(type);
            key += static_cast<char>(arguments);
            key += static_cast<char>(literals);
            key += static_cast<char>(temporaries);
            key += static_cast<char>(result);
            for (uint8 index = 0; index < length; ++index) {
                const auto&[code, target, sources] = this->code[index];
                key += static_cast<char>(code);
                key += static_cast<char>(target);
                for (const uint8 source : sources) {
                    key += static_cast<char>(source);
                }
            }
            return key;
        }

        // The recorded chunk, read by the executors. Only the first length / arguments / literals
        // entries are meaningful.
        Dtype type;
        uint8 length{0};
        uint8 arguments{0};
        uint8 literals{0};
        uint8 temporaries{0};
        uint8 result{slot::none};
        Instruction code[capacity]{};
        const void* inputs[capacity]{};
        double constants[capacity]{};

    private:
        // Rewrites `product + addend` into one fma when the product was the last recorded instruction.
        uint8 contract(const uint8 product, const uint8 addend) noexcept {
            if (length == 0 || (product & slot::kinds) != slot::temporary) return slot::none;
            Instruction& last = code[length - 1];
            if (last.code != Code::mul || last.target != product) return slot::none;
            last.code = Code::fma;
            last.sources[2] = addend;
            release(addend);
            return product;
        }

        void release(const uint8 id) noexcept {
            if ((id & slot::kinds) == slot::temporary) {
                busy &= ~(uint64{1} << (id & ~slot::kinds));
            }
        }

        uint8 acquire() noexcept {
            if (~busy == 0) return slot::none;
            const auto index = static_cast<uint8>(std::countr_one(busy));
            busy |= uint64{1} << index;
            temporaries = std::max<uint8>(temporaries, index + 1);
            return slot::temporary | index;
        }

        uint8 fail() noexcept {
            overflow = true;
            return slot::none;
        }

        uint64 busy{0};
        bool overflow{false};
    };

}
