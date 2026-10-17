#include "core/drivers/processor/compiler/neon.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)

#include "core/bytecode/chunk.hpp"
#include "core/bytecode/precision.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/code.hpp"
#include "core/types.hpp"

#include <asmjit/a64.h>

namespace sci::core::drivers::processor::compiler {

    bool neon(asmjit::CodeHolder& code, const Chunk& chunk) {
        using namespace asmjit;

        const bool wide = chunk.precision == Precision::wide;
        const uint32 bytes = wide ? sizeof(double) : sizeof(float);
        const uint32 lanes = 16 / bytes;

        // A register seen as four floats or two doubles.
        const auto shape = [wide](const a64::Vec& vector) { return wide ? vector.d2() : vector.s4(); };

        a64::Compiler compiler(&code);
        FuncNode* function = compiler.add_func(FuncSignature::build<void, const void* const*, const void*, void*, size_t>());
        const a64::Gp inputs = compiler.new_gp_ptr("inputs");
        const a64::Gp constants = compiler.new_gp_ptr("constants");
        const a64::Gp output = compiler.new_gp_ptr("output");
        const a64::Gp count = compiler.new_gpx("count");
        function->set_arg(0, inputs);
        function->set_arg(1, constants);
        function->set_arg(2, output);
        function->set_arg(3, count);

        // Every input the chunk reads gets a pointer that walks its array.
        bool reads[Chunk::capacity]{};
        const auto note = [&](const uint8 identifier) {
            if (identifier != slot::none && (identifier & slot::kinds) == slot::input) reads[identifier & ~slot::kinds] = true;
        };
        for (uint8 step = 0; step < chunk.length; ++step) {
            for (const uint8 source : chunk.code[step].sources) note(source);
        }
        note(chunk.result);
        a64::Gp pointers[Chunk::capacity];
        for (uint8 index = 0; index < chunk.arguments; ++index) {
            if (!reads[index]) continue;
            pointers[index] = compiler.new_gp_ptr("pointer");
            compiler.ldr(pointers[index], a64::ptr(inputs, static_cast<int32_t>(index * sizeof(void*))));
        }

        // Constants are broadcast into registers once, before the loop.
        a64::Vec literals[Chunk::capacity];
        const a64::Gp address = compiler.new_gp_ptr("address");
        for (uint8 index = 0; index < chunk.literals; ++index) {
            literals[index] = compiler.new_vec128("literal");
            compiler.add(address, constants, index * bytes);
            compiler.ld1r(shape(literals[index]), a64::ptr(address));
        }

        const Label loop = compiler.new_label();
        const Label done = compiler.new_label();
        compiler.cbz(count, done);
        compiler.bind(loop);

        // Values of this iteration by value identifier; inputs are loaded the first time they are read.
        a64::Vec values[256];
        bool loaded[Chunk::capacity]{};
        const auto value = [&](const uint8 identifier) -> a64::Vec {
            const uint8 index = identifier & ~slot::kinds;
            if ((identifier & slot::kinds) == slot::literal) return literals[index];
            if ((identifier & slot::kinds) == slot::input && !loaded[index]) {
                values[identifier] = compiler.new_vec128("input");
                compiler.ldr(values[identifier].q(), a64::ptr(pointers[index]));
                loaded[index] = true;
            }
            return values[identifier];
        };

        for (uint8 step = 0; step < chunk.length; ++step) {
            const auto&[code, target, sources] = chunk.code[step];
            const a64::Vec first = value(sources[0]);
            a64::Vec result = compiler.new_vec128("value");
            switch (code) {
                case Code::copy: result = first; break;
                case Code::negate: compiler.fneg(shape(result), shape(first)); break;
                case Code::absolute: compiler.fabs(shape(result), shape(first)); break;
                case Code::root: compiler.fsqrt(shape(result), shape(first)); break;
                case Code::add: compiler.fadd(shape(result), shape(first), shape(value(sources[1]))); break;
                case Code::subtract: compiler.fsub(shape(result), shape(first), shape(value(sources[1]))); break;
                case Code::multiply: compiler.fmul(shape(result), shape(first), shape(value(sources[1]))); break;
                case Code::divide: compiler.fdiv(shape(result), shape(first), shape(value(sources[1]))); break;
                // FMIN and FMAX propagate NaN, like the NEON kernels.
                case Code::minimum: compiler.fmin(shape(result), shape(first), shape(value(sources[1]))); break;
                case Code::maximum: compiler.fmax(shape(result), shape(first), shape(value(sources[1]))); break;
                case Code::fuse:
                    compiler.mov(result.b16(), value(sources[2]).b16());
                    compiler.fmla(shape(result), shape(first), shape(value(sources[1])));
                    break;
                default: return false;
            }
            values[target] = result;
        }

        compiler.str(value(chunk.result).q(), a64::ptr(output));
        compiler.add(output, output, 16);
        for (uint8 index = 0; index < chunk.arguments; ++index) {
            if (reads[index]) compiler.add(pointers[index], pointers[index], 16);
        }
        compiler.subs(count, count, lanes);
        compiler.b_ne(loop);
        compiler.bind(done);
        compiler.end_func();
        return compiler.finalize() == Error::kOk;
    }

}

#else

namespace sci::core::drivers::processor::compiler {

    bool neon(asmjit::CodeHolder&, const Chunk&) {
        return false;
    }

}

#endif
