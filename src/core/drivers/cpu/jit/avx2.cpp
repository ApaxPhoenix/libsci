#include "core/drivers/cpu/jit/avx2.hpp"

#if defined(__x86_64__) || defined(_M_X64)

#include "core/bytecode/chunk.hpp"
#include "core/bytecode/dtype.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/opcode.hpp"
#include "core/types.hpp"

#include <asmjit/x86.h>

#include <cstdint>

namespace sci::core::drivers::cpu::jit {

    bool avx2(asmjit::CodeHolder& code, const Chunk& chunk) {
        using namespace asmjit;

        const bool wide = chunk.type == Dtype::float64;
        const uint32 bytes = wide ? sizeof(double) : sizeof(float);
        const uint32 lanes = 32 / bytes;

        x86::Compiler compiler(&code);
        FuncNode* function = compiler.add_func(FuncSignature::build<void, const void* const*, const void*, void*, size_t>());
        function->frame().set_avx_enabled();
        function->frame().set_avx_cleanup();
        const x86::Gp inputs = compiler.new_gp_ptr("inputs");
        const x86::Gp constants = compiler.new_gp_ptr("constants");
        const x86::Gp output = compiler.new_gp_ptr("output");
        const x86::Gp count = compiler.new_gp_ptr("count");
        function->set_arg(0, inputs);
        function->set_arg(1, constants);
        function->set_arg(2, output);
        function->set_arg(3, count);

        // Every input the chunk reads gets a pointer that walks its array.
        bool reads[Chunk::capacity]{};
        bool signs = false;
        const auto note = [&](const uint8 id) {
            if (id != slot::none && (id & slot::kinds) == slot::input) reads[id & ~slot::kinds] = true;
        };
        for (uint8 step = 0; step < chunk.length; ++step) {
            for (const uint8 source : chunk.code[step].sources) note(source);
            signs = signs || chunk.code[step].opcode == Opcode::neg || chunk.code[step].opcode == Opcode::abs;
        }
        note(chunk.result);
        x86::Gp pointers[Chunk::capacity];
        for (uint8 index = 0; index < chunk.arguments; ++index) {
            if (!reads[index]) continue;
            pointers[index] = compiler.new_gp_ptr("pointer");
            compiler.mov(pointers[index], x86::ptr(inputs, static_cast<int32_t>(index * sizeof(void*))));
        }

        // Constants are broadcast into registers once, before the loop.
        x86::Vec literals[Chunk::capacity];
        for (uint8 index = 0; index < chunk.literals; ++index) {
            literals[index] = compiler.new_vec256("literal");
            const x86::Mem constant = x86::ptr(constants, static_cast<int32_t>(index * bytes));
            if (wide) {
                compiler.vbroadcastsd(literals[index], constant);
            } else {
                compiler.vbroadcastss(literals[index], constant);
            }
        }

        // The sign bit of every lane (-0.0), for neg and abs as the AVX2 kernels compute them.
        x86::Vec sign;
        if (signs) {
            std::uint64_t pattern[4];
            for (std::uint64_t& word : pattern) word = wide ? 0x8000000000000000ull : 0x8000000080000000ull;
            sign = compiler.new_vec256("sign");
            compiler.vmovups(sign, compiler.new_const(ConstPoolScope::kLocal, pattern, sizeof(pattern)));
        }

        const Label loop = compiler.new_label();
        const Label done = compiler.new_label();
        compiler.test(count, count);
        compiler.jz(done);
        compiler.bind(loop);

        // Values of this iteration by value id; inputs are loaded the first time they are read.
        x86::Vec values[256];
        bool loaded[Chunk::capacity]{};
        const auto value = [&](const uint8 id) -> x86::Vec {
            const uint8 index = id & ~slot::kinds;
            if ((id & slot::kinds) == slot::literal) return literals[index];
            if ((id & slot::kinds) == slot::input && !loaded[index]) {
                values[id] = compiler.new_vec256("input");
                compiler.vmovups(values[id], x86::ptr(pointers[index]));
                loaded[index] = true;
            }
            return values[id];
        };

        // NaN-propagating min / max, as the AVX2 kernels compute them: where `left` is NaN take it,
        // otherwise min/max(left, right), which returns `right` when `right` is NaN.
        const auto extreme = [&](const x86::Vec& result, const x86::Vec& left, const x86::Vec& right, const bool lesser) {
            const x86::Vec candidate = compiler.new_vec256("candidate");
            const x86::Vec unordered = compiler.new_vec256("unordered");
            if (wide) {
                if (lesser) compiler.vminpd(candidate, left, right); else compiler.vmaxpd(candidate, left, right);
                compiler.vcmppd(unordered, left, left, Imm(3));
                compiler.vblendvpd(result, candidate, left, unordered);
            } else {
                if (lesser) compiler.vminps(candidate, left, right); else compiler.vmaxps(candidate, left, right);
                compiler.vcmpps(unordered, left, left, Imm(3));
                compiler.vblendvps(result, candidate, left, unordered);
            }
        };

        for (uint8 step = 0; step < chunk.length; ++step) {
            const Instruction& instruction = chunk.code[step];
            const x86::Vec first = value(instruction.sources[0]);
            x86::Vec result = compiler.new_vec256("value");
            switch (instruction.opcode) {
                case Opcode::copy: result = first; break;
                case Opcode::neg:
                    if (wide) compiler.vxorpd(result, first, sign); else compiler.vxorps(result, first, sign);
                    break;
                case Opcode::abs:
                    if (wide) compiler.vandnpd(result, sign, first); else compiler.vandnps(result, sign, first);
                    break;
                case Opcode::sqrt:
                    if (wide) compiler.vsqrtpd(result, first); else compiler.vsqrtps(result, first);
                    break;
                case Opcode::add:
                    if (wide) compiler.vaddpd(result, first, value(instruction.sources[1])); else compiler.vaddps(result, first, value(instruction.sources[1]));
                    break;
                case Opcode::sub:
                    if (wide) compiler.vsubpd(result, first, value(instruction.sources[1])); else compiler.vsubps(result, first, value(instruction.sources[1]));
                    break;
                case Opcode::mul:
                    if (wide) compiler.vmulpd(result, first, value(instruction.sources[1])); else compiler.vmulps(result, first, value(instruction.sources[1]));
                    break;
                case Opcode::div:
                    if (wide) compiler.vdivpd(result, first, value(instruction.sources[1])); else compiler.vdivps(result, first, value(instruction.sources[1]));
                    break;
                case Opcode::min: extreme(result, first, value(instruction.sources[1]), true); break;
                case Opcode::max: extreme(result, first, value(instruction.sources[1]), false); break;
                case Opcode::fma:
                    compiler.vmovaps(result, value(instruction.sources[2]));
                    if (wide) {
                        compiler.vfmadd231pd(result, first, value(instruction.sources[1]));
                    } else {
                        compiler.vfmadd231ps(result, first, value(instruction.sources[1]));
                    }
                    break;
                default: return false;
            }
            values[instruction.target] = result;
        }

        compiler.vmovups(x86::ptr(output), value(chunk.result));
        compiler.add(output, 32);
        for (uint8 index = 0; index < chunk.arguments; ++index) {
            if (reads[index]) compiler.add(pointers[index], 32);
        }
        compiler.sub(count, lanes);
        compiler.jnz(loop);
        compiler.bind(done);
        compiler.end_func();
        return compiler.finalize() == Error::kOk;
    }

}

#else

namespace sci::core::drivers::cpu::jit {

    bool avx2(asmjit::CodeHolder&, const Chunk&) {
        return false;
    }

}

#endif
