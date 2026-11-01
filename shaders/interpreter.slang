// The GPU bytecode interpreter: runs a core::Chunk (include/core/bytecode/chunk.hpp) over float32
// arrays.
//
// The CPU engine and this shader run the same instructions, so any array expression works on the
// GPU without compiling a shader for it. Each thread carries four elements through the whole chunk,
// so decoding an instruction is paid once per four elements. The GPU JIT later replaces `run` with
// code generated for the chunk (src/core/drivers/vulkan/compiler.cpp); everything else is shared.

#include "common/layout.slang"

float4 fetch(Chunk* chunk, uint identifier, uint4 index, inout float4 registers[16]) {
    const uint kind = identifier >> 6;
    const uint slot = identifier & 63;
    if (kind == input) {
        return load(chunk, slot, index);
    }
    if (kind == literal) {
        return float4(chunk.constants[slot]);
    }
    if (kind == temporary) {
        return registers[slot];
    }
    return float4(0.0);
}

// Applies one operation to its operands. Operations with fewer inputs ignore the later ones. The
// case numbers are the values of core::Code, so they must stay in the same order as the enum.
float4 apply(uint code, float4 alpha, float4 beta, float4 gamma) {
    switch (code) {
        case 0:  return alpha;
        case 1:  return -alpha;
        case 2:  return abs(alpha);
        case 3:  return sqrt(alpha);
        case 4:  return exp(alpha);
        case 5:  return log(alpha);

        case 6:  return sin(alpha);
        case 7:  return cos(alpha);
        case 8:  return tan(alpha);
        case 9:  return asin(alpha);
        case 10: return acos(alpha);
        case 11: return atan(alpha);
        case 12: return 1.0 / sin(alpha);
        case 13: return 1.0 / cos(alpha);
        case 14: return 1.0 / tan(alpha);
        case 15: return asin(1.0 / alpha);
        case 16: return acos(1.0 / alpha);
        case 17: return atan(1.0 / alpha);

        case 18: return sinh(alpha);
        case 19: return cosh(alpha);
        case 20: return tanh(alpha);
        case 21: return asinh(alpha);
        case 22: return acosh(alpha);
        case 23: return atanh(alpha);
        case 24: return 1.0 / sinh(alpha);
        case 25: return 1.0 / cosh(alpha);
        case 26: return 1.0 / tanh(alpha);
        case 27: return asinh(1.0 / alpha);
        case 28: return acosh(1.0 / alpha);
        case 29: return atanh(1.0 / alpha);

        case 30: return alpha + beta;
        case 31: return alpha - beta;
        case 32: return alpha * beta;
        case 33: return alpha / beta;
        case 34: return lesser(alpha, beta);
        case 35: return greater(alpha, beta);
        case 36: return fma(alpha, beta, gamma);

        default: return float4(0.0);
    }
}

// Runs every instruction for four elements and returns the chunk's result.
float4 run(Chunk* chunk, uint4 index, bool4 live) {
    float4 registers[16];
    float4 value = float4(0.0);
    const uint length = chunk.length;
    for (uint step = 0; step < length; ++step) {
        const uint2 word = chunk.code[step];
        const uint code = word.x & 0xFF;
        const uint target = (word.x >> 8) & 0xFF;
        const float4 alpha = fetch(chunk, (word.x >> 16) & 0xFF, index, registers);
        const float4 beta = fetch(chunk, word.x >> 24, index, registers);
        const float4 gamma = fetch(chunk, word.y, index, registers);
        value = apply(code, alpha, beta, gamma);
        if (target == output) {
            store(index, live, value);
        } else {
            registers[target & 63] = value;
        }
    }
    const uint result = chunk.result;
    return result == output ? value : fetch(chunk, result, index, registers);
}

#include "common/entries.slang"