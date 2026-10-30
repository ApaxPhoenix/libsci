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

float4 apply(uint code, float4 first, float4 second, float4 third) {
    switch (code) {
        case 0:  return first;
        case 1:  return -first;
        case 2:  return abs(first);
        case 3:  return sqrt(first);
        case 4:  return exp(first);
        case 5:  return log(first);
        case 6:  return sin(first);
        case 7:  return cos(first);
        case 8:  return tan(first);
        case 9:  return asin(first);
        case 10: return acos(first);
        case 11: return atan(first);
        case 12: return sinh(first);
        case 13: return cosh(first);
        case 14: return tanh(first);
        case 15: return asinh(first);
        case 16: return acosh(first);
        case 17: return atanh(first);
        case 18: return 1.0 / sinh(first);
        case 19: return 1.0 / cosh(first);
        case 20: return 1.0 / tanh(first);
        case 21: return asinh(1.0 / first);
        case 22: return acosh(1.0 / first);
        case 23: return atanh(1.0 / first);
        case 24: return first + second;
        case 25: return first - second;
        case 26: return first * second;
        case 27: return first / second;
        case 28: return lesser(first, second);
        case 29: return greater(first, second);
        default: return fma(first, second, third);
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
        const float4 first = fetch(chunk, (word.x >> 16) & 0xFF, index, registers);
        const float4 second = fetch(chunk, word.x >> 24, index, registers);
        const float4 third = fetch(chunk, word.y, index, registers);
        value = apply(code, first, second, third);
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
