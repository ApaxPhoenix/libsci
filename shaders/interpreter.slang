// The GPU bytecode interpreter: runs a core::Chunk (include/core/bytecode/chunk.hpp) over float32
// arrays.
//
// The CPU engine and this shader run the same instructions, so any array expression works on the
// GPU without compiling a shader for it. Each thread carries four elements through the whole chunk,
// so decoding an instruction is paid once per four elements. The GPU JIT later replaces `run` with
// code generated for the chunk (src/core/drivers/vulkan/jit.cpp); everything else is shared.

#include "common/layout.slang"

float4 fetch(Chunk* chunk, uint id, uint4 index, inout float4 registers[16]) {
    const uint kind = id >> 6;
    const uint slot = id & 63;
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

float4 apply(uint code, float4 a, float4 b, float4 c) {
    switch (code) {
        case 0: return a;
        case 1: return -a;
        case 2: return abs(a);
        case 3: return sqrt(a);
        case 4: return exp(a);
        case 5: return log(a);
        case 6: return sin(a);
        case 7: return cos(a);
        case 8: return a + b;
        case 9: return a - b;
        case 10: return a * b;
        case 11: return a / b;
        case 12: return lesser(a, b);
        case 13: return greater(a, b);
        default: return fma(a, b, c);
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
        const float4 a = fetch(chunk, (word.x >> 16) & 0xFF, index, registers);
        const float4 b = fetch(chunk, word.x >> 24, index, registers);
        const float4 c = fetch(chunk, word.y, index, registers);
        value = apply(code, a, b, c);
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
