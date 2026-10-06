// The GPU bytecode interpreter: runs a core::Chunk (include/core/bytecode/chunk.hpp) over float32
// arrays.
//
// The CPU engine and this shader run the same instructions, so any array expression works on the
// GPU without compiling a shader for it. Each thread carries four elements through the whole chunk,
// so decoding an instruction is paid once per four elements, and every temporary stays in registers.

// Mirrors `Layout` in src/core/drivers/device.cpp. Input addresses are plain 64-bit integers:
// SPIRV-Cross (and so MoltenVK) mistranslates arrays of pointers inside pointed-to structs.
struct Chunk {
    uint length;
    uint result;
    uint reserved[2];
    uint2 code[64];
    float constants[64];
    uint64_t inputs[64];
};

struct Arguments {
    Chunk* chunk;
    float* output;
    uint count;
    uint mode;
    uint groups;
    uint reserved;
};

[[vk::push_constant]] Arguments arguments;

static const uint lanes = 4;
static const uint threads = 256;

// Value ids: kind in the top two bits, index in the low six.
static const uint input = 0;
static const uint literal = 1;
static const uint temporary = 2;
static const uint output = 0xC0;

float lesser(float left, float right) {
    return isnan(left) || isnan(right) ? left + right : min(left, right);
}

float greater(float left, float right) {
    return isnan(left) || isnan(right) ? left + right : max(left, right);
}

float4 fetch(Chunk* chunk, uint id, uint4 index, inout float4 registers[16]) {
    const uint kind = id >> 6;
    const uint slot = id & 63;
    if (kind == input) {
        float* source = (float*)chunk.inputs[slot];
        return float4(source[index.x], source[index.y], source[index.z], source[index.w]);
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
        case 12: return float4(lesser(a.x, b.x), lesser(a.y, b.y), lesser(a.z, b.z), lesser(a.w, b.w));
        case 13: return float4(greater(a.x, b.x), greater(a.y, b.y), greater(a.z, b.z), greater(a.w, b.w));
        default: return fma(a, b, c);
    }
}

// Runs every instruction for four elements. Lanes past the end read element 0, which always
// exists, and are masked off by the caller.
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
            float* destination = arguments.output;
            if (live.x) destination[index.x] = value.x;
            if (live.y) destination[index.y] = value.y;
            if (live.z) destination[index.z] = value.z;
            if (live.w) destination[index.w] = value.w;
        } else {
            registers[target & 63] = value;
        }
    }
    const uint result = chunk.result;
    return result == output ? value : fetch(chunk, result, index, registers);
}

// Workgroups sweep the array together: each step covers threads * lanes elements per group.
[shader("compute")]
[numthreads(256, 1, 1)]
void evaluate(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    const uint count = arguments.count;
    const uint stride = threads * lanes;
    for (uint first = group.x * stride + local.x; first < count; first += arguments.groups * stride) {
        const uint4 index = first + uint4(0, threads, 2 * threads, 3 * threads);
        const bool4 live = index < count;
        run(arguments.chunk, select(live, index, uint4(0)), live);
    }
}

groupshared float partial[threads / 32];
groupshared uint flagged[threads / 32];

float combine(uint mode, float left, float right) {
    return mode == 0 ? left + right : (mode == 1 ? min(left, right) : max(left, right));
}

// One partial per workgroup in output[2 * group] (value) and output[2 * group + 1] (1 if a NaN
// was seen); min and max skip NaN here and the flag restores NumPy's propagation. The host folds
// the partials in double precision.
[shader("compute")]
[numthreads(256, 1, 1)]
void reduce(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    const uint count = arguments.count;
    const uint mode = arguments.mode;
    const uint stride = threads * lanes;
    const float infinity = asfloat(0x7f800000u);
    const float identity = mode == 0 ? 0.0 : (mode == 1 ? infinity : -infinity);

    float total = identity;
    bool nan = false;
    for (uint first = group.x * stride + local.x; first < count; first += arguments.groups * stride) {
        const uint4 index = first + uint4(0, threads, 2 * threads, 3 * threads);
        const bool4 live = index < count;
        const float4 value = run(arguments.chunk, select(live, index, uint4(0)), live);
        for (uint lane = 0; lane < lanes; ++lane) {
            const float item = live[lane] ? value[lane] : identity;
            nan = nan || isnan(item);
            total = isnan(item) ? total : combine(mode, total, item);
        }
    }

    const float folded = mode == 0 ? WaveActiveSum(total) : (mode == 1 ? WaveActiveMin(total) : WaveActiveMax(total));
    const bool poisoned = WaveActiveAnyTrue(nan);
    const uint width = WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        partial[local.x / width] = folded;
        flagged[local.x / width] = poisoned ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (local.x == 0) {
        float value = partial[0];
        uint seen = flagged[0];
        for (uint wave = 1; wave < threads / width; ++wave) {
            value = combine(mode, value, partial[wave]);
            seen |= flagged[wave];
        }
        arguments.output[2 * group.x] = value;
        arguments.output[2 * group.x + 1] = seen != 0 ? 1.0 : 0.0;
    }
}
