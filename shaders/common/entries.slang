// The entry points, shared by shaders/interpreter.slang and the kernels the GPU JIT generates. Both
// call `run(chunk, index, live)`, which computes four elements per thread (and writes them when the
// chunk has an output) and returns the chunk's result.

// The chunk block, the arguments and the thread counts, unless the including file already has them.
// Including them here also lets an editor check this file on its own.
#ifndef LAYOUT
#include "layout.slang"
#endif

// Defined by the file that includes this one: the interpreter's bytecode loop, or the code the GPU
// JIT generates for one chunk.
float4 run(Chunk* chunk, uint4 index, bool4 live);

// Workgroups sweep the array together: each step covers threads * lanes elements per group. Lanes
// past the end read element 0, which always exists, and are masked off.
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
    bool invalid = false;
    for (uint first = group.x * stride + local.x; first < count; first += arguments.groups * stride) {
        const uint4 index = first + uint4(0, threads, 2 * threads, 3 * threads);
        const bool4 live = index < count;
        const float4 value = run(arguments.chunk, select(live, index, uint4(0)), live);
        for (uint lane = 0; lane < lanes; ++lane) {
            const float item = live[lane] ? value[lane] : identity;
            invalid = invalid || isnan(item);
            total = isnan(item) ? total : combine(mode, total, item);
        }
    }

    const float folded = mode == 0 ? WaveActiveSum(total) : (mode == 1 ? WaveActiveMin(total) : WaveActiveMax(total));
    const bool poisoned = WaveActiveAnyTrue(invalid);
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
