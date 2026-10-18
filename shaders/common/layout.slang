// Shared by shaders/interpreter.slang and the kernels the GPU JIT generates
// (src/core/drivers/vulkan/compiler.cpp): the chunk block, the arguments, and the helpers that load and
// store four elements per thread.

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

float4 lesser(float4 left, float4 right) {
    return float4(lesser(left.x, right.x), lesser(left.y, right.y), lesser(left.z, right.z), lesser(left.w, right.w));
}

float4 greater(float4 left, float4 right) {
    return float4(greater(left.x, right.x), greater(left.y, right.y), greater(left.z, right.z), greater(left.w, right.w));
}

// Four elements of input `slot`, one per lane.
float4 load(Chunk* chunk, uint slot, uint4 index) {
    float* source = (float*)chunk.inputs[slot];
    return float4(source[index.x], source[index.y], source[index.z], source[index.w]);
}

// Writes the live lanes to the output array.
void store(uint4 index, bool4 live, float4 value) {
    float* destination = arguments.output;
    if (live.x) destination[index.x] = value.x;
    if (live.y) destination[index.y] = value.y;
    if (live.z) destination[index.z] = value.z;
    if (live.w) destination[index.w] = value.w;
}
