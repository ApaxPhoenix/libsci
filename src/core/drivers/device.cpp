#include "core/drivers/device.hpp"
#include "core/arch/alignment.hpp"
#include "core/assert.hpp"
#include "core/drivers/cpu/dispatch.hpp"
#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/drivers/vulkan/pipeline.hpp"
#include "core/drivers/vulkan/region.hpp"
#include "core/drivers/vulkan/stream.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/dtype.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/types.hpp"
#include "shaders/spirv.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <print>
#include <utility>
#include <vector>

namespace sci::core::drivers {

    // Mirrors Chunk in shaders/interpreter.slang.
    struct Layout {
        uint32 length;
        uint32 result;
        uint32 reserved[2];
        uint32 code[Chunk::capacity][2];
        float constants[Chunk::capacity];
        uint64 inputs[Chunk::capacity];
    };

    static_assert(offsetof(Layout, code) == 16 && offsetof(Layout, constants) == 528 && offsetof(Layout, inputs) == 784);

    // Mirrors Arguments in shaders/interpreter.slang.
    struct Evaluation {
        uint64 chunk;
        uint64 output;
        uint32 count;
        uint32 mode;
        uint32 groups;
        uint32 reserved;
    };

    // Mirrors Arguments in shaders/matmul.slang.
    struct Product {
        uint64 left;
        uint64 right;
        uint64 output;
        uint32 rows;
        uint32 columns;
        uint32 depth;
        uint32 first;
    };

    // Shape of the shaders: float4 registers[16], 256 threads x 4 lanes per workgroup step,
    // 64 x 64 output tiles for matmul.
    constexpr uint32 registers = 16;
    constexpr uint32 stride = 256 * 4;
    constexpr uint32 tile = 64;

    // Work per dispatch, in elements x chunk cost or in multiply-adds / 16: on the order of
    // 100 ms on an M1. Larger operations are split, so no single dispatch can trip the macOS GPU
    // watchdog (which takes the whole device down).
    constexpr size slice = size{1} << 30;

    // A reduction has to come back to the host, which costs a GPU round trip (0.5-1 ms on an M1),
    // and with unified memory the CPU streams the same arrays about as fast as the GPU. So only
    // expensive expressions are reduced on the GPU: below this much weighted work (elements x
    // chunk cost) the CPU finishes first, reading the host-mapped memory once pending work is done.
    constexpr size handoff = size{1} << 27;

    static constexpr uint32 groups(const size count, const size limit) noexcept {
        const size needed = (count + stride - 1) / stride;
        return static_cast<uint32>(needed < limit ? needed : limit);
    }

    // Everything a GPU device owns. Members are destroyed in reverse order, so `context` (the Vulkan
    // device) outlives every object created from it.
    struct Gpu {
        vulkan::Context context;
        vulkan::Allocator allocator;
        vulkan::Stream stream;
        vulkan::Pipeline evaluate;
        vulkan::Pipeline reduce;
        vulkan::Pipeline matmul;
        std::mutex guard;

        // Takes over an opened context; everything else is built on top of it.
        explicit Gpu(vulkan::Context&& opened)
            : context(std::move(opened)),
              allocator(context),
              stream(context, allocator),
              evaluate(context, shaders::interpreter, "evaluate", sizeof(Evaluation)),
              reduce(context, shaders::interpreter, "reduce", sizeof(Evaluation)),
              matmul(context, shaders::matmul, "matmul", sizeof(Product)) {}

        [[nodiscard]] bool valid() const noexcept {
            return evaluate.valid() && reduce.valid() && matmul.valid();
        }

        // Translates a chunk for interpreter.slang. False when the GPU cannot run it: float64,
        // too many temporaries, or an array that does not live on this device.
        [[nodiscard]] bool encode(const Chunk& chunk, const size count, Layout& layout) const noexcept {
            if (chunk.type != Dtype::float32 || chunk.temporaries > registers) return false;
            if (count > std::numeric_limits<uint32>::max() / 2) return false;

            layout.length = chunk.length;
            layout.result = chunk.result;
            for (uint8 index = 0; index < chunk.length; ++index) {
                const Instruction& step = chunk.code[index];
                layout.code[index][0] = static_cast<uint32>(step.opcode) | uint32{step.target} << 8 |
                                       uint32{step.sources[0]} << 16 | uint32{step.sources[1]} << 24;
                layout.code[index][1] = step.sources[2];
            }
            for (uint8 index = 0; index < chunk.literals; ++index) {
                layout.constants[index] = static_cast<float>(chunk.constants[index]);
            }
            for (uint8 index = 0; index < chunk.arguments; ++index) {
                layout.inputs[index] = allocator.locate(chunk.inputs[index]);
                if (layout.inputs[index] == 0) return false;
            }
            return true;
        }

        // Copies a chunk into the batch's scratch memory and returns its device address, with
        // every input advanced by `start` elements (for split dispatches).
        [[nodiscard]] uint64 upload(const Layout& layout, const uint8 arguments, const size start) {
            const vulkan::Region region = stream.reserve(sizeof(Layout));
            auto* copy = static_cast<Layout*>(region.host);
            std::memcpy(copy, &layout, sizeof(Layout));
            for (uint8 index = 0; index < arguments; ++index) {
                copy->inputs[index] += start * sizeof(float);
            }
            return region.address;
        }
    };

    Device::Device(const Target target, const uint32 index)
        : kind(target), ordinal(index) {
        if (kind != Target::gpu) return;

        if (vulkan::Context context = vulkan::Context::compose(ordinal); context.valid()) {
            gpu = std::make_unique<Gpu>(std::move(context));
            if (!gpu->valid()) gpu.reset();
        }

        if (!gpu) {
            std::println(stderr, "[sci] no usable Vulkan device at index {}; running on the CPU", ordinal);
            kind = Target::cpu;
        }
    }

    Device::~Device() = default;

    const char* Device::name() const noexcept {
        return gpu ? gpu->context.properties.deviceName : "cpu";
    }

    void Device::synchronize() noexcept {
        if (gpu) {
            std::scoped_lock lock(gpu->guard);
            gpu->stream.wait();
        }
    }

    bool Device::ready() const noexcept {
        if (!gpu) return true;
        std::scoped_lock lock(gpu->guard);
        return gpu->stream.ready();
    }

    void Device::allocate(const size bytes, Memory& memory) {
        memory.bytes = bytes;
        memory.target = kind;
        if (gpu) {
            std::scoped_lock lock(gpu->guard);
            memory.buffer = gpu->allocator.compose(bytes);
            memory.host = memory.buffer.host;
        } else {
            memory.host = ::operator new(bytes, std::align_val_t{arch::line});
        }
        ASSERT(memory.host != nullptr);
    }

    void Device::dispose(Memory& memory) noexcept {
        if (memory.target == Target::gpu && gpu) {
            std::scoped_lock lock(gpu->guard);
            gpu->stream.retire(memory.buffer);
        } else if (memory.host) {
            ::operator delete(memory.host, std::align_val_t{arch::line});
        }
        memory = {};
    }

    void Device::execute(const Chunk& chunk, void* output, const size count) {
        if (count == 0) return;
        if (gpu) {
            std::unique_lock lock(gpu->guard);
            Layout layout;
            const uint64 destination = gpu->allocator.locate(output);
            if (destination != 0 && gpu->encode(chunk, count, layout)) {
                const size cost = chunk.cost();
                const size limit = std::max<size>(stride, slice / cost / stride * stride);
                for (size start = 0; start < count; start += limit) {
                    const size length = std::min(limit, count - start);
                    const Evaluation arguments{
                        .chunk = gpu->upload(layout, chunk.arguments, start),
                        .output = destination + start * sizeof(float),
                        .count = static_cast<uint32>(length),
                        .mode = 0,
                        .groups = groups(length, 65535),
                        .reserved = 0,
                    };
                    gpu->stream.record(gpu->evaluate, &arguments, length * cost, arguments.groups);
                }
                return;
            }
            gpu->stream.wait();
        }
        cpu::Dispatch::global().execute(chunk, output, count);
    }

    double Device::reduce(const Chunk& chunk, const Fold fold, const size count) {
        if (count == 0) {
            return fold == Fold::sum ? 0.0 : std::numeric_limits<double>::quiet_NaN();
        }
        if (gpu) {
            std::unique_lock lock(gpu->guard);
            Layout layout;
            const size cost = chunk.cost();
            if (count * cost >= handoff && gpu->encode(chunk, count, layout)) {
                // One dispatch per bounded slice, each leaving one partial per workgroup.
                struct Piece {
                    const float* values;
                    uint32 width;
                };
                std::vector<Piece> pieces;
                const size limit = std::max<size>(stride, slice / cost / stride * stride);
                for (size start = 0; start < count; start += limit) {
                    const size length = std::min(limit, count - start);
                    const uint32 width = groups(length, 1024);
                    const uint64 address = gpu->upload(layout, chunk.arguments, start);
                    const vulkan::Region partials = gpu->stream.reserve(size{width} * 2 * sizeof(float));
                    const Evaluation arguments{
                        .chunk = address,
                        .output = partials.address,
                        .count = static_cast<uint32>(length),
                        .mode = static_cast<uint32>(fold),
                        .groups = width,
                        .reserved = 0,
                    };
                    gpu->stream.record(gpu->reduce, &arguments, length * cost, width);
                    pieces.push_back({.values = static_cast<const float*>(partials.host), .width = width});
                }
                gpu->stream.wait();

                // Fold the partials in double precision, starting from the fold's identity; a NaN
                // anywhere wins.
                constexpr double infinity = std::numeric_limits<double>::infinity();
                double result = fold == Fold::sum ? 0.0 : (fold == Fold::min ? infinity : -infinity);
                bool nan = false;
                for (const Piece& piece : pieces) {
                    for (uint32 group = 0; group < piece.width; ++group) {
                        const double value = piece.values[2 * group];
                        nan = nan || piece.values[2 * group + 1] != 0.0f;
                        if (fold == Fold::sum) {
                            result += value;
                        } else if (fold == Fold::min) {
                            result = std::min(result, value);
                        } else {
                            result = std::max(result, value);
                        }
                    }
                }
                return nan ? std::numeric_limits<double>::quiet_NaN() : result;
            }
            gpu->stream.wait();
        }
        return cpu::Dispatch::global().reduce(chunk, fold, count);
    }

    void Device::matmul(const Dtype type, const void* left, const void* right, void* output, const size rows, const size columns, const size depth) {
        if (rows == 0 || columns == 0) return;
        if (gpu) {
            std::unique_lock lock(gpu->guard);
            const uint64 first = gpu->allocator.locate(left);
            const uint64 second = gpu->allocator.locate(right);
            const uint64 destination = gpu->allocator.locate(output);
            constexpr size limit = std::numeric_limits<uint32>::max();
            if (type == Dtype::float32 && first && second && destination && rows < limit && columns < limit && depth < limit) {
                // Bands of rows, each a bounded amount of work.
                const size area = std::max<size>(columns * std::max<size>(depth, 1) / 16, 1);
                const size band = std::max<size>(tile, slice / area / tile * tile);
                for (size top = 0; top < rows; top += band) {
                    const size height = std::min(band, rows - top);
                    const Product arguments{
                        .left = first,
                        .right = second,
                        .output = destination,
                        .rows = static_cast<uint32>(rows),
                        .columns = static_cast<uint32>(columns),
                        .depth = static_cast<uint32>(depth),
                        .first = static_cast<uint32>(top),
                    };
                    gpu->stream.record(gpu->matmul, &arguments, height * area,
                                       static_cast<uint32>((columns + tile - 1) / tile),
                                       static_cast<uint32>((height + tile - 1) / tile));
                }
                return;
            }
            gpu->stream.wait();
        }
        cpu::Dispatch::matmul(type, left, right, output, rows, columns, depth);
    }

}
