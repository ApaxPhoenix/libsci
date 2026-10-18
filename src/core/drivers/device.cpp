#include "core/drivers/device.hpp"
#include "core/alignment.hpp"
#include "core/assert.hpp"
#include "core/drivers/processor/dispatch.hpp"
#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/drivers/vulkan/compiler.hpp"
#include "core/drivers/vulkan/kernel.hpp"
#include "core/drivers/vulkan/pipeline.hpp"
#include "core/drivers/vulkan/region.hpp"
#include "core/drivers/vulkan/stream.hpp"
#include "core/memory/allocation.hpp"
#include "core/bytecode/accumulator.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/precision.hpp"
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

    // Mirrors Chunk in shaders/common/layout.slang, byte for byte: the GPU reads fields by offset.
    // `reserved` is padding that puts `code` at byte 16 in both; it is never read.
    struct Layout {
        uint32 length;
        uint32 result;
        uint32 reserved[2];
        uint32 code[Chunk::capacity][2];
        float constants[Chunk::capacity];
        uint64 inputs[Chunk::capacity];
    };

    static_assert(offsetof(Layout, code) == 16 && offsetof(Layout, constants) == 528 && offsetof(Layout, inputs) == 784);

    // Mirrors Arguments in shaders/common/layout.slang; `reserved` pads it to the shader's 32 bytes.
    struct Evaluation {
        uint64 chunk;
        uint64 output;
        uint32 count;
        uint32 mode;
        uint32 groups;
        uint32 reserved;
    };

    static_assert(sizeof(Evaluation) == 32);

    // Mirrors Arguments in shaders/product.slang.
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
    // 64 x 64 output tiles for matrix products.
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

    // Everything a GPU device owns. Members are destroyed in reverse order: the stream first, which
    // waits for queued GPU work, then the pipelines that work used, and `context` (the Vulkan device)
    // last, after every object created from it.
    struct Graphics {
        vulkan::Context context;
        vulkan::Allocator allocator;
        vulkan::Pipeline evaluate;
        vulkan::Pipeline reduce;
        vulkan::Pipeline product;
        vulkan::Compiler compiler;
        vulkan::Stream stream;
        std::mutex guard;

        // Takes over an opened context; everything else is built on top of it.
        explicit Graphics(vulkan::Context&& opened)
            : context(std::move(opened)),
              allocator(context),
              evaluate(context, shaders::interpreter, "evaluate", sizeof(Evaluation)),
              reduce(context, shaders::interpreter, "reduce", sizeof(Evaluation)),
              product(context, shaders::product, "product", sizeof(Product)),
              compiler(context, sizeof(Evaluation)),
              stream(context, allocator) {}

        [[nodiscard]] bool valid() const noexcept {
            return evaluate.valid() && reduce.valid() && product.valid();
        }

        // Translates a chunk for interpreter.slang and the JIT's kernels, which read the same block.
        // False when the GPU cannot run it: float64, more temporaries than the interpreter has
        // registers (unless the chunk is `compiled`), or an array that does not live on this device.
        [[nodiscard]] bool encode(const Chunk& chunk, const size count, Layout& layout, const bool compiled) const noexcept {
            if (chunk.precision != Precision::single || (!compiled && chunk.temporaries > registers)) return false;
            if (count > std::numeric_limits<uint32>::max() / 2) return false;

            layout.length = chunk.length;
            layout.result = chunk.result;
            for (uint8 index = 0; index < chunk.length; ++index) {
                const Instruction& step = chunk.code[index];
                layout.code[index][0] = static_cast<uint32>(step.code) | uint32{step.target} << 8 |
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
        if (kind != Target::graphics) return;

        if (vulkan::Context context = vulkan::Context::compose(ordinal); context.valid()) {
            graphics = std::make_unique<Graphics>(std::move(context));
            if (!graphics->valid()) graphics.reset();
        }

        if (!graphics) {
            std::println(stderr, "[sci] no usable Vulkan device at index {}; running on the CPU", ordinal);
            kind = Target::processor;
        }
    }

    Device::~Device() = default;

    const char* Device::name() const noexcept {
        return graphics ? graphics->context.properties.deviceName : "cpu";
    }

    void Device::synchronize() noexcept {
        if (graphics) {
            std::scoped_lock lock(graphics->guard);
            graphics->stream.wait();
        }
    }

    bool Device::ready() const noexcept {
        if (!graphics) return true;
        std::scoped_lock lock(graphics->guard);
        return graphics->stream.ready();
    }

    void Device::allocate(const size bytes, Memory& memory) {
        memory.bytes = bytes;
        memory.target = kind;
        if (graphics) {
            std::scoped_lock lock(graphics->guard);
            memory.buffer = graphics->allocator.compose(bytes);
            memory.host = memory.buffer.host;
        } else {
            memory.allocation = core::memory::Allocation(bytes, alignment::line);
            memory.host = memory.allocation.data();
        }
        ASSERT(memory.host != nullptr);
    }

    void Device::dispose(Memory& memory) noexcept {
        if (memory.target == Target::graphics && graphics) {
            std::scoped_lock lock(graphics->guard);
            graphics->stream.retire(memory.buffer);
        }
        // Host memory frees itself with the allocation.
        memory = {};
    }

    void Device::execute(const Chunk& chunk, void* output, const size count) {
        if (count == 0) return;
        if (graphics) {
            std::unique_lock lock(graphics->guard);
            Layout layout;
            const uint64 destination = graphics->allocator.locate(output);
            const vulkan::Kernel* kernel = destination != 0 ? graphics->compiler.find(chunk) : nullptr;
            if (destination != 0 && graphics->encode(chunk, count, layout, kernel != nullptr)) {
                const vulkan::Pipeline& pipeline = kernel ? *kernel->evaluate : graphics->evaluate;
                const size cost = chunk.cost();
                const size limit = std::max<size>(stride, slice / cost / stride * stride);
                for (size start = 0; start < count; start += limit) {
                    const size length = std::min(limit, count - start);
                    const Evaluation arguments{
                        .chunk = graphics->upload(layout, chunk.arguments, start),
                        .output = destination + start * sizeof(float),
                        .count = static_cast<uint32>(length),
                        .mode = 0,
                        .groups = static_cast<uint32>(std::min<size>((length + stride - 1) / stride, 65535)),
                        .reserved = 0,
                    };
                    graphics->stream.record(pipeline, &arguments, length * cost, arguments.groups);
                }
                return;
            }
            graphics->stream.wait();
        }
        processor::Dispatch::global().execute(chunk, output, count);
    }

    double Device::reduce(const Chunk& chunk, const Fold fold, const size count) {
        if (count == 0) {
            return fold == Fold::sum ? 0.0 : std::numeric_limits<double>::quiet_NaN();
        }
        if (graphics) {
            std::unique_lock lock(graphics->guard);
            Layout layout{};
            const size cost = chunk.cost();
            const bool large = count * cost >= handoff;
            const vulkan::Kernel* kernel = large ? graphics->compiler.find(chunk) : nullptr;
            if (large && graphics->encode(chunk, count, layout, kernel != nullptr)) {
                const vulkan::Pipeline& pipeline = kernel ? *kernel->reduce : graphics->reduce;
                // One dispatch per bounded slice, each leaving one partial per workgroup.
                struct Piece {
                    const float* values;
                    uint32 width;
                };
                std::vector<Piece> pieces;
                const size limit = std::max<size>(stride, slice / cost / stride * stride);
                for (size start = 0; start < count; start += limit) {
                    const size length = std::min(limit, count - start);
                    const auto width = static_cast<uint32>(std::min<size>((length + stride - 1) / stride, 1024));
                    const uint64 address = graphics->upload(layout, chunk.arguments, start);
                    const vulkan::Region partials = graphics->stream.reserve(size{width} * 2 * sizeof(float));
                    const Evaluation arguments{
                        .chunk = address,
                        .output = partials.address,
                        .count = static_cast<uint32>(length),
                        .mode = static_cast<uint32>(fold),
                        .groups = width,
                        .reserved = 0,
                    };
                    graphics->stream.record(pipeline, &arguments, length * cost, width);
                    pieces.push_back({.values = static_cast<const float*>(partials.host), .width = width});
                }
                graphics->stream.wait();

                // Fold the partials in double precision; a NaN anywhere wins.
                Accumulator accumulator(fold);
                bool poisoned = false;
                for (const Piece& piece : pieces) {
                    for (uint32 group = 0; group < piece.width; ++group) {
                        accumulator.add(piece.values[2 * group]);
                        poisoned = poisoned || piece.values[2 * group + 1] != 0.0f;
                    }
                }
                return poisoned ? std::numeric_limits<double>::quiet_NaN() : accumulator.result();
            }
            graphics->stream.wait();
        }
        return processor::Dispatch::global().reduce(chunk, fold, count);
    }

    void Device::product(const Precision precision, const void* left, const void* right, void* output, const size rows, const size columns, const size depth) {
        if (rows == 0 || columns == 0) return;
        if (graphics) {
            std::unique_lock lock(graphics->guard);
            const uint64 first = graphics->allocator.locate(left);
            const uint64 second = graphics->allocator.locate(right);
            const uint64 destination = graphics->allocator.locate(output);
            constexpr size limit = std::numeric_limits<uint32>::max();
            if (precision == Precision::single && first && second && destination && rows < limit && columns < limit && depth < limit) {
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
                    graphics->stream.record(graphics->product, &arguments, height * area,
                                       static_cast<uint32>((columns + tile - 1) / tile),
                                       static_cast<uint32>((height + tile - 1) / tile));
                }
                return;
            }
            graphics->stream.wait();
        }
        processor::Dispatch::product(precision, left, right, output, rows, columns, depth);
    }

}
