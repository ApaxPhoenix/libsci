#pragma once

#include "core/drivers/memory.hpp"
#include "core/drivers/target.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/dtype.hpp"
#include "core/bytecode/fold.hpp"
#include "core/types.hpp"

#include <memory>

namespace sci::core::drivers {

    struct Gpu;

    // A compute device, in the spirit of torch.device / cupy.cuda.Device:
    //
    //     Device gpu(Target::gpu);
    //     auto a = host.to(gpu);       // arrays on the GPU
    //     NDArray<float, 1> b = a * 2.0f + 1.0f;   // runs there, asynchronously
    //     gpu.synchronize();           // wait for it (reading an array waits by itself)
    //
    // GPU work is queued and submitted in batches; results are ready once synchronize() returns.
    class Device {
    public:
        // Falls back to the CPU (target() says so) when no usable Vulkan device exists.
        explicit Device(Target target = Target::gpu, uint32 index = 0);
        ~Device();

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        [[nodiscard]] Target target() const noexcept { return kind; }
        [[nodiscard]] uint32 index() const noexcept { return ordinal; }
        [[nodiscard]] const char* name() const noexcept;

        // Blocks until all queued GPU work has finished. Cheap when nothing is queued.
        void synchronize() noexcept;

        // True when no GPU work is queued or running.
        [[nodiscard]] bool ready() const noexcept;

        // Engine entry points, used by NDArray and the ops; GPU work falls back to the CPU (after a
        // synchronize) when the GPU cannot run it: float64, or arrays that live elsewhere.
        void allocate(size bytes, Memory& memory);
        void dispose(Memory& memory) noexcept;
        void execute(const Chunk& chunk, void* output, size count);
        [[nodiscard]] double reduce(const Chunk& chunk, Fold fold, size count);

        // Row-major output = left * right, with left rows x depth and right depth x columns.
        void matmul(Dtype type, const void* left, const void* right, void* output, size rows, size columns, size depth);

    private:
        Target kind{Target::cpu};
        uint32 ordinal{0};
        std::unique_ptr<Gpu> gpu;
    };

}
