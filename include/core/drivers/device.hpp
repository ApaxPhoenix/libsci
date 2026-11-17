#pragma once

/// @file
/// @brief A compute device, in the spirit of torch.device / cupy.cuda.Device.

#include "core/bytecode/chunk.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/precision.hpp"
#include "core/drivers/memory.hpp"
#include "core/drivers/target.hpp"
#include "core/types.hpp"

#include <memory>

namespace sci::core::drivers {

    struct Graphics;

    /// A compute device, in the spirit of torch.device / cupy.cuda.Device:
    ///
    ///     Device graphics(Target::graphics);
    ///     auto first = values.to(graphics);                   // arrays on the GPU
    ///     Array<float, 1> second = first * 2.0f + 1.0f;       // runs there, asynchronously
    ///     graphics.synchronize();                             // wait for it (reading an array waits by itself)
    ///
    /// GPU work is queued and submitted in batches; results are ready once synchronize() returns.
    class Device {
    public:
        /// Falls back to the CPU (target() says so) when no usable Vulkan device exists.
        /// @param target The kind of device to open.
        /// @param index Which of them.
        explicit Device(Target target = Target::graphics, uint32 index = 0);
        ~Device();

        Device(const Device &) = delete;
        Device &operator=(const Device &) = delete;

        /// @brief Whether it is the CPU or a GPU.
        /// @return The target.
        [[nodiscard]] Target target() const noexcept { return type; }
        /// @brief Which device of its kind it is.
        /// @return The index.
        [[nodiscard]] uint32 index() const noexcept { return ordinal; }
        /// @brief The device's name.
        /// @return The name.
        [[nodiscard]] const char *name() const noexcept;

        /// Blocks until all queued GPU work has finished. Cheap when nothing is queued.
        void synchronize() noexcept;

        /// Engine entry points, used by Array and the ops; GPU work falls back to the CPU (after a
        /// synchronize) when the GPU cannot run it: float64, or arrays that live elsewhere.
        /// @param bytes How many bytes.
        /// @param memory Receives the memory.
        void allocate(size bytes, Memory &memory);
        /// @brief Frees memory allocate() gave.
        /// @param memory The memory.
        void dispose(Memory &memory) noexcept;
        /// @brief Evaluates a chunk over every element, writing the results.
        /// @param chunk The recorded expression.
        /// @param output The output array's first element.
        /// @param count How many elements.
        void execute(const Chunk &chunk, void *output, size count);
        /// @brief Reduces a chunk's result over every element.
        /// @param chunk The recorded expression.
        /// @param fold Sum, minimum or maximum.
        /// @param count How many elements.
        /// @return The sum, minimum or maximum.
        [[nodiscard]] double reduce(const Chunk &chunk, Fold fold, size count);

        /// Row-major output = left * right, with left rows x depth and right depth x columns.
        /// @param precision Whether the elements are float or double.
        /// @param left The left matrix's first element.
        /// @param right The right matrix's first element.
        /// @param output The product's first element.
        /// @param rows The output's rows.
        /// @param columns The output's columns.
        /// @param depth The shared dimension: left's columns and right's rows.
        void product(Precision precision, const void *left, const void *right, void *output, size rows, size columns,
                     size depth);

    private:
        /// @brief Whether it is the CPU or a GPU.
        Target type{Target::processor};
        /// @brief Which device of its kind it is.
        uint32 ordinal{0};
        /// @brief The GPU and what runs on it; null for the CPU.
        std::unique_ptr<Graphics> graphics;
    };

}
