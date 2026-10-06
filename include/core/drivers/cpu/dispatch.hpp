#pragma once

#include "core/drivers/cpu/context.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/dtype.hpp"
#include "core/bytecode/fold.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu {

    namespace kernels {
        struct Table;
    }

    // Runs bytecode chunks on the CPU: block by block, so an expression's temporaries stay in L1, and across the
    // global pool once a chunk carries enough work to pay for the fork.
    class Dispatch {
    public:
        explicit Dispatch(Context context = Context::probe()) noexcept;
        explicit Dispatch(const kernels::Table* table) noexcept;

        // Uses SCI_ISA (fallback, neon, avx2, avx512) when set and supported.
        [[nodiscard]] static const Dispatch& global() noexcept;

        // Null when the named ISA is unknown or unsupported on this machine.
        [[nodiscard]] static const kernels::Table* select(const char* name) noexcept;

        [[nodiscard]] const char* name() const noexcept;

        void execute(const Chunk& chunk, void* output, size count) const;

        // Sum, min or max of the chunk result. Min and max of nothing are NaN.
        [[nodiscard]] double reduce(const Chunk& chunk, Fold kind, size count) const;

        // Row-major output = left * right, with left rows x depth and right depth x columns.
        static void matmul(Dtype type, const void* left, const void* right, void* output, size rows, size columns, size depth);

    private:
        const kernels::Table* table{nullptr};
    };

}
