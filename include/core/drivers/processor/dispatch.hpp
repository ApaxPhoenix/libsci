#pragma once

/// @file
/// @brief Runs bytecode chunks on the CPU.

#include "core/bytecode/chunk.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/precision.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::processor {

    namespace kernels {
        struct Table;
    }

    /// Runs bytecode chunks on the CPU: block by block, so an expression's temporaries stay in L1, and
    /// across the global pool once a chunk carries enough work to pay for the fork.
    class Dispatch {
    public:
        /// Null runs the fallback kernels.
        /// @param table The kernels, or null.
        explicit Dispatch(const kernels::Table *table) noexcept;

        /// The kernels SCI_ISA names (fallback, neon, avx2, avx512) when set and supported, otherwise
        /// the widest this CPU runs.
        /// @return The dispatcher.
        [[nodiscard]] static const Dispatch &global() noexcept;

        /// @brief Evaluates a chunk over every element, across the pool when it is worth it.
        /// @param chunk The recorded expression.
        /// @param output The output array's first element.
        /// @param count How many elements.
        void execute(const Chunk &chunk, void *output, size count) const;

        /// Sum, minimum or maximum of the chunk result. The minimum and maximum of nothing are NaN.
        /// @param chunk The recorded expression.
        /// @param fold Sum, minimum or maximum.
        /// @param count How many elements.
        /// @return The result.
        [[nodiscard]] double reduce(const Chunk &chunk, Fold fold, size count) const;

        /// Row-major output = left * right, with left rows x depth and right depth x columns.
        /// @param precision Whether the elements are float or double.
        /// @param left The left matrix's first element.
        /// @param right The right matrix's first element.
        /// @param output The product's first element.
        /// @param rows The output's rows.
        /// @param columns The output's columns.
        /// @param depth The shared dimension: left's columns and right's rows.
        static void product(Precision precision, const void *left, const void *right, void *output, size rows,
                            size columns, size depth);

    private:
        /// @brief The kernels, or null for the fallback ones.
        const kernels::Table *table{nullptr};
    };

}
