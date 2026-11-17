#pragma once

/// @file
/// @brief Every kernel for one element type, indexed by code / fold.

#include "core/bytecode/code.hpp"
#include "core/bytecode/fold.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    /// Every kernel for one element type, indexed by code / fold.
    template <typename type> struct Routines {
        /// Elementwise kernel over one block. The target may alias a source.
        using Kernel = void (*)(const type *const *sources, type *target, size count) noexcept;

        /// Reduction over a non-empty run of elements.
        using Reduction = type (*)(const type *source, size count) noexcept;

        /// Sum of products of two runs: sum(a * b) without materialising a * b.
        using Contraction = type (*)(const type *left, const type *right, size count) noexcept;

        Kernel kernels[static_cast<size>(Code::count)]{};
        Reduction folds[static_cast<size>(Fold::count)]{};
        Contraction dot{nullptr};
    };

}
