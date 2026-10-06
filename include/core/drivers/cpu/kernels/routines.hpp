#pragma once

#include "core/bytecode/fold.hpp"
#include "core/bytecode/opcode.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    // Every kernel for one element type, indexed by opcode / fold.
    template <typename type>
    struct Routines {
        // Elementwise kernel over one block. The target may alias a source.
        using Kernel = void (*)(const type* const* sources, type* target, size count) noexcept;

        // Reduction over a non-empty run of elements.
        using Reduction = type (*)(const type* source, size count) noexcept;

        // Sum of products of two runs: sum(a * b) without materialising a * b.
        using Contraction = type (*)(const type* left, const type* right, size count) noexcept;

        Kernel kernels[static_cast<size>(Opcode::count)]{};
        Reduction folds[static_cast<size>(Fold::count)]{};
        Contraction dot{nullptr};
    };

}
