#pragma once

#include "core/bytecode/fold.hpp"
#include "core/bytecode/opcode.hpp"
#include "core/drivers/cpu/kernels/generic/elementwise.hpp"
#include "core/drivers/cpu/kernels/generic/operation.hpp"
#include "core/drivers/cpu/kernels/generic/reductions.hpp"
#include "core/drivers/cpu/kernels/routines.hpp"
#include "core/drivers/cpu/kernels/table.hpp"
#include "core/types.hpp"

// Only the kernel translation units (src/core/drivers/cpu/kernels/*.cpp) include generic/. Each ISA
// file defines a pack named after its ISA (Fallback, Neon, Avx2, Avx512) wrapping its vector
// registers, then instantiates every kernel in generic/ through compose<Neon>() and so on. Loop tails
// reuse the same code through `narrow`, a one-lane Scalar pack, so results never depend on alignment
// or length.
//
// Every template in generic/ is instantiated with that ISA's pack (Scalar carries it as its tag), so
// each ISA file gets its own copies under its own names: the linker can never swap an AVX-compiled
// kernel into the baseline path. A new ISA file needs a pack name of its own for the same reason.
//
// A pack provides: element, value, bits, mask, width, narrow; load, store, splat, fill; add, sub,
// mul, div, fma (a*b+c), fnma (c-a*b), neg, abs, sqrt, min, max (NaN-propagating); less, greater,
// equal, unordered, test, any, select; tobits, cast, band, bor, bxor, shl<n>, shr<n>.

namespace sci::core::drivers::cpu::kernels {

    template <typename pack>
    constexpr Routines<typename pack::element> assemble() noexcept {
        Routines<typename pack::element> routines{};
        routines.kernels[static_cast<size>(Opcode::copy)] = &unary<pack, Operation<Opcode::copy>>;
        routines.kernels[static_cast<size>(Opcode::neg)] = &unary<pack, Operation<Opcode::neg>>;
        routines.kernels[static_cast<size>(Opcode::abs)] = &unary<pack, Operation<Opcode::abs>>;
        routines.kernels[static_cast<size>(Opcode::sqrt)] = &unary<pack, Operation<Opcode::sqrt>>;
        routines.kernels[static_cast<size>(Opcode::exp)] = &unary<pack, Operation<Opcode::exp>>;
        routines.kernels[static_cast<size>(Opcode::log)] = &unary<pack, Operation<Opcode::log>>;
        routines.kernels[static_cast<size>(Opcode::sin)] = &unary<pack, Operation<Opcode::sin>>;
        routines.kernels[static_cast<size>(Opcode::cos)] = &unary<pack, Operation<Opcode::cos>>;
        routines.kernels[static_cast<size>(Opcode::add)] = &binary<pack, Operation<Opcode::add>>;
        routines.kernels[static_cast<size>(Opcode::sub)] = &binary<pack, Operation<Opcode::sub>>;
        routines.kernels[static_cast<size>(Opcode::mul)] = &binary<pack, Operation<Opcode::mul>>;
        routines.kernels[static_cast<size>(Opcode::div)] = &binary<pack, Operation<Opcode::div>>;
        routines.kernels[static_cast<size>(Opcode::min)] = &binary<pack, Operation<Opcode::min>>;
        routines.kernels[static_cast<size>(Opcode::max)] = &binary<pack, Operation<Opcode::max>>;
        routines.kernels[static_cast<size>(Opcode::fma)] = &ternary<pack>;
        routines.folds[static_cast<size>(Fold::sum)] = &sum<pack>;
        routines.folds[static_cast<size>(Fold::min)] = &extreme<pack, Operation<Opcode::min>>;
        routines.folds[static_cast<size>(Fold::max)] = &extreme<pack, Operation<Opcode::max>>;
        routines.dot = &dot<pack>;
        return routines;
    }

    template <template <typename> class pack>
    constexpr Table compose(const char* name) noexcept {
        return Table{name, assemble<pack<float>>(), assemble<pack<double>>()};
    }

}
