#pragma once

#include "core/bytecode/fold.hpp"
#include "core/bytecode/code.hpp"
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
        routines.kernels[static_cast<size>(Code::copy)] = &unary<pack, Operation<Code::copy>>;
        routines.kernels[static_cast<size>(Code::neg)] = &unary<pack, Operation<Code::neg>>;
        routines.kernels[static_cast<size>(Code::abs)] = &unary<pack, Operation<Code::abs>>;
        routines.kernels[static_cast<size>(Code::sqrt)] = &unary<pack, Operation<Code::sqrt>>;
        routines.kernels[static_cast<size>(Code::exp)] = &unary<pack, Operation<Code::exp>>;
        routines.kernels[static_cast<size>(Code::log)] = &unary<pack, Operation<Code::log>>;
        routines.kernels[static_cast<size>(Code::sin)] = &unary<pack, Operation<Code::sin>>;
        routines.kernels[static_cast<size>(Code::cos)] = &unary<pack, Operation<Code::cos>>;
        routines.kernels[static_cast<size>(Code::tan)] = &unary<pack, Operation<Code::tan>>;
        routines.kernels[static_cast<size>(Code::asin)] = &unary<pack, Operation<Code::asin>>;
        routines.kernels[static_cast<size>(Code::acos)] = &unary<pack, Operation<Code::acos>>;
        routines.kernels[static_cast<size>(Code::atan)] = &unary<pack, Operation<Code::atan>>;
        routines.kernels[static_cast<size>(Code::sinh)] = &unary<pack, Operation<Code::sinh>>;
        routines.kernels[static_cast<size>(Code::cosh)] = &unary<pack, Operation<Code::cosh>>;
        routines.kernels[static_cast<size>(Code::tanh)] = &unary<pack, Operation<Code::tanh>>;
        routines.kernels[static_cast<size>(Code::asinh)] = &unary<pack, Operation<Code::asinh>>;
        routines.kernels[static_cast<size>(Code::acosh)] = &unary<pack, Operation<Code::acosh>>;
        routines.kernels[static_cast<size>(Code::atanh)] = &unary<pack, Operation<Code::atanh>>;
        routines.kernels[static_cast<size>(Code::add)] = &binary<pack, Operation<Code::add>>;
        routines.kernels[static_cast<size>(Code::sub)] = &binary<pack, Operation<Code::sub>>;
        routines.kernels[static_cast<size>(Code::mul)] = &binary<pack, Operation<Code::mul>>;
        routines.kernels[static_cast<size>(Code::div)] = &binary<pack, Operation<Code::div>>;
        routines.kernels[static_cast<size>(Code::min)] = &binary<pack, Operation<Code::min>>;
        routines.kernels[static_cast<size>(Code::max)] = &binary<pack, Operation<Code::max>>;
        routines.kernels[static_cast<size>(Code::fma)] = &ternary<pack>;
        routines.folds[static_cast<size>(Fold::sum)] = &sum<pack>;
        routines.folds[static_cast<size>(Fold::min)] = &extreme<pack, Operation<Code::min>>;
        routines.folds[static_cast<size>(Fold::max)] = &extreme<pack, Operation<Code::max>>;
        routines.dot = &dot<pack>;
        return routines;
    }

    template <template <typename> class pack>
    constexpr Table compose(const char* name) noexcept {
        return Table{name, assemble<pack<float>>(), assemble<pack<double>>()};
    }

}
