#pragma once

#include "core/bytecode/fold.hpp"
#include "core/bytecode/code.hpp"
#include "core/drivers/processor/kernels/generic/elementwise.hpp"
#include "core/drivers/processor/kernels/generic/operation.hpp"
#include "core/drivers/processor/kernels/generic/reductions.hpp"
#include "core/drivers/processor/kernels/routines.hpp"
#include "core/drivers/processor/kernels/table.hpp"
#include "core/types.hpp"

// Only the kernel translation units (src/core/drivers/processor/kernels/*.cpp) include generic/. Each ISA
// file defines a pack named after its ISA (Fallback, Neon, Avx2, Avx512) wrapping its vector
// registers, then instantiates every kernel in generic/ through compose<Neon>() and so on. Loop tails
// reuse the same code through `narrow`, a one-lane Scalar pack, so results never depend on alignment
// or length.
//
// Every template in generic/ is instantiated with that ISA's pack (Scalar carries it as its tag), so
// each ISA file gets its own copies under its own names: the linker can never swap an AVX-compiled
// kernel into the baseline path. A new ISA file needs a pack name of its own for the same reason.
//
// A pack provides: element, value, bits, mask, width, narrow; load, store, splat, fill; add,
// subtract, multiply, divide, fuse (a*b+c, one rounding), negate, absolute, root, minimum, maximum
// (NaN-propagating); less, greater, equal, unordered, test, any, select; raw (the bits of a value),
// cast (bits back to a value), intersect, unite, toggle (bitwise and, or, xor) and shift<n> (left
// for positive n, right for negative).

namespace sci::core::drivers::processor::kernels {

    template <typename pack>
    constexpr Routines<typename pack::element> assemble() noexcept {
        Routines<typename pack::element> routines{};
        routines.kernels[static_cast<size>(Code::copy)] = &unary<pack, Operation<Code::copy>>;
        routines.kernels[static_cast<size>(Code::negate)] = &unary<pack, Operation<Code::negate>>;
        routines.kernels[static_cast<size>(Code::absolute)] = &unary<pack, Operation<Code::absolute>>;
        routines.kernels[static_cast<size>(Code::root)] = &unary<pack, Operation<Code::root>>;
        routines.kernels[static_cast<size>(Code::exponential)] = &unary<pack, Operation<Code::exponential>>;
        routines.kernels[static_cast<size>(Code::logarithm)] = &unary<pack, Operation<Code::logarithm>>;
        routines.kernels[static_cast<size>(Code::sine)] = &unary<pack, Operation<Code::sine>>;
        routines.kernels[static_cast<size>(Code::cosine)] = &unary<pack, Operation<Code::cosine>>;
        routines.kernels[static_cast<size>(Code::tangent)] = &unary<pack, Operation<Code::tangent>>;
        routines.kernels[static_cast<size>(Code::arcsine)] = &unary<pack, Operation<Code::arcsine>>;
        routines.kernels[static_cast<size>(Code::arccosine)] = &unary<pack, Operation<Code::arccosine>>;
        routines.kernels[static_cast<size>(Code::arctangent)] = &unary<pack, Operation<Code::arctangent>>;
        routines.kernels[static_cast<size>(hyperbolic::sine)] = &unary<pack, Operation<hyperbolic::sine>>;
        routines.kernels[static_cast<size>(hyperbolic::cosine)] = &unary<pack, Operation<hyperbolic::cosine>>;
        routines.kernels[static_cast<size>(hyperbolic::tangent)] = &unary<pack, Operation<hyperbolic::tangent>>;
        routines.kernels[static_cast<size>(hyperbolic::arcsine)] = &unary<pack, Operation<hyperbolic::arcsine>>;
        routines.kernels[static_cast<size>(hyperbolic::arccosine)] = &unary<pack, Operation<hyperbolic::arccosine>>;
        routines.kernels[static_cast<size>(hyperbolic::arctangent)] = &unary<pack, Operation<hyperbolic::arctangent>>;
        routines.kernels[static_cast<size>(Code::add)] = &binary<pack, Operation<Code::add>>;
        routines.kernels[static_cast<size>(Code::subtract)] = &binary<pack, Operation<Code::subtract>>;
        routines.kernels[static_cast<size>(Code::multiply)] = &binary<pack, Operation<Code::multiply>>;
        routines.kernels[static_cast<size>(Code::divide)] = &binary<pack, Operation<Code::divide>>;
        routines.kernels[static_cast<size>(Code::minimum)] = &binary<pack, Operation<Code::minimum>>;
        routines.kernels[static_cast<size>(Code::maximum)] = &binary<pack, Operation<Code::maximum>>;
        routines.kernels[static_cast<size>(Code::fuse)] = &ternary<pack>;
        routines.folds[static_cast<size>(Fold::sum)] = &sum<pack>;
        routines.folds[static_cast<size>(Fold::minimum)] = &extreme<pack, Operation<Code::minimum>>;
        routines.folds[static_cast<size>(Fold::maximum)] = &extreme<pack, Operation<Code::maximum>>;
        routines.dot = &dot<pack>;
        return routines;
    }

    template <template <typename> class pack>
    constexpr Table compose(const char* name) noexcept {
        return Table{name, assemble<pack<float>>(), assemble<pack<double>>()};
    }

}
