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
        routines.kernels[static_cast<size>(Code::sin)] = &unary<pack, Operation<Code::sin>>;
        routines.kernels[static_cast<size>(Code::cos)] = &unary<pack, Operation<Code::cos>>;
        routines.kernels[static_cast<size>(Code::tan)] = &unary<pack, Operation<Code::tan>>;
        routines.kernels[static_cast<size>(Code::asin)] = &unary<pack, Operation<Code::asin>>;
        routines.kernels[static_cast<size>(Code::acos)] = &unary<pack, Operation<Code::acos>>;
        routines.kernels[static_cast<size>(Code::atan)] = &unary<pack, Operation<Code::atan>>;
        routines.kernels[static_cast<size>(Code::csc)] = &unary<pack, Operation<Code::csc>>;
        routines.kernels[static_cast<size>(Code::sec)] = &unary<pack, Operation<Code::sec>>;
        routines.kernels[static_cast<size>(Code::cot)] = &unary<pack, Operation<Code::cot>>;
        routines.kernels[static_cast<size>(Code::acsc)] = &unary<pack, Operation<Code::acsc>>;
        routines.kernels[static_cast<size>(Code::asec)] = &unary<pack, Operation<Code::asec>>;
        routines.kernels[static_cast<size>(Code::acot)] = &unary<pack, Operation<Code::acot>>;
        routines.kernels[static_cast<size>(Code::sinh)] = &unary<pack, Operation<Code::sinh>>;
        routines.kernels[static_cast<size>(Code::cosh)] = &unary<pack, Operation<Code::cosh>>;
        routines.kernels[static_cast<size>(Code::tanh)] = &unary<pack, Operation<Code::tanh>>;
        routines.kernels[static_cast<size>(Code::asinh)] = &unary<pack, Operation<Code::asinh>>;
        routines.kernels[static_cast<size>(Code::acosh)] = &unary<pack, Operation<Code::acosh>>;
        routines.kernels[static_cast<size>(Code::atanh)] = &unary<pack, Operation<Code::atanh>>;
        routines.kernels[static_cast<size>(Code::csch)] = &unary<pack, Operation<Code::csch>>;
        routines.kernels[static_cast<size>(Code::sech)] = &unary<pack, Operation<Code::sech>>;
        routines.kernels[static_cast<size>(Code::coth)] = &unary<pack, Operation<Code::coth>>;
        routines.kernels[static_cast<size>(Code::acsch)] = &unary<pack, Operation<Code::acsch>>;
        routines.kernels[static_cast<size>(Code::asech)] = &unary<pack, Operation<Code::asech>>;
        routines.kernels[static_cast<size>(Code::acoth)] = &unary<pack, Operation<Code::acoth>>;
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