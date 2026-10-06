#pragma once

#include "core/drivers/cpu/kernels/generic/constants.hpp"
#include "core/drivers/cpu/kernels/generic/polynomial.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    // e^x over a whole pack.
    template <typename pack>
    struct Exponential {
        using type = pack::element;
        using value = pack::value;
        using limit = Constants<type>;

        static value exp(value x) noexcept {
            x = pack::max(pack::min(x, pack::splat(limit::upper)), pack::splat(limit::lower));

            const value k = pack::sub(pack::fma(x, pack::splat(limit::log2e), pack::splat(limit::round)), pack::splat(limit::round));
            value r = pack::fnma(k, pack::splat(limit::ln2hi), x);
            r = pack::fnma(k, pack::splat(limit::ln2lo), r);

            const value p = pack::fma(r, horner<pack, 1>(r, expseries), pack::splat(type{1}));

            // Usually 2^k is a normal number and one multiply scales the result. Near overflow or
            // underflow the scale is split in two, so subnormal and infinite results still round correctly.
            const value edge = pack::splat(limit::normal);
            if (!pack::any(pack::greater(pack::abs(k), edge))) [[likely]] {
                return pack::mul(p, power(k));
            }
            const value half = pack::sub(pack::fma(k, pack::splat(type{0.5}), pack::splat(limit::round)), pack::splat(limit::round));
            const value rest = pack::sub(k, half);
            return pack::mul(pack::mul(p, power(half)), power(rest));
        }

    private:
        static constexpr auto expseries = factorials<type, limit::expdegree + 1>();

        // 2^k for integral k in the normal exponent range, built straight from the exponent bits.
        static value power(const value k) noexcept {
            return pack::cast(pack::template shl<limit::mantissa>(pack::tobits(pack::add(k, pack::splat(limit::exponent)))));
        }
    };

}
