#pragma once

#include "core/drivers/cpu/kernels/generic/constants.hpp"
#include "core/drivers/cpu/kernels/generic/polynomial.hpp"
#include "core/types.hpp"

#include <limits>

namespace sci::core::drivers::cpu::kernels {

    // Natural logarithm over a whole pack.
    template <typename pack>
    struct Logarithm {
        using type = pack::element;
        using value = pack::value;
        using limit = Constants<type>;

        static value log(const value x) noexcept {
            const auto small = pack::less(x, pack::splat(limit::tiny));
            const value y = pack::select(small, pack::mul(x, pack::splat(limit::lift)), x);
            const auto bits = pack::tobits(y);

            value e = pack::sub(
                pack::cast(pack::bor(pack::template shr<limit::mantissa>(bits), pack::tobits(pack::splat(limit::magnitude)))),
                pack::splat(limit::exponent));
            value m = pack::cast(pack::bor(pack::band(bits, pack::fill(limit::fraction)), pack::fill(limit::one)));

            const auto large = pack::greater(m, pack::splat(limit::sqrt2));
            m = pack::select(large, pack::mul(m, pack::splat(type{0.5})), m);
            e = pack::select(large, pack::add(e, pack::splat(type{1})), e);
            e = pack::select(small, pack::sub(e, pack::splat(limit::shift)), e);

            // fdlibm's form: log(1 + f) = f - (f^2/2 - s * (f^2/2 + R)). The leading f is exact, so the
            // rounding in s only touches a correction term.
            const value f = pack::sub(m, pack::splat(type{1}));
            const value s = pack::div(f, pack::add(f, pack::splat(type{2})));
            const value z = pack::mul(s, s);
            const value r = pack::mul(z, horner<pack, 0>(z, logseries));
            const value half = pack::mul(pack::mul(f, f), pack::splat(type{0.5}));
            const value inner = pack::fma(s, pack::add(half, r), pack::mul(e, pack::splat(limit::ln2lo)));
            value result = pack::sub(pack::mul(e, pack::splat(limit::ln2hi)), pack::sub(pack::sub(half, inner), f));

            constexpr type infinity = std::numeric_limits<type>::infinity();
            constexpr type nan = std::numeric_limits<type>::quiet_NaN();
            result = pack::select(pack::equal(x, pack::splat(infinity)), x, result);
            result = pack::select(pack::equal(x, pack::splat(type{0})), pack::splat(-infinity), result);
            result = pack::select(pack::less(x, pack::splat(type{0})), pack::splat(nan), result);
            return pack::select(pack::unordered(x), x, result);
        }

    private:
        static constexpr auto logseries = reciprocals<type, limit::logdegree>();
    };

}
