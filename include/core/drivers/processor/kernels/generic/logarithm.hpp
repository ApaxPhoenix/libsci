#pragma once

#include "core/drivers/processor/kernels/generic/constants.hpp"
#include "core/drivers/processor/kernels/generic/polynomial.hpp"
#include "core/types.hpp"

#include <limits>

namespace sci::core::drivers::processor::kernels {

    // Natural logarithm over a whole pack.
    template <typename pack>
    struct Logarithm {
        using type = pack::element;
        using value = pack::value;
        using limit = Constants<type>;

        static value logarithm(const value input) noexcept {
            // Subnormal inputs are scaled into the normal range first, and the scale taken back below.
            const auto small = pack::less(input, pack::splat(limit::tiny));
            const value scaled = pack::select(small, pack::multiply(input, pack::splat(limit::lift)), input);
            const auto pattern = pack::raw(scaled);

            // x = 2^e * m with m in [sqrt(2)/2, sqrt(2)).
            value exponent = pack::subtract(
                pack::cast(pack::unite(pack::template shift<-limit::mantissa>(pattern), pack::raw(pack::splat(limit::magnitude)))),
                pack::splat(limit::exponent));
            value mantissa = pack::cast(pack::unite(pack::intersect(pattern, pack::fill(limit::fraction)), pack::fill(limit::one)));

            const auto large = pack::greater(mantissa, pack::splat(limit::root));
            mantissa = pack::select(large, pack::multiply(mantissa, pack::splat(type{0.5})), mantissa);
            exponent = pack::select(large, pack::add(exponent, pack::splat(type{1})), exponent);
            exponent = pack::select(small, pack::subtract(exponent, pack::splat(limit::shift)), exponent);

            // fdlibm's form: log(1 + f) = f - (f^2/2 - s * (f^2/2 + R)) with f = m - 1, s = f / (2 + f)
            // and z = s^2. The leading f is exact, so the rounding in s only touches a correction term.
            const value fraction = pack::subtract(mantissa, pack::splat(type{1}));
            const value ratio = pack::divide(fraction, pack::add(fraction, pack::splat(type{2})));
            const value square = pack::multiply(ratio, ratio);
            const value tail = pack::multiply(square, series.template evaluate<pack, 0>(square));
            const value half = pack::multiply(pack::multiply(fraction, fraction), pack::splat(type{0.5}));
            const value inner = pack::fuse(ratio, pack::add(half, tail), pack::multiply(exponent, pack::splat(limit::remainder)));
            value result = pack::subtract(pack::multiply(exponent, pack::splat(limit::natural)), pack::subtract(pack::subtract(half, inner), fraction));

            constexpr type infinity = std::numeric_limits<type>::infinity();
            constexpr type undefined = std::numeric_limits<type>::quiet_NaN();
            result = pack::select(pack::equal(input, pack::splat(infinity)), input, result);
            result = pack::select(pack::equal(input, pack::splat(type{0})), pack::splat(-infinity), result);
            result = pack::select(pack::less(input, pack::splat(type{0})), pack::splat(undefined), result);
            return pack::select(pack::unordered(input), input, result);
        }

    private:
        // The series tail R(z) (see Polynomial::reciprocals).
        static constexpr auto series = Polynomial<type, limit::logarithm>::reciprocals();
    };

}
