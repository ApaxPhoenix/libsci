#pragma once

#include "core/drivers/processor/kernels/generic/constants.hpp"
#include "core/drivers/processor/kernels/generic/polynomial.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    // e^x over a whole pack.
    template <typename pack>
    struct Exponential {
        using type = pack::element;
        using value = pack::value;
        using limit = Constants<type>;

        static value exponential(const value input) noexcept {
            const value clamped = pack::maximum(pack::minimum(input, pack::splat(limit::upper)), pack::splat(limit::lower));

            // e^x = 2^k * e^r with k the nearest integer to x / ln 2 and r = x - k ln 2.
            const value exponent = pack::subtract(pack::fuse(clamped, pack::splat(limit::binary), pack::splat(limit::round)), pack::splat(limit::round));
            const value negative = pack::negate(exponent);
            value reduced = pack::fuse(negative, pack::splat(limit::natural), clamped);
            reduced = pack::fuse(negative, pack::splat(limit::remainder), reduced);

            const value significand = pack::fuse(reduced, series.template evaluate<pack, 1>(reduced), pack::splat(type{1}));

            // Usually 2^k is a normal number and one multiply scales the result. Near overflow or
            // underflow the scale is split in two, so subnormal and infinite results still round correctly.
            const value edge = pack::splat(limit::normal);
            if (!pack::any(pack::greater(pack::absolute(exponent), edge))) [[likely]] {
                return pack::multiply(significand, scale(exponent));
            }
            const value half = pack::subtract(pack::fuse(exponent, pack::splat(type{0.5}), pack::splat(limit::round)), pack::splat(limit::round));
            const value rest = pack::subtract(exponent, half);
            return pack::multiply(pack::multiply(significand, scale(half)), scale(rest));
        }

    private:
        // Taylor series of e^r.
        static constexpr auto series = Polynomial<type, limit::exponential + 1>::factorials();

        // 2^k for integral k in the normal exponent range, built straight from the exponent bits.
        static value scale(const value exponent) noexcept {
            return pack::cast(pack::template shift<limit::mantissa>(pack::raw(pack::add(exponent, pack::splat(limit::exponent)))));
        }
    };

}
