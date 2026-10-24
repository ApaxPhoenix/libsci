#pragma once

#include "core/drivers/processor/kernels/generic/constants.hpp"
#include "core/drivers/processor/kernels/generic/polynomial.hpp"
#include "core/types.hpp"

// The C functions (::sinf, ::cos, ...), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::processor::kernels {

    // Sine and cosine over a whole pack, each through its own polynomial. Tangent is their quotient
    // and the inverses go through the C library (see Operation).
    template <typename pack>
    struct Trigonometry {
        using type = pack::element;
        using value = pack::value;
        using limit = Constants<type>;

        // `phase` false gives the sine, true the cosine (the sine a quarter turn later).
        template <bool phase>
        static value evaluate(const value input) noexcept {
            // Nearest whole number of quarter turns, j; the low bits of `shifted` hold j mod 4.
            const value shifted = pack::fuse(input, pack::splat(limit::quadrant), pack::splat(limit::round));
            const value quarters = pack::subtract(shifted, pack::splat(limit::round));

            // r = x - j * pi/2, with pi/2 in four parts.
            const value negative = pack::negate(quarters);
            value reduced = pack::fuse(negative, pack::splat(limit::quarter[0]), input);
            reduced = pack::fuse(negative, pack::splat(limit::quarter[1]), reduced);
            reduced = pack::fuse(negative, pack::splat(limit::quarter[2]), reduced);
            reduced = pack::fuse(negative, pack::splat(limit::quarter[3]), reduced);

            const value square = pack::multiply(reduced, reduced);
            const value sine = pack::fuse(pack::multiply(reduced, square), odd.template evaluate<pack, 1>(square), reduced);
            const value cosine = pack::fuse(square, even.template evaluate<pack, 1>(square), pack::splat(type{1}));

            auto quadrant = pack::raw(shifted);
            if constexpr (phase) {
                quadrant = pack::toggle(quadrant, pack::template shift<1>(quadrant));
            }
            const auto flip = pack::template shift<limit::width - 2>(pack::intersect(quadrant, pack::fill(2)));
            const auto swap = pack::test(pack::intersect(pack::raw(shifted), pack::fill(1)));

            value result = phase ? pack::select(swap, sine, cosine) : pack::select(swap, cosine, sine);
            result = pack::cast(pack::toggle(pack::raw(result), flip));

            // Arguments beyond the Cody-Waite range (and inf) take the C library path, lane by lane.
            if (pack::any(pack::greater(pack::absolute(input), pack::splat(limit::reach)))) [[unlikely]] {
                type inputs[pack::width];
                type outputs[pack::width];
                pack::store(inputs, input);
                pack::store(outputs, result);
                for (size lane = 0; lane < pack::width; ++lane) {
                    if (!(inputs[lane] <= limit::reach && inputs[lane] >= -limit::reach)) {
                        if constexpr (sizeof(type) == 4) {
                            outputs[lane] = phase ? ::cosf(inputs[lane]) : ::sinf(inputs[lane]);
                        } else {
                            outputs[lane] = phase ? ::cos(inputs[lane]) : ::sin(inputs[lane]);
                        }
                    }
                }
                result = pack::load(outputs);
            }
            return result;
        }

    private:
        // Taylor series of sine (odd powers) and cosine (even powers), in powers of r^2.
        static constexpr auto odd = Polynomial<type, limit::sine + 1>::template alternating<1>();
        static constexpr auto even = Polynomial<type, limit::cosine + 1>::template alternating<0>();
    };

}
