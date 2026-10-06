#pragma once

#include "core/drivers/cpu/kernels/generic/constants.hpp"
#include "core/drivers/cpu/kernels/generic/polynomial.hpp"
#include "core/types.hpp"

// The C functions (::sinf, ::cos, ...), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::cpu::kernels {

    // Sine and cosine over a whole pack.
    template <typename pack>
    struct Trigonometry {
        using type = pack::element;
        using value = pack::value;
        using limit = Constants<type>;

        static value sin(const value x) noexcept {
            return trig<false>(x);
        }

        static value cos(const value x) noexcept {
            return trig<true>(x);
        }

    private:
        static constexpr auto sinseries = alternating<type, limit::sindegree + 1, 1>();
        static constexpr auto cosseries = alternating<type, limit::cosdegree + 1, 0>();

        template <bool cosine>
        static value trig(const value x) noexcept {
            const value t = pack::fma(x, pack::splat(limit::twoopi), pack::splat(limit::round));
            const value j = pack::sub(t, pack::splat(limit::round));
            value r = pack::fnma(j, pack::splat(limit::pio2a), x);
            r = pack::fnma(j, pack::splat(limit::pio2b), r);
            r = pack::fnma(j, pack::splat(limit::pio2c), r);
            r = pack::fnma(j, pack::splat(limit::pio2d), r);

            const value r2 = pack::mul(r, r);
            const value odd = pack::fma(pack::mul(r, r2), horner<pack, 1>(r2, sinseries), r);
            const value even = pack::fma(r2, horner<pack, 1>(r2, cosseries), pack::splat(type{1}));

            // The low bits of t hold the quadrant j mod 4.
            auto quadrant = pack::tobits(t);
            if constexpr (cosine) {
                quadrant = pack::bxor(quadrant, pack::template shl<1>(quadrant));
            }
            const auto flip = pack::template shl<limit::width - 2>(pack::band(quadrant, pack::fill(2)));
            const auto swap = pack::test(pack::band(pack::tobits(t), pack::fill(1)));

            value result = cosine ? pack::select(swap, odd, even) : pack::select(swap, even, odd);
            result = pack::cast(pack::bxor(pack::tobits(result), flip));

            // Arguments beyond the Cody-Waite range (and inf) take the libm path, lane by lane.
            if (pack::any(pack::greater(pack::abs(x), pack::splat(limit::reach)))) [[unlikely]] {
                type inputs[pack::width];
                type outputs[pack::width];
                pack::store(inputs, x);
                pack::store(outputs, result);
                for (size lane = 0; lane < pack::width; ++lane) {
                    if (!(inputs[lane] <= limit::reach && inputs[lane] >= -limit::reach)) {
                        outputs[lane] = libm<cosine>(inputs[lane]);
                    }
                }
                result = pack::load(outputs);
            }
            return result;
        }

        template <bool cosine>
        static type libm(const type x) noexcept {
            if constexpr (sizeof(type) == 4) {
                return cosine ? ::cosf(x) : ::sinf(x);
            } else {
                return cosine ? ::cos(x) : ::sin(x);
            }
        }
    };

}
