#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    // Coefficients of a polynomial: coefficients[k] multiplies x^k.
    template <typename type, size count>
    struct Polynomial {
        type coefficients[count]{};

        // Taylor coefficients 1/k!.
        static constexpr Polynomial factorials() noexcept {
            Polynomial polynomial{};
            double factor = 1.0;
            for (size index = 0; index < count; ++index) {
                if (index > 0) factor /= static_cast<double>(index);
                polynomial.coefficients[index] = static_cast<type>(factor);
            }
            return polynomial;
        }

        // atanh series tail: log((1 + s) / (1 - s)) = 2s + s * R(z) with z = s^2 and
        // R(z) = 2z/3 + 2z^2/5 + ... = z * sum 2 z^k / (2k + 3).
        static constexpr Polynomial reciprocals() noexcept {
            Polynomial polynomial{};
            for (size index = 0; index < count; ++index) {
                polynomial.coefficients[index] = static_cast<type>(2.0 / static_cast<double>(2 * index + 3));
            }
            return polynomial;
        }

        // Taylor coefficients (-1)^k / (2k + offset)!, in powers of r^2.
        template <size offset>
        static constexpr Polynomial alternating() noexcept {
            Polynomial polynomial{};
            for (size index = 0; index < count; ++index) {
                double factor = 1.0;
                for (size step = 2; step <= 2 * index + offset; ++step) {
                    factor *= static_cast<double>(step);
                }
                polynomial.coefficients[index] = static_cast<type>((index % 2 == 0 ? 1.0 : -1.0) / factor);
            }
            return polynomial;
        }

        // sum c[k] x^(k - from) for k in [from, count), by Horner's rule.
        template <typename pack, size from>
        pack::value evaluate(const pack::value input) const noexcept {
            typename pack::value result = pack::splat(coefficients[count - 1]);
            for (size index = count - 1; index-- > from;) {
                result = pack::fuse(result, input, pack::splat(coefficients[index]));
            }
            return result;
        }
    };

}
