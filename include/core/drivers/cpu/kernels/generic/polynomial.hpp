#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    // Coefficients of a polynomial: coefficients[k] multiplies x^k.
    template <typename type, size count>
    struct Polynomial {
        type coefficients[count]{};
    };

    // Taylor coefficients 1/k!.
    template <typename type, size count>
    constexpr Polynomial<type, count> factorials() noexcept {
        Polynomial<type, count> polynomial{};
        double factor = 1.0;
        for (size index = 0; index < count; ++index) {
            if (index > 0) factor /= static_cast<double>(index);
            polynomial.coefficients[index] = static_cast<type>(factor);
        }
        return polynomial;
    }

    // atanh series tail: log((1 + s) / (1 - s)) = 2s + s * R(z) with z = s^2 and
    // R(z) = 2z/3 + 2z^2/5 + ... = z * sum 2 z^k / (2k + 3).
    template <typename type, size count>
    constexpr Polynomial<type, count> reciprocals() noexcept {
        Polynomial<type, count> polynomial{};
        for (size index = 0; index < count; ++index) {
            polynomial.coefficients[index] = static_cast<type>(2.0 / static_cast<double>(2 * index + 3));
        }
        return polynomial;
    }

    // Taylor coefficients (-1)^k / (2k + offset)!, in powers of r^2.
    template <typename type, size count, size offset>
    constexpr Polynomial<type, count> alternating() noexcept {
        Polynomial<type, count> polynomial{};
        for (size index = 0; index < count; ++index) {
            double factor = 1.0;
            for (size step = 2; step <= 2 * index + offset; ++step) {
                factor *= static_cast<double>(step);
            }
            polynomial.coefficients[index] = static_cast<type>((index % 2 == 0 ? 1.0 : -1.0) / factor);
        }
        return polynomial;
    }

    // Evaluates sum c[k] x^(k - from) for k in [from, count).
    template <typename pack, size from, typename type, size count>
    pack::value horner(const typename pack::value x, const Polynomial<type, count>& polynomial) noexcept {
        typename pack::value result = pack::splat(polynomial.coefficients[count - 1]);
        for (size index = count - 1; index-- > from;) {
            result = pack::fma(result, x, pack::splat(polynomial.coefficients[index]));
        }
        return result;
    }

}
