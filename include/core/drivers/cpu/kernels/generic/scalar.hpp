#pragma once

#include "core/types.hpp"

// The C functions (::sqrtf, ::sqrt), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::cpu::kernels {

    // One-lane pack over plain scalars: the fallback ISA and every loop tail. `tag` is the owning
    // ISA's pack (Neon<float>, ...), which keeps each file's instantiations apart.
    template <typename type, typename integer, typename tag>
    struct Scalar {
        using element = type;
        using value = type;
        using bits = integer;
        using mask = bool;
        using narrow = Scalar;

        static constexpr size width = 1;

        static value load(const type* source) noexcept { return *source; }
        static void store(type* target, const value item) noexcept { *target = item; }
        static value splat(const type item) noexcept { return item; }
        static bits fill(const integer item) noexcept { return item; }

        static value add(const value left, const value right) noexcept { return left + right; }
        static value sub(const value left, const value right) noexcept { return left - right; }
        static value mul(const value left, const value right) noexcept { return left * right; }
        static value div(const value left, const value right) noexcept { return left / right; }

        // Without a fused multiply-add instruction a float fma is still exact enough through double (the product
        // is exact there), which the trig range reduction relies on. Doubles fall back to two roundings.
        static value fma(const value left, const value right, const value addend) noexcept {
            #if defined(__FMA__) || defined(__aarch64__)
                if constexpr (sizeof(type) == 4) {
                    return __builtin_fmaf(left, right, addend);
                } else {
                    return __builtin_fma(left, right, addend);
                }
            #else
                if constexpr (sizeof(type) == 4) {
                    return static_cast<float>(static_cast<double>(left) * static_cast<double>(right) + static_cast<double>(addend));
                } else {
                    return left * right + addend;
                }
            #endif
        }

        static value fnma(const value left, const value right, const value addend) noexcept {
            return fma(-left, right, addend);
        }

        static value neg(const value item) noexcept { return -item; }
        static value abs(const value item) noexcept { return cast(band(tobits(item), ~sign)); }
        static value sqrt(const value item) noexcept {
            if constexpr (sizeof(type) == 4) {
                return ::sqrtf(item);
            } else {
                return ::sqrt(item);
            }
        }

        static value min(const value left, const value right) noexcept {
            if (__builtin_isnan(left) || __builtin_isnan(right)) return left + right;
            return right < left ? right : left;
        }

        static value max(const value left, const value right) noexcept {
            if (__builtin_isnan(left) || __builtin_isnan(right)) return left + right;
            return right > left ? right : left;
        }

        static mask less(const value left, const value right) noexcept { return left < right; }
        static mask greater(const value left, const value right) noexcept { return left > right; }
        static mask equal(const value left, const value right) noexcept { return left == right; }
        static mask unordered(const value item) noexcept { return __builtin_isnan(item); }
        static mask test(const bits item) noexcept { return item != 0; }
        static bool any(const mask item) noexcept { return item; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return choice ? left : right;
        }

        static bits tobits(const value item) noexcept { return __builtin_bit_cast(integer, item); }
        static value cast(const bits item) noexcept { return __builtin_bit_cast(type, item); }

        static bits band(const bits left, const bits right) noexcept { return left & right; }
        static bits bor(const bits left, const bits right) noexcept { return left | right; }
        static bits bxor(const bits left, const bits right) noexcept { return left ^ right; }

        template <int count>
        static bits shl(const bits item) noexcept { return item << count; }

        template <int count>
        static bits shr(const bits item) noexcept { return item >> count; }

    private:
        static constexpr integer sign = integer{1} << (sizeof(integer) * 8 - 1);
    };

}
