#pragma once

#include "core/types.hpp"

// The C functions (::sqrtf, ::sqrt), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::processor::kernels {

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
        static value subtract(const value left, const value right) noexcept { return left - right; }
        static value multiply(const value left, const value right) noexcept { return left * right; }
        static value divide(const value left, const value right) noexcept { return left / right; }

        // Without a fused multiply-add instruction a float fma is still exact enough through double (the product
        // is exact there), which the range reduction of sine and cosine relies on. Doubles fall back to two roundings.
        static value fuse(const value left, const value right, const value addend) noexcept {
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

        static value negate(const value item) noexcept { return -item; }
        static value absolute(const value item) noexcept { return cast(intersect(raw(item), ~sign)); }
        static value root(const value item) noexcept {
            if constexpr (sizeof(type) == 4) {
                return ::sqrtf(item);
            } else {
                return ::sqrt(item);
            }
        }

        static value minimum(const value left, const value right) noexcept {
            if (__builtin_isnan(left) || __builtin_isnan(right)) return left + right;
            return right < left ? right : left;
        }

        static value maximum(const value left, const value right) noexcept {
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

        static bits raw(const value item) noexcept { return __builtin_bit_cast(integer, item); }
        static value cast(const bits item) noexcept { return __builtin_bit_cast(type, item); }

        static bits intersect(const bits left, const bits right) noexcept { return left & right; }
        static bits unite(const bits left, const bits right) noexcept { return left | right; }
        static bits toggle(const bits left, const bits right) noexcept { return left ^ right; }

        // Every lane shifted left by `count` bits, or right by -count when it is negative.
        template <int count>
        static bits shift(const bits item) noexcept {
            if constexpr (count > 0) {
                return item << count;
            } else {
                return item >> -count;
            }
        }

    private:
        static constexpr integer sign = integer{1} << (sizeof(integer) * 8 - 1);
    };

}
