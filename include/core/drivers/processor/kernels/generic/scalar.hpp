#pragma once

/// @file
/// @brief One-lane pack over plain scalars.

#include "core/types.hpp"

// The C functions (::sqrtf, ::sqrt), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::processor::kernels {

    /// One-lane pack over plain scalars: the fallback ISA and every loop tail. `tag` is the owning
    /// ISA's pack (Neon<float>, ...), which keeps each file's instantiations apart.
    template <typename type, typename integer, typename tag> struct Scalar {
        /// @brief The element type.
        using element = type;
        /// @brief A one-lane pack: the element.
        using value = type;
        /// @brief The element's bits, as an integer.
        using bits = integer;
        /// @brief A one-lane comparison.
        using mask = bool;
        /// @brief The pack of the narrower type, itself.
        using narrow = Scalar;

        /// @brief One lane.
        static constexpr size width = 1;

        /// @brief Reads one element.
        /// @param source Where.
        /// @return The element.
        static value load(const type *source) noexcept { return *source; }
        /// @brief Writes one element.
        /// @param target Where.
        /// @param item The element.
        static void store(type *target, const value item) noexcept { *target = item; }
        /// @brief A pack of one value in every lane.
        /// @param item The value.
        /// @return The pack.
        static value splat(const type item) noexcept { return item; }
        /// @brief Integer bits in every lane.
        /// @param item The bits.
        /// @return The bits.
        static bits fill(const integer item) noexcept { return item; }

        /// @brief The sum of two packs.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The sum.
        static value add(const value left, const value right) noexcept { return left + right; }
        /// @brief The difference of two packs.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The difference.
        static value subtract(const value left, const value right) noexcept { return left - right; }
        /// @brief The product of two packs.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The product.
        static value multiply(const value left, const value right) noexcept { return left * right; }
        /// @brief The quotient of two packs.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The quotient.
        static value divide(const value left, const value right) noexcept { return left / right; }

        /// Without a fused multiply-add instruction a float fma is still exact enough through double (the product
        /// is exact there), which the range reduction of sine and cosine relies on. Doubles fall back to two roundings.
        /// @param left The first factor.
        /// @param right The second factor.
        /// @param addend What is added.
        /// @return left * right + addend, rounded once.
        static value fuse(const value left, const value right, const value addend) noexcept {
#if defined(__FMA__) || defined(__aarch64__)
            if constexpr (sizeof(type) == 4) {
                return __builtin_fmaf(left, right, addend);
            } else {
                return __builtin_fma(left, right, addend);
            }
#else
            if constexpr (sizeof(type) == 4) {
                return static_cast<float>(static_cast<double>(left) * static_cast<double>(right) +
                                          static_cast<double>(addend));
            } else {
                return left * right + addend;
            }
#endif
        }

        /// @brief The negation.
        /// @param item The operand.
        /// @return -item.
        static value negate(const value item) noexcept { return -item; }
        /// @brief The absolute value.
        /// @param item The operand.
        /// @return |item|.
        static value absolute(const value item) noexcept { return cast(intersect(raw(item), ~sign)); }
        /// @brief The square root.
        /// @param item The operand.
        /// @return The root.
        static value root(const value item) noexcept {
            if constexpr (sizeof(type) == 4) {
                return ::sqrtf(item);
            } else {
                return ::sqrt(item);
            }
        }

        /// @brief The lesser of two; NaN in either gives NaN.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The minimum.
        static value minimum(const value left, const value right) noexcept {
            if (__builtin_isnan(left) || __builtin_isnan(right)) return left + right;
            return right < left ? right : left;
        }

        /// @brief The greater of two; NaN in either gives NaN.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The maximum.
        static value maximum(const value left, const value right) noexcept {
            if (__builtin_isnan(left) || __builtin_isnan(right)) return left + right;
            return right > left ? right : left;
        }

        /// @brief Whether left is less than right.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The mask.
        static mask less(const value left, const value right) noexcept { return left < right; }
        /// @brief Whether left is greater than right.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The mask.
        static mask greater(const value left, const value right) noexcept { return left > right; }
        /// @brief Whether they are equal.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return The mask.
        static mask equal(const value left, const value right) noexcept { return left == right; }
        /// @brief Whether it is NaN.
        /// @param item The operand.
        /// @return The mask.
        static mask unordered(const value item) noexcept { return __builtin_isnan(item); }
        /// @brief Whether any bit is set.
        /// @param item The operand.
        /// @return The mask.
        static mask test(const bits item) noexcept { return item != 0; }
        /// @brief Whether any lane is set.
        /// @param item The operand.
        /// @return Whether it is.
        static bool any(const mask item) noexcept { return item; }

        /// @brief The left value where the mask is set, the right where it is not.
        /// @param choice Which lanes take the left value.
        /// @param left Taken where it is set.
        /// @param right Taken where it is not.
        /// @return The pack.
        static value select(const mask choice, const value left, const value right) noexcept {
            return choice ? left : right;
        }

        /// @brief The element's bits.
        /// @param item The operand.
        /// @return The bits.
        static bits raw(const value item) noexcept { return __builtin_bit_cast(integer, item); }
        /// @brief Bits as an element.
        /// @param item The operand.
        /// @return The element.
        static value cast(const bits item) noexcept { return __builtin_bit_cast(type, item); }

        /// @brief The bits set in both.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return left & right.
        static bits intersect(const bits left, const bits right) noexcept { return left & right; }
        /// @brief The bits set in either.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return left | right.
        static bits unite(const bits left, const bits right) noexcept { return left | right; }
        /// @brief The bits set in one of them only.
        /// @param left The left operand.
        /// @param right The right operand.
        /// @return left ^ right.
        static bits toggle(const bits left, const bits right) noexcept { return left ^ right; }

        /// Every lane shifted left by `count` bits, or right by -count when it is negative.
        /// @param item The bits.
        /// @return The shifted bits.
        template <int count> static bits shift(const bits item) noexcept {
            if constexpr (count > 0) {
                return item << count;
            } else {
                return item >> -count;
            }
        }

    private:
        /// @brief The sign bit.
        static constexpr integer sign = integer{1} << (sizeof(integer) * 8 - 1);
    };

}
