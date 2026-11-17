#pragma once

/// @file
/// @brief Reductions: sum, mean, minimum and maximum, fused with what they reduce.

#include <array/array.hpp>

// Whole-array reductions, named as in numpy (numpy.min and numpy.max become minimum and maximum
// with one operand). Reducing an expression fuses it into the reduction, so sum(a * b) never
// materialises a * b.

namespace sci::operations {

    using array::operand;

    /// @brief The sum of every element (numpy.sum).
    /// @param input The array or expression.
    /// @return The sum.
    template <operand source> source::element sum(const source &input) { return array::reduce(input, core::Fold::sum); }

    /// @brief The mean of every element (numpy.mean).
    /// @param input The array or expression.
    /// @return The mean.
    template <operand source> source::element mean(const source &input) {
        using element = source::element;
        const core::size count = input.shape().total();
        return count == 0 ? element{} : sum(input) / static_cast<element>(count);
    }

    /// NaN anywhere gives NaN, as in numpy.min / numpy.max.
    /// @param input The array or expression.
    /// @return The minimum.
    template <operand source> source::element minimum(const source &input) {
        return array::reduce(input, core::Fold::minimum);
    }

    /// @brief The greatest element (numpy.max); NaN anywhere gives NaN.
    /// @param input The array or expression.
    /// @return The maximum.
    template <operand source> source::element maximum(const source &input) {
        return array::reduce(input, core::Fold::maximum);
    }

}
