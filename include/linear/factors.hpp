#pragma once

/// @file
/// @brief A = permutation * lower * upper, as in scipy.linalg.lu.

#include "array/array.hpp"

namespace sci::linear {

    using array::Array;

    /// A = permutation * lower * upper, as in scipy.linalg.lu: lower has a unit diagonal, upper is
    /// upper triangular, permutation records the row exchanges of partial pivoting.
    ///
    ///     auto [permutation, lower, upper] = sci::linear::decompose(matrix);
    template <typename element> struct Factors {
        /// @brief The row permutation.
        Array<element, 2> permutation;
        /// @brief The lower triangle, with a unit diagonal.
        Array<element, 2> lower;
        /// @brief The upper triangle.
        Array<element, 2> upper;
    };

}
