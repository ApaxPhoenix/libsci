#pragma once

#include "array/array.hpp"

namespace sci::linear {

    using array::Array;

    // A = permutation * lower * upper, as in scipy.linalg.lu: lower has a unit diagonal, upper is
    // upper triangular, permutation records the row exchanges of partial pivoting.
    //
    //     auto [permutation, lower, upper] = sci::linear::decompose(matrix);
    template <typename element>
    struct Factors {
        Array<element, 2> permutation;
        Array<element, 2> lower;
        Array<element, 2> upper;
    };

}
