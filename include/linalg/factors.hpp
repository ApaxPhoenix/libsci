#pragma once

#include "array/ndarray.hpp"

namespace sci::linalg {

    using array::NDArray;

    // A = permutation * lower * upper, as in scipy.linalg.lu: lower has a unit diagonal, upper is
    // upper triangular, permutation records the row exchanges of partial pivoting.
    //
    //     auto [p, l, u] = sci::linalg::lu(a);
    template <typename element>
    struct Factors {
        NDArray<element, 2> permutation;
        NDArray<element, 2> lower;
        NDArray<element, 2> upper;
    };

}
