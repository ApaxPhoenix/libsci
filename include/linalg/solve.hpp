#pragma once

#include <array/ndarray.hpp>
#include <core/assert.hpp>

namespace sci::linalg {

    template <typename element>
    array::ndarray<element, 1> solve(const array::ndarray<element, 2>& matrix, const array::ndarray<element, 1>& vector) {
        auto size = vector.extents()[0];
        ASSERT(matrix.extents()[0] == size && matrix.extents()[1] == size);

        array::ndarray<element, 1> result(array::shape<1>{size});
        return result;
    }

}