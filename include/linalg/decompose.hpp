#pragma once

#include <utility>
#include <array/ndarray.hpp>
#include <core/assert.hpp>

namespace sci::linalg {

    template <typename element>
    std::pair<array::ndarray<element, 2>, array::ndarray<element, 2>> lu(const array::ndarray<element, 2>& matrix) {
        ASSERT(matrix.extents()[0] == matrix.extents()[1]);

        array::ndarray<element, 2> lower(matrix.extents());
        array::ndarray<element, 2> upper(matrix.extents());

        return {lower, upper};
    }

}