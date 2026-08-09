#pragma once

#include <array/ndarray.hpp>
#include <core/assert.hpp>

namespace sci::linalg {

    template <typename element>
    array::ndarray<element, 2> matmul(const array::ndarray<element, 2>& left, const array::ndarray<element, 2>& right) {
        auto rows = left.extents()[0];
        auto inner = left.extents()[1];
        auto cols = right.extents()[1];

        ASSERT(inner == right.extents()[0]);

        array::ndarray<element, 2> output(array::shape<2>{rows, cols});

        const element* lhs = left.data();
        const element* rhs = right.data();
        element* dst = output.data();

        for (core::size index = 0; index < output.size(); ++index) {
            dst[index] = 0;
        }

        for (core::size i = 0; i < rows; ++i) {
            for (core::size k = 0; k < inner; ++k) {
                for (core::size j = 0; j < cols; ++j) {
                    dst[i * cols + j] += lhs[i * inner + k] * rhs[k * cols + j];
                }
            }
        }
        return output;
    }

}