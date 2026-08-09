#pragma once

#include <array/ndarray.hpp>

namespace sci::ops {

    template <typename element, core::size rank>
    array::ndarray<element, rank> add(const array::ndarray<element, rank>& left, const array::ndarray<element, rank>& right) {
        array::ndarray<element, rank> output(left.extents());
        const element* lhs = left.data();
        const element* rhs = right.data();
        element* dst = output.data();
        for (core::size index = 0; index < left.size(); ++index) {
            dst[index] = lhs[index] + rhs[index];
        }
        return output;
    }

    template <typename element, core::size rank>
    array::ndarray<element, rank> sub(const array::ndarray<element, rank>& left, const array::ndarray<element, rank>& right) {
        array::ndarray<element, rank> output(left.extents());
        const element* lhs = left.data();
        const element* rhs = right.data();
        element* dst = output.data();
        for (core::size index = 0; index < left.size(); ++index) {
            dst[index] = lhs[index] - rhs[index];
        }
        return output;
    }

    template <typename element, core::size rank>
    array::ndarray<element, rank> mul(const array::ndarray<element, rank>& left, const array::ndarray<element, rank>& right) {
        array::ndarray<element, rank> output(left.extents());
        const element* lhs = left.data();
        const element* rhs = right.data();
        element* dst = output.data();
        for (core::size index = 0; index < left.size(); ++index) {
            dst[index] = lhs[index] * rhs[index];
        }
        return output;
    }

    template <typename element, core::size rank>
    array::ndarray<element, rank> div(const array::ndarray<element, rank>& left, const array::ndarray<element, rank>& right) {
        array::ndarray<element, rank> output(left.extents());
        const element* lhs = left.data();
        const element* rhs = right.data();
        element* dst = output.data();
        for (core::size index = 0; index < left.size(); ++index) {
            dst[index] = lhs[index] / rhs[index];
        }
        return output;
    }

}