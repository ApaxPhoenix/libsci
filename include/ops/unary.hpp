#pragma once

#include <cmath>
#include <array/ndarray.hpp>

namespace sci::ops {

    template <typename element, core::size rank>
    array::ndarray<element, rank> abs(const array::ndarray<element, rank>& input) {
        array::ndarray<element, rank> output(input.extents());
        const element* source = input.data();
        element* destination = output.data();
        for (core::size index = 0; index < input.size(); ++index) {
            destination[index] = std::abs(source[index]);
        }
        return output;
    }

    template <typename element, core::size rank>
    array::ndarray<element, rank> exp(const array::ndarray<element, rank>& input) {
        array::ndarray<element, rank> output(input.extents());
        const element* source = input.data();
        element* destination = output.data();
        for (core::size index = 0; index < input.size(); ++index) {
            destination[index] = std::exp(source[index]);
        }
        return output;
    }

    template <typename element, core::size rank>
    array::ndarray<element, rank> log(const array::ndarray<element, rank>& input) {
        array::ndarray<element, rank> output(input.extents());
        const element* source = input.data();
        element* destination = output.data();
        for (core::size index = 0; index < input.size(); ++index) {
            destination[index] = std::log(source[index]);
        }
        return output;
    }

    template <typename element, core::size rank>
    array::ndarray<element, rank> sin(const array::ndarray<element, rank>& input) {
        array::ndarray<element, rank> output(input.extents());
        const element* source = input.data();
        element* destination = output.data();
        for (core::size index = 0; index < input.size(); ++index) {
            destination[index] = std::sin(source[index]);
        }
        return output;
    }

}