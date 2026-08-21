#pragma once

#include <algorithm>
#include <array/ndarray.hpp>

namespace sci::ops {

    template <typename element, core::size rank>
    element sum(const array::ndarray<element, rank>& input) {
        element total = 0;
        const element* src = input.data();
        for (core::size index = 0; index < input.size(); ++index) {
            total += src[index];
        }
        return total;
    }

    template <typename element, core::size rank>
    element mean(const array::ndarray<element, rank>& input) {
        return sum(input) / static_cast<element>(input.size());
    }

    template <typename element, core::size rank>
    element min(const array::ndarray<element, rank>& input) {
        const element* src = input.data();
        element value = src[0];
        for (core::size index = 1; index < input.size(); ++index) {
            value = std::min(value, src[index]);
        }
        return value;
    }

    template <typename element, core::size rank>
    element max(const array::ndarray<element, rank>& input) {
        const element* src = input.data();
        element value = src[0];
        for (core::size index = 1; index < input.size(); ++index) {
            value = std::max(value, src[index]);
        }
        return value;
    }

}