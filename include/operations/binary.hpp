#pragma once

#include <array/ndarray.hpp>

// Elementwise operations on two arrays (or an array and a scalar), named as in numpy. Each returns
// a lazy expression that runs when it is assigned to an NDArray, reduced, or copied.

namespace sci::operations {

    using array::compatible;
    using array::operand;

    template <typename left, typename right>
    requires compatible<left, right>
    auto add(left&& first, right&& second) {
        return array::apply<core::Code::add>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto subtract(left&& first, right&& second) {
        return array::apply<core::Code::sub>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto multiply(left&& first, right&& second) {
        return array::apply<core::Code::mul>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto divide(left&& first, right&& second) {
        return array::apply<core::Code::div>(std::forward<left>(first), std::forward<right>(second));
    }

    // Elementwise minimum / maximum; NaN in either operand gives NaN, as in numpy.
    template <typename left, typename right>
    requires compatible<left, right>
    auto minimum(left&& first, right&& second) {
        return array::apply<core::Code::min>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto maximum(left&& first, right&& second) {
        return array::apply<core::Code::max>(std::forward<left>(first), std::forward<right>(second));
    }

    // A new array holding the values of an array or expression (numpy.copy).
    template <operand source>
    auto copy(source&& item) {
        return array::NDArray<array::element_t<source>, array::rank_v<source>>(array::wrap(std::forward<source>(item)));
    }

}
