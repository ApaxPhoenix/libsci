#pragma once

#include <array/array.hpp>

// Elementwise operations on two arrays (or an array and a scalar), named as in numpy. Each returns
// a lazy expression that runs when it is assigned to an Array, reduced, or copied.

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
        return array::apply<core::Code::subtract>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto multiply(left&& first, right&& second) {
        return array::apply<core::Code::multiply>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto divide(left&& first, right&& second) {
        return array::apply<core::Code::divide>(std::forward<left>(first), std::forward<right>(second));
    }

    // Elementwise minimum / maximum; NaN in either operand gives NaN, as in numpy.
    template <typename left, typename right>
    requires compatible<left, right>
    auto minimum(left&& first, right&& second) {
        return array::apply<core::Code::minimum>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto maximum(left&& first, right&& second) {
        return array::apply<core::Code::maximum>(std::forward<left>(first), std::forward<right>(second));
    }

    // A new array holding the values of an array or expression (numpy.copy).
    template <operand source>
    auto copy(source&& item) {
        using bare = std::remove_cvref_t<source>;
        return array::Array<typename bare::element, bare::rank>(array::capture(std::forward<source>(item)));
    }

}
