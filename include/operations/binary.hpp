#pragma once

#include <array/array.hpp>

// Elementwise operations on two arrays (or an array and a scalar), named as in numpy. Each returns
// a lazy expression that runs when it is assigned to an Array, reduced, or copied.

namespace sci::operations {

    using array::compatible;
    using array::operand;

    template <typename alpha, typename beta>
        requires compatible<alpha, beta>
    auto add(alpha &&first, beta &&second) {
        return array::apply<core::Code::add>(std::forward<alpha>(first), std::forward<beta>(second));
    }

    template <typename alpha, typename beta>
        requires compatible<alpha, beta>
    auto subtract(alpha &&first, beta &&second) {
        return array::apply<core::Code::subtract>(std::forward<alpha>(first), std::forward<beta>(second));
    }

    template <typename alpha, typename beta>
        requires compatible<alpha, beta>
    auto multiply(alpha &&first, beta &&second) {
        return array::apply<core::Code::multiply>(std::forward<alpha>(first), std::forward<beta>(second));
    }

    template <typename alpha, typename beta>
        requires compatible<alpha, beta>
    auto divide(alpha &&first, beta &&second) {
        return array::apply<core::Code::divide>(std::forward<alpha>(first), std::forward<beta>(second));
    }

    // Elementwise minimum / maximum; NaN in either operand gives NaN, as in numpy.
    template <typename alpha, typename beta>
        requires compatible<alpha, beta>
    auto minimum(alpha &&first, beta &&second) {
        return array::apply<core::Code::minimum>(std::forward<alpha>(first), std::forward<beta>(second));
    }

    template <typename alpha, typename beta>
        requires compatible<alpha, beta>
    auto maximum(alpha &&first, beta &&second) {
        return array::apply<core::Code::maximum>(std::forward<alpha>(first), std::forward<beta>(second));
    }

    // A new array holding the values of an array or expression (numpy.copy).
    template <operand source> auto copy(source &&item) {
        using bare = std::remove_cvref_t<source>;
        return array::Array<typename bare::element, bare::rank>(array::capture(std::forward<source>(item)));
    }

}
