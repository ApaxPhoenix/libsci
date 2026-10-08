#pragma once

#include <array/ndarray.hpp>

// Elementwise functions of one array, named as in numpy and lazy like the binary ones. On float and
// double arrays, exp and log are within 1 ulp and sin and cos within 2 ulp (3 on the GPU).

namespace sci::operations {

    using array::operand;

    template <operand source>
    auto negative(source&& input) {
        return array::apply<core::Code::neg>(std::forward<source>(input));
    }

    template <operand source>
    auto abs(source&& input) {
        return array::apply<core::Code::abs>(std::forward<source>(input));
    }

    template <operand source>
    auto sqrt(source&& input) {
        return array::apply<core::Code::sqrt>(std::forward<source>(input));
    }

    template <operand source>
    auto exp(source&& input) {
        return array::apply<core::Code::exp>(std::forward<source>(input));
    }

    template <operand source>
    auto log(source&& input) {
        return array::apply<core::Code::log>(std::forward<source>(input));
    }

    template <operand source>
    auto sin(source&& input) {
        return array::apply<core::Code::sin>(std::forward<source>(input));
    }

    template <operand source>
    auto cos(source&& input) {
        return array::apply<core::Code::cos>(std::forward<source>(input));
    }

    template <operand source>
    auto tan(source&& input) {
        return array::apply<core::Code::tan>(std::forward<source>(input));
    }

    template <operand source>
    auto asin(source&& input) {
        return array::apply<core::Code::asin>(std::forward<source>(input));
    }

    template <operand source>
    auto acos(source&& input) {
        return array::apply<core::Code::acos>(std::forward<source>(input));
    }

    template <operand source>
    auto atan(source&& input) {
        return array::apply<core::Code::atan>(std::forward<source>(input));
    }

    template <operand source>
    auto sinh(source&& input) {
        return array::apply<core::Code::sinh>(std::forward<source>(input));
    }

    template <operand source>
    auto cosh(source&& input) {
        return array::apply<core::Code::cosh>(std::forward<source>(input));
    }

    template <operand source>
    auto tanh(source&& input) {
        return array::apply<core::Code::tanh>(std::forward<source>(input));
    }

    template <operand source>
    auto asinh(source&& input) {
        return array::apply<core::Code::asinh>(std::forward<source>(input));
    }

    template <operand source>
    auto acosh(source&& input) {
        return array::apply<core::Code::acosh>(std::forward<source>(input));
    }

    template <operand source>
    auto atanh(source&& input) {
        return array::apply<core::Code::atanh>(std::forward<source>(input));
    }

}
