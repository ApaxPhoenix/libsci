#pragma once

#include <array/array.hpp>

// Elementwise functions of one array, lazy like the binary ones. On float and double arrays the
// exponential and logarithm are within 1 ulp, and sine and cosine within 2 ulp (3 on the GPU).

namespace sci::operations {

    using array::operand;

    template <operand source>
    auto negate(source&& input) {
        return array::apply<core::Code::negate>(std::forward<source>(input));
    }

    template <operand source>
    auto absolute(source&& input) {
        return array::apply<core::Code::absolute>(std::forward<source>(input));
    }

    template <operand source>
    auto root(source&& input) {
        return array::apply<core::Code::root>(std::forward<source>(input));
    }

    template <operand source>
    auto exponential(source&& input) {
        return array::apply<core::Code::exponential>(std::forward<source>(input));
    }

    template <operand source>
    auto logarithm(source&& input) {
        return array::apply<core::Code::logarithm>(std::forward<source>(input));
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
    auto csc(source&& input) {
        return array::apply<core::Code::csc>(std::forward<source>(input));
    }

    template <operand source>
    auto sec(source&& input) {
        return array::apply<core::Code::sec>(std::forward<source>(input));
    }

    template <operand source>
    auto cot(source&& input) {
        return array::apply<core::Code::cot>(std::forward<source>(input));
    }

    template <operand source>
    auto acsc(source&& input) {
        return array::apply<core::Code::acsc>(std::forward<source>(input));
    }

    template <operand source>
    auto asec(source&& input) {
        return array::apply<core::Code::asec>(std::forward<source>(input));
    }

    template <operand source>
    auto acot(source&& input) {
        return array::apply<core::Code::acot>(std::forward<source>(input));
    }

    // sinh, cosh, tanh and their inverses, with the h suffix from C.
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

    template <operand source>
    auto csch(source&& input) {
        return array::apply<core::Code::csch>(std::forward<source>(input));
    }

    template <operand source>
    auto sech(source&& input) {
        return array::apply<core::Code::sech>(std::forward<source>(input));
    }

    template <operand source>
    auto coth(source&& input) {
        return array::apply<core::Code::coth>(std::forward<source>(input));
    }

    template <operand source>
    auto acsch(source&& input) {
        return array::apply<core::Code::acsch>(std::forward<source>(input));
    }

    template <operand source>
    auto asech(source&& input) {
        return array::apply<core::Code::asech>(std::forward<source>(input));
    }

    template <operand source>
    auto acoth(source&& input) {
        return array::apply<core::Code::acoth>(std::forward<source>(input));
    }

}