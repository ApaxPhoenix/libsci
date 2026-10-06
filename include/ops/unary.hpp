#pragma once

#include <array/ndarray.hpp>

// Elementwise functions of one array, named as in numpy and lazy like the binary ones. On float and
// double arrays, exp and log are within 1 ulp and sin and cos within 2 ulp (3 on the GPU).

namespace sci::ops {

    using array::operand;

    template <operand source>
    auto negative(source&& input) {
        return array::apply<core::Opcode::neg>(std::forward<source>(input));
    }

    template <operand source>
    auto abs(source&& input) {
        return array::apply<core::Opcode::abs>(std::forward<source>(input));
    }

    template <operand source>
    auto sqrt(source&& input) {
        return array::apply<core::Opcode::sqrt>(std::forward<source>(input));
    }

    template <operand source>
    auto exp(source&& input) {
        return array::apply<core::Opcode::exp>(std::forward<source>(input));
    }

    template <operand source>
    auto log(source&& input) {
        return array::apply<core::Opcode::log>(std::forward<source>(input));
    }

    template <operand source>
    auto sin(source&& input) {
        return array::apply<core::Opcode::sin>(std::forward<source>(input));
    }

    template <operand source>
    auto cos(source&& input) {
        return array::apply<core::Opcode::cos>(std::forward<source>(input));
    }

}
