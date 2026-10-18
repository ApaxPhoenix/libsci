#pragma once

#include <array/array.hpp>

// Elementwise functions of one array, lazy like the binary ones. On float and double arrays the
// exponential and logarithm are within 1 ulp, and sine and cosine within 2 ulp (3 on the GPU).

namespace sci::operations {

    using array::operand;

    template <operand source>
    auto negative(source&& input) {
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
    auto sine(source&& input) {
        return array::apply<core::Code::sine>(std::forward<source>(input));
    }

    template <operand source>
    auto cosine(source&& input) {
        return array::apply<core::Code::cosine>(std::forward<source>(input));
    }

    template <operand source>
    auto tangent(source&& input) {
        return array::apply<core::Code::tangent>(std::forward<source>(input));
    }

    template <operand source>
    auto arcsine(source&& input) {
        return array::apply<core::Code::arcsine>(std::forward<source>(input));
    }

    template <operand source>
    auto arccosine(source&& input) {
        return array::apply<core::Code::arccosine>(std::forward<source>(input));
    }

    template <operand source>
    auto arctangent(source&& input) {
        return array::apply<core::Code::arctangent>(std::forward<source>(input));
    }

    // sinh, cosh, tanh and their inverses: sci::hyperbolic::sine and so on.
    namespace hyperbolic {

        template <operand source>
        auto sine(source&& input) {
            return array::apply<core::hyperbolic::sine>(std::forward<source>(input));
        }

        template <operand source>
        auto cosine(source&& input) {
            return array::apply<core::hyperbolic::cosine>(std::forward<source>(input));
        }

        template <operand source>
        auto tangent(source&& input) {
            return array::apply<core::hyperbolic::tangent>(std::forward<source>(input));
        }

        template <operand source>
        auto arcsine(source&& input) {
            return array::apply<core::hyperbolic::arcsine>(std::forward<source>(input));
        }

        template <operand source>
        auto arccosine(source&& input) {
            return array::apply<core::hyperbolic::arccosine>(std::forward<source>(input));
        }

        template <operand source>
        auto arctangent(source&& input) {
            return array::apply<core::hyperbolic::arctangent>(std::forward<source>(input));
        }

    }

}
