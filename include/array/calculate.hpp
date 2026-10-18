#pragma once

#include "core/bytecode/code.hpp"

#include <cmath>

namespace sci::array {

    // Scalar semantics of each code, for element types the engine does not run natively.
    template <core::Code code, typename type>
    type calculate(const type first, const type second = type{}, const type third = type{}) noexcept {
        using core::Code;
        if constexpr (code == Code::copy) return first;
        else if constexpr (code == Code::negate) return static_cast<type>(-first);
        else if constexpr (code == Code::absolute) return first < type{} ? static_cast<type>(-first) : first;
        else if constexpr (code == Code::root) return static_cast<type>(std::sqrt(first));
        else if constexpr (code == Code::exponential) return static_cast<type>(std::exp(first));
        else if constexpr (code == Code::logarithm) return static_cast<type>(std::log(first));
        else if constexpr (code == Code::sine) return static_cast<type>(std::sin(first));
        else if constexpr (code == Code::cosine) return static_cast<type>(std::cos(first));
        else if constexpr (code == Code::tangent) return static_cast<type>(std::tan(first));
        else if constexpr (code == Code::arcsine) return static_cast<type>(std::asin(first));
        else if constexpr (code == Code::arccosine) return static_cast<type>(std::acos(first));
        else if constexpr (code == Code::arctangent) return static_cast<type>(std::atan(first));
        else if constexpr (code == core::hyperbolic::sine) return static_cast<type>(std::sinh(first));
        else if constexpr (code == core::hyperbolic::cosine) return static_cast<type>(std::cosh(first));
        else if constexpr (code == core::hyperbolic::tangent) return static_cast<type>(std::tanh(first));
        else if constexpr (code == core::hyperbolic::arcsine) return static_cast<type>(std::asinh(first));
        else if constexpr (code == core::hyperbolic::arccosine) return static_cast<type>(std::acosh(first));
        else if constexpr (code == core::hyperbolic::arctangent) return static_cast<type>(std::atanh(first));
        else if constexpr (code == Code::add) return static_cast<type>(first + second);
        else if constexpr (code == Code::subtract) return static_cast<type>(first - second);
        else if constexpr (code == Code::multiply) return static_cast<type>(first * second);
        else if constexpr (code == Code::divide) return static_cast<type>(first / second);
        else if constexpr (code == Code::minimum) return second < first ? second : first;
        else if constexpr (code == Code::maximum) return second > first ? second : first;
        else return static_cast<type>(first * second + third);
    }

}
