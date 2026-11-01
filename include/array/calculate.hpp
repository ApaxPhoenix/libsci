#pragma once

#include "core/bytecode/code.hpp"

#include <cmath>

namespace sci::array {

    // Scalar semantics of each code, for element types the engine does not run natively.
    template <core::Code code, typename type>
    type calculate(const type alpha, const type beta = type{}, const type gamma = type{}) noexcept {
        using core::Code;
        if constexpr (code == Code::copy) return alpha;
        else if constexpr (code == Code::negate) return static_cast<type>(-alpha);
        else if constexpr (code == Code::absolute) return alpha < type{} ? static_cast<type>(-alpha) : alpha;
        else if constexpr (code == Code::root) return static_cast<type>(std::sqrt(alpha));
        else if constexpr (code == Code::exponential) return static_cast<type>(std::exp(alpha));
        else if constexpr (code == Code::logarithm) return static_cast<type>(std::log(alpha));
        else if constexpr (code == Code::sin) return static_cast<type>(std::sin(alpha));
        else if constexpr (code == Code::cos) return static_cast<type>(std::cos(alpha));
        else if constexpr (code == Code::tan) return static_cast<type>(std::tan(alpha));
        else if constexpr (code == Code::asin) return static_cast<type>(std::asin(alpha));
        else if constexpr (code == Code::acos) return static_cast<type>(std::acos(alpha));
        else if constexpr (code == Code::atan) return static_cast<type>(std::atan(alpha));
        else if constexpr (code == Code::csc) return static_cast<type>(1.0 / std::sin(alpha));
        else if constexpr (code == Code::sec) return static_cast<type>(1.0 / std::cos(alpha));
        else if constexpr (code == Code::cot) return static_cast<type>(1.0 / std::tan(alpha));
        else if constexpr (code == Code::acsc) return static_cast<type>(std::asin(1.0 / alpha));
        else if constexpr (code == Code::asec) return static_cast<type>(std::acos(1.0 / alpha));
        else if constexpr (code == Code::acot) return static_cast<type>(std::atan(1.0 / alpha));
        else if constexpr (code == Code::sinh) return static_cast<type>(std::sinh(alpha));
        else if constexpr (code == Code::cosh) return static_cast<type>(std::cosh(alpha));
        else if constexpr (code == Code::tanh) return static_cast<type>(std::tanh(alpha));
        else if constexpr (code == Code::asinh) return static_cast<type>(std::asinh(alpha));
        else if constexpr (code == Code::acosh) return static_cast<type>(std::acosh(alpha));
        else if constexpr (code == Code::atanh) return static_cast<type>(std::atanh(alpha));
        else if constexpr (code == Code::csch) return static_cast<type>(1.0 / std::sinh(alpha));
        else if constexpr (code == Code::sech) return static_cast<type>(1.0 / std::cosh(alpha));
        else if constexpr (code == Code::coth) return static_cast<type>(1.0 / std::tanh(alpha));
        else if constexpr (code == Code::acsch) return static_cast<type>(std::asinh(1.0 / alpha));
        else if constexpr (code == Code::asech) return static_cast<type>(std::acosh(1.0 / alpha));
        else if constexpr (code == Code::acoth) return static_cast<type>(std::atanh(1.0 / alpha));
        else if constexpr (code == Code::add) return static_cast<type>(alpha + beta);
        else if constexpr (code == Code::subtract) return static_cast<type>(alpha - beta);
        else if constexpr (code == Code::multiply) return static_cast<type>(alpha * beta);
        else if constexpr (code == Code::divide) return static_cast<type>(alpha / beta);
        else if constexpr (code == Code::minimum) return beta < alpha ? beta : alpha;
        else if constexpr (code == Code::maximum) return beta > alpha ? beta : alpha;
        else return static_cast<type>(alpha * beta + gamma);
    }

}