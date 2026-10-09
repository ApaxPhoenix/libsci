#pragma once

#include "core/bytecode/code.hpp"

#include <cmath>

namespace sci::array {

    // Scalar semantics of each code, for element types the engine does not run natively.
    template <core::Code code, typename type>
    type calculate(const type first, const type second = type{}, const type third = type{}) noexcept {
        using core::Code;
        if constexpr (code == Code::copy) return first;
        else if constexpr (code == Code::neg) return static_cast<type>(-first);
        else if constexpr (code == Code::abs) return first < type{} ? static_cast<type>(-first) : first;
        else if constexpr (code == Code::sqrt) return static_cast<type>(std::sqrt(first));
        else if constexpr (code == Code::exp) return static_cast<type>(std::exp(first));
        else if constexpr (code == Code::log) return static_cast<type>(std::log(first));
        else if constexpr (code == Code::sin) return static_cast<type>(std::sin(first));
        else if constexpr (code == Code::cos) return static_cast<type>(std::cos(first));
        else if constexpr (code == Code::tan) return static_cast<type>(std::tan(first));
        else if constexpr (code == Code::asin) return static_cast<type>(std::asin(first));
        else if constexpr (code == Code::acos) return static_cast<type>(std::acos(first));
        else if constexpr (code == Code::atan) return static_cast<type>(std::atan(first));
        else if constexpr (code == Code::sinh) return static_cast<type>(std::sinh(first));
        else if constexpr (code == Code::cosh) return static_cast<type>(std::cosh(first));
        else if constexpr (code == Code::tanh) return static_cast<type>(std::tanh(first));
        else if constexpr (code == Code::asinh) return static_cast<type>(std::asinh(first));
        else if constexpr (code == Code::acosh) return static_cast<type>(std::acosh(first));
        else if constexpr (code == Code::atanh) return static_cast<type>(std::atanh(first));
        else if constexpr (code == Code::add) return static_cast<type>(first + second);
        else if constexpr (code == Code::sub) return static_cast<type>(first - second);
        else if constexpr (code == Code::mul) return static_cast<type>(first * second);
        else if constexpr (code == Code::div) return static_cast<type>(first / second);
        else if constexpr (code == Code::min) return second < first ? second : first;
        else if constexpr (code == Code::max) return second > first ? second : first;
        else return static_cast<type>(first * second + third);
    }

}
