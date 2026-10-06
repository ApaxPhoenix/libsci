#pragma once

#include "core/bytecode/opcode.hpp"

#include <cmath>

namespace sci::array {

    // Scalar semantics of each opcode, for element types the engine does not run natively.
    template <core::Opcode code, typename type>
    type calculate(const type first, const type second = type{}, const type third = type{}) noexcept {
        using core::Opcode;
        if constexpr (code == Opcode::copy) return first;
        else if constexpr (code == Opcode::neg) return static_cast<type>(-first);
        else if constexpr (code == Opcode::abs) return first < type{} ? static_cast<type>(-first) : first;
        else if constexpr (code == Opcode::sqrt) return static_cast<type>(std::sqrt(first));
        else if constexpr (code == Opcode::exp) return static_cast<type>(std::exp(first));
        else if constexpr (code == Opcode::log) return static_cast<type>(std::log(first));
        else if constexpr (code == Opcode::sin) return static_cast<type>(std::sin(first));
        else if constexpr (code == Opcode::cos) return static_cast<type>(std::cos(first));
        else if constexpr (code == Opcode::add) return static_cast<type>(first + second);
        else if constexpr (code == Opcode::sub) return static_cast<type>(first - second);
        else if constexpr (code == Opcode::mul) return static_cast<type>(first * second);
        else if constexpr (code == Opcode::div) return static_cast<type>(first / second);
        else if constexpr (code == Opcode::min) return second < first ? second : first;
        else if constexpr (code == Opcode::max) return second > first ? second : first;
        else return static_cast<type>(first * second + third);
    }

}
