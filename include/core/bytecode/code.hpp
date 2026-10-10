#pragma once

#include "core/types.hpp"

namespace sci::core {

    // Operations a chunk can contain. The order is shared with the CPU kernel tables and with
    // shaders/interpreter.slang, so append new operations at the end (before count).
    enum class Code : uint8 {
        copy,
        negate,
        absolute,
        root,
        exponential,
        logarithm,
        sine,
        cosine,
        tangent,
        arcsine,
        arccosine,
        arctangent,
        // 12 to 17 are the hyperbolic functions, below.
        add = 18,
        subtract,
        multiply,
        divide,
        minimum,
        maximum,
        fuse,
        count
    };

    // The hyperbolic functions share their names with the circular ones, so they get a namespace of
    // their own. They are never `case` labels: compilers warn about values without an enumerator.
    namespace hyperbolic {
        inline constexpr Code sine{12};
        inline constexpr Code cosine{13};
        inline constexpr Code tangent{14};
        inline constexpr Code arcsine{15};
        inline constexpr Code arccosine{16};
        inline constexpr Code arctangent{17};
    }

}
