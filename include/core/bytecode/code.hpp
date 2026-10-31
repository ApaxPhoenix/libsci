#pragma once

#include "core/types.hpp"

namespace sci::core {

    // Operations a chunk can contain. The order is shared with the CPU kernel tables and with
    // shaders/interpreter.slang, so append new operations at the end (before count).
    //
    // Naming: one lowercase word per operation. Functions that exist in <math.h> keep their C
    // name (sin, atanh). Everything else is spelled out in full (minimum, exponential).
    //
    // Every operation is elementwise: it reads one element from each input and writes one
    // element of the output.
    enum class Code : uint8 {
        copy,
        negate,
        absolute,

        root,
        exponential,
        logarithm,

        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        csc,
        sec,
        cot,
        acsc,
        asec,
        acot,

        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
        csch,
        sech,
        coth,
        acsch,
        asech,
        acoth,

        add,
        subtract,
        multiply,
        divide,
        minimum,
        maximum,
        fuse,
        count
    };

}