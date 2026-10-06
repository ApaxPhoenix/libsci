#pragma once

#include "core/types.hpp"

namespace sci::core {

    // Operations a chunk can contain. The order is shared with the CPU kernel tables and with
    // shaders/interpreter.slang, so append new operations at the end (before count).
    enum class Opcode : uint8 {
        copy,
        neg,
        abs,
        sqrt,
        exp,
        log,
        sin,
        cos,
        add,
        sub,
        mul,
        div,
        min,
        max,
        fma,
        count
    };

}
