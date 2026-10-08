#pragma once

#include "core/bytecode/code.hpp"
#include "core/types.hpp"

namespace sci::core {

    // A value id packs its kind into the top two bits and its index into the low six.
    namespace slot {
        inline constexpr uint8 input = 0x00;
        inline constexpr uint8 literal = 0x40;
        inline constexpr uint8 temporary = 0x80;
        inline constexpr uint8 output = 0xC0;
        inline constexpr uint8 none = 0xFF;
        inline constexpr uint8 kinds = 0xC0;
    }

    // target = code(sources...), all value ids.
    struct Instruction {
        Code code;
        uint8 target;
        uint8 sources[3];
    };

}
