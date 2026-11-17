#pragma once

/// @file
/// @brief One instruction of a chunk, and the value ids it reads and writes.

#include "core/bytecode/code.hpp"
#include "core/types.hpp"

namespace sci::core {

    /// A value id packs its type into the top two bits and its index into the low six.
    namespace slot {
        /// @brief The tag of an input's value id.
        inline constexpr uint8 input = 0x00;
        /// @brief The tag of a constant's value id.
        inline constexpr uint8 literal = 0x40;
        /// @brief The tag of a temporary's value id.
        inline constexpr uint8 temporary = 0x80;
        /// @brief The output's value id.
        inline constexpr uint8 output = 0xC0;
        /// @brief No value.
        inline constexpr uint8 none = 0xFF;
        /// @brief The bits holding a value id's tag.
        inline constexpr uint8 types = 0xC0;
    }

    /// target = code(sources...), all value ids.
    struct Instruction {
        /// @brief What the instruction does.
        Code code;
        /// @brief The value it writes.
        uint8 target;
        /// @brief The values it reads; slot::none for those it does not.
        uint8 sources[3];
    };

}
