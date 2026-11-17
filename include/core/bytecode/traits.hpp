#pragma once

/// @file
/// @brief Maps a C++ element type to its Precision.

#include "core/bytecode/precision.hpp"

namespace sci::core {

    /// Maps a C++ element type to its Precision.
    template <typename element> struct Traits;

    /// @brief float runs natively, at single precision.
    template <> struct Traits<float> {
        static constexpr auto precision = Precision::single;
    };

    /// @brief double runs natively, at wide precision.
    template <> struct Traits<double> {
        static constexpr auto precision = Precision::wide;
    };

    template <typename element>
    concept native = requires { Traits<element>::precision; };

}
