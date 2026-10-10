#pragma once

#include "core/bytecode/precision.hpp"

namespace sci::core {

    // Maps a C++ element type to its Precision.
    template <typename element>
    struct Traits;

    template <>
    struct Traits<float> {
        static constexpr auto precision = Precision::single;
    };

    template <>
    struct Traits<double> {
        static constexpr auto precision = Precision::wide;
    };

    template <typename element>
    concept native = requires { Traits<element>::precision; };

}
