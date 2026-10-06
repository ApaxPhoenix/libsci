#pragma once

#include "core/bytecode/dtype.hpp"

namespace sci::core {

    // Maps a C++ element type to its Dtype.
    template <typename element>
    struct Traits;

    template <>
    struct Traits<float> {
        static constexpr Dtype type = Dtype::float32;
    };

    template <>
    struct Traits<double> {
        static constexpr Dtype type = Dtype::float64;
    };

    template <typename element>
    concept native = requires { Traits<element>::type; };

}
