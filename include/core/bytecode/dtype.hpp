#pragma once

#include "core/types.hpp"

namespace sci::core {

    // Element types the engine evaluates natively; everything else uses a plain C++ loop.
    enum class Dtype : uint8 {
        float32,
        float64
    };

}
