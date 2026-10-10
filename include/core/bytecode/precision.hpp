#pragma once

#include "core/types.hpp"

namespace sci::core {

    // Element types the engine evaluates natively (float and double); everything else uses a plain
    // C++ loop.
    enum class Precision : uint8 {
        single,
        wide
    };

}
