#pragma once

#include "core/types.hpp"

namespace sci::core::drivers {

    // Where an array lives and where its expressions run.
    enum class Target : uint8 {
        cpu,
        gpu
    };

}
