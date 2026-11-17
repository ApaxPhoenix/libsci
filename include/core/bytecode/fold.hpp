#pragma once

/// @file
/// @brief Reductions over a chunk's result.

#include "core/types.hpp"

namespace sci::core {

    /// Reductions over a chunk's result.
    enum class Fold : uint8 { sum, minimum, maximum, count };

}
