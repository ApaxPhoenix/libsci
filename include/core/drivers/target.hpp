#pragma once

/// @file
/// @brief Where an array lives and where its expressions run.

#include "core/types.hpp"

namespace sci::core::drivers {

    /// Where an array lives and where its expressions run.
    enum class Target : uint8 { processor, graphics };

}
