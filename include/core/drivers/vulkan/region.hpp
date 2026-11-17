#pragma once

/// @file
/// @brief A piece of scratch memory, by device address and host pointer.

#include "core/types.hpp"

namespace sci::core::drivers::vulkan {

    /// A piece of scratch memory, by device address and host pointer.
    struct Region {
        /// @brief Its device address.
        uint64 address{0};
        /// @brief Where the host sees it.
        void *host{nullptr};
    };

}
