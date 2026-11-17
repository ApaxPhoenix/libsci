#pragma once

/// @file
/// @brief One piece of memory in an Arena's chain.

#include "core/memory/allocation.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <memory>

namespace sci::core::memory {

    /// One piece of memory in an Arena's chain: the bytes, how far they are used, and the next block.
    struct Block {
        /// @brief The block's memory.
        Allocation allocation;
        /// @brief Where its next piece starts.
        Cursor<std::byte> head;
        /// @brief The block after it.
        std::unique_ptr<Block> next;

        /// @brief A block of memory.
        /// @param bytes How many bytes.
        /// @param bound The alignment, a power of two.
        Block(const size bytes, const size bound)
            : allocation(bytes, bound), head(static_cast<std::byte *>(allocation.data())) {}
    };

}
