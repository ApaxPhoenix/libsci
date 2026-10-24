#pragma once

#include "core/memory/allocation.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <memory>

namespace sci::core::memory {

    // One piece of memory in an Arena's chain: the bytes, how far they are used, and the next block.
    struct Block {
        Allocation allocation;
        Cursor<std::byte> head;
        std::unique_ptr<Block> next;

        Block(const size bytes, const size bound)
            : allocation(bytes, bound), head(static_cast<std::byte*>(allocation.data())) {}
    };

}
