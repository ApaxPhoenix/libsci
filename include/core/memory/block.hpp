#pragma once

#include "core/memory/allocation.hpp"
#include "core/memory/cursor.hpp"

#include <cstddef>

namespace sci::core::memory {

    // One piece of memory in an Arena's chain: the bytes, how far they are used, and the next block.
    struct Block {
        Allocation allocation;
        Cursor<std::byte> head;
        Block* next{nullptr};
    };

}
