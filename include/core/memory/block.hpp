#pragma once

#include "core/arch/alignment.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>

namespace sci::core::memory {

    // One piece of memory in an Arena's chain: the bytes, how far they are used, and the next block.
    struct Block {
        std::byte* data{nullptr};
        Cursor<std::byte> head;
        size total{0};
        size bound{arch::cpu};
        Block* next{nullptr};
    };

}
