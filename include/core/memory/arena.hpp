#pragma once

#include "core/alignment.hpp"
#include "core/memory/block.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace sci::core::memory {

    // Bump allocator over a chain of blocks: allocate() hands out the next piece of the active block,
    // reset() makes every block reusable at once, and the memory is freed with the arena. Move-only.
    class Arena {
    public:
        explicit Arena(const size bytes)
            : first(std::make_unique<Block>(bytes, alignment::processor)), active(first.get()), limit(bytes) {}

        template <typename type>
        [[nodiscard]] type* allocate(const size count, const size bound = alignment::processor) {
            const size bytes = count * sizeof(type) + bound - 1 & ~(bound - 1);
            // Blocks kept from before a reset() can be too small for this piece; those are skipped.
            while (std::cmp_less(static_cast<std::byte*>(active->allocation.data()) + active->allocation.capacity() - active->head.raw(), bytes)) {
                if (!active->next) {
                    active->next = std::make_unique<Block>(std::max(bytes, limit), bound);
                }
                active = active->next.get();
            }
            type* pointer = reinterpret_cast<type*>(active->head.raw());
            active->head.advance(bytes);
            return pointer;
        }

        void reset() noexcept {
            active = first.get();
            for (Block* block = first.get(); block; block = block->next.get()) {
                block->head = Cursor(static_cast<std::byte*>(block->allocation.data()));
            }
        }

        [[nodiscard]] size capacity() const noexcept {
            size total = 0;
            for (const Block* block = first.get(); block; block = block->next.get()) {
                total += block->allocation.capacity();
            }
            return total;
        }

    private:
        std::unique_ptr<Block> first;
        Block* active{nullptr};
        size limit{0};
    };

}
