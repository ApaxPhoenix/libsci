#pragma once

#include "core/alignment.hpp"
#include "core/memory/allocation.hpp"
#include "core/memory/block.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <utility>

namespace sci::core::memory {

    // Bump allocator over a chain of blocks: allocate() hands out the next piece of the active block,
    // reset() makes every block reusable at once, and the memory is freed with the arena.
    class Arena {
    public:
        explicit Arena(const size bytes)
            : limit(bytes) {
            first = compose(limit, alignment::processor);
            active = first;
        }

        ~Arena() {
            dispose();
        }

        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        Arena(Arena&& arena) noexcept
            : first(std::exchange(arena.first, nullptr)),
              active(std::exchange(arena.active, nullptr)),
              limit(std::exchange(arena.limit, 0)) {}

        Arena& operator=(Arena&& arena) noexcept {
            if (this != &arena) {
                dispose();
                first = std::exchange(arena.first, nullptr);
                active = std::exchange(arena.active, nullptr);
                limit = std::exchange(arena.limit, 0);
            }
            return *this;
        }

        template <typename type>
        [[nodiscard]] type* allocate(const size count, const size bound = alignment::processor) {
            const size bytes = count * sizeof(type) + bound - 1 & ~(bound - 1);
            const std::byte* end = static_cast<std::byte*>(active->allocation.data()) + active->allocation.capacity();
            if (std::cmp_less(end - active->head.raw(), bytes)) {
                const size scale = bytes > limit ? bytes : limit;
                if (!active->next) {
                    active->next = compose(scale, bound);
                }
                active = active->next;
            }
            type* pointer = reinterpret_cast<type*>(active->head.raw());
            active->head.advance(bytes);
            return pointer;
        }

        void reset() noexcept {
            active = first;
            for (Block* block = first; block; block = block->next) {
                block->head = Cursor(static_cast<std::byte*>(block->allocation.data()));
            }
        }

        void dispose() noexcept {
            while (first) {
                delete std::exchange(first, first->next);
            }
            active = nullptr;
            limit = 0;
        }

        [[nodiscard]] size capacity() const noexcept {
            size total = 0;
            for (const Block* block = first; block; block = block->next) {
                total += block->allocation.capacity();
            }
            return total;
        }

    private:
        [[nodiscard]] static Block* compose(const size bytes, const size bound) {
            auto* block = new Block{.allocation = Allocation(bytes, bound), .head = {}, .next = nullptr};
            block->head = Cursor(static_cast<std::byte*>(block->allocation.data()));
            return block;
        }

        Block* first{nullptr};
        Block* active{nullptr};
        size limit{0};
    };

}
