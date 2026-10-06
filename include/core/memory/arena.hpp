#pragma once

#include "core/arch/alignment.hpp"
#include "core/memory/block.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <new>
#include <utility>

namespace sci::core::memory {

class Arena {
public:
    explicit Arena(const size bytes)
        : limit(bytes) {
        first = compose(limit, arch::cpu);
        active = first;
    }

    ~Arena() {
        dispose();
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& arena) noexcept
        : first(arena.first), active(arena.active), limit(arena.limit) {
        arena.first = nullptr;
        arena.active = nullptr;
        arena.limit = 0;
    }

    Arena& operator=(Arena&& arena) noexcept {
        if (this != &arena) {
            dispose();
            first = arena.first;
            active = arena.active;
            limit = arena.limit;
            arena.first = nullptr;
            arena.active = nullptr;
            arena.limit = 0;
        }
        return *this;
    }

    template <typename type>
    [[nodiscard]] type* allocate(const size count, const size bound = arch::cpu) {
        const size bytes = count * sizeof(type) + bound - 1 & ~(bound - 1);
        if (std::cmp_less(active->data + active->total - active->head.raw(), bytes)) {
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

    template <typename type>
    [[nodiscard]] Cursor<type> acquire(const size count, const size bound = arch::cpu) {
        return Cursor<type>(allocate<type>(count, bound));
    }

    void reset() noexcept {
        active = first;
        for (auto* block = first; block; block = block->next) {
            block->head = Cursor(block->data);
        }
    }

    void dispose() noexcept {
        while (first) {
            auto* next = first->next;
            if (first->data) {
                operator delete[](first->data, std::align_val_t{first->bound});
            }
            delete first;
            first = next;
        }
        active = nullptr;
        limit = 0;
    }

    [[nodiscard]] size capacity() const noexcept {
        size sum = 0;
        for (const auto* block = first; block; block = block->next) {
            sum += block->total;
        }
        return sum;
    }

private:
    [[nodiscard]] static Block* compose(const size bytes, const size bound) {
        auto* block = new Block();
        block->data = static_cast<std::byte*>(
            operator new[](bytes, std::align_val_t{bound}));
        block->head = Cursor(block->data);
        block->total = bytes;
        block->bound = bound;
        block->next = nullptr;
        return block;
    }

    Block* first{nullptr};
    Block* active{nullptr};
    size limit{0};
};

}