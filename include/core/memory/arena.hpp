#pragma once

#include "core/arch/alignment.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <new>

namespace sci::core::memory {

class arena {
    struct chunk {
        std::byte* data{nullptr};
        cursor<std::byte> head{};
        size total{0};
        size bound{arch::cpu};
        chunk* next{nullptr};
    };

public:
    explicit arena(const size bytes)
        : limit(bytes) {
        first = compose(limit, arch::cpu);
        active = first;
    }

    ~arena() {
        dispose();
    }

    arena(const arena&) = delete;
    arena& operator=(const arena&) = delete;

    arena(arena&& other) noexcept
        : first(other.first), active(other.active), limit(other.limit) {
        other.first = nullptr;
        other.active = nullptr;
        other.limit = 0;
    }

    arena& operator=(arena&& other) noexcept {
        if (this != &other) {
            dispose();
            first = other.first;
            active = other.active;
            limit = other.limit;
            other.first = nullptr;
            other.active = nullptr;
            other.limit = 0;
        }
        return *this;
    }

    template <typename type>
    [[nodiscard]] type* allocate(const size count, const size bound = arch::cpu) {
        const size bytes = count * sizeof(type) + bound - 1 & ~(bound - 1);
        if (static_cast<size>(active->data + active->total - active->head.raw()) < bytes) {
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
    [[nodiscard]] cursor<type> acquire(const size count, const size bound = arch::cpu) {
        return cursor<type>(allocate<type>(count, bound));
    }

    void reset() noexcept {
        active = first;
        for (auto* node = first; node; node = node->next) {
            node->head = cursor(node->data);
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
        for (const auto* node = first; node; node = node->next) {
            sum += node->total;
        }
        return sum;
    }

private:
    [[nodiscard]] static chunk* compose(const size bytes, const size bound) {
        auto* node = new chunk();
        node->data = static_cast<std::byte*>(
            operator new[](bytes, std::align_val_t{bound}));
        node->head = cursor(node->data);
        node->total = bytes;
        node->bound = bound;
        node->next = nullptr;
        return node;
    }

    chunk* first{nullptr};
    chunk* active{nullptr};
    size limit{0};
};

}