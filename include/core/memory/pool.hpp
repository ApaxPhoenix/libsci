#pragma once

#include "core/memory/cursor.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <new>
#include <utility>

namespace sci::core::memory {

template <typename type, size count, size align = 0>
class pool {
    struct node {
        node* next{nullptr};
    };

    static constexpr size base = alignof(type) > alignof(node) ? alignof(type) : alignof(node);
    static constexpr size bound = align != 0 ? align : base;
    static constexpr size raw = sizeof(type) > sizeof(node) ? sizeof(type) : sizeof(node);
    static constexpr size stride = raw + bound - 1 & ~(bound - 1);

public:
    pool() {
        block = static_cast<std::byte*>(
            ::operator new[](stride * count, std::align_val_t{bound}));
        reset();
    }

    ~pool() {
        dispose();
    }

    pool(const pool&) = delete;
    pool& operator=(const pool&) = delete;

    pool(pool&& other) noexcept
        : block(other.block), free(other.free) {
        other.block = nullptr;
        other.free = nullptr;
    }

    pool& operator=(pool&& other) noexcept {
        if (this != &other) {
            dispose();
            block = other.block;
            free = other.free;
            other.block = nullptr;
            other.free = nullptr;
        }
        return *this;
    }

    template <typename... arguments>
    [[nodiscard]] type* compose(arguments&&... list) {
        if (!free) {
            throw std::bad_alloc();
        }
        auto* item = free;
        free = free->next;
        type* pointer = reinterpret_cast<type*>(item);
        ::new (static_cast<void*>(pointer)) type(std::forward<arguments>(list)...);
        return pointer;
    }

    template <typename... arguments>
    [[nodiscard]] cursor<type> acquire(arguments&&... list) {
        return cursor<type>(compose(std::forward<arguments>(list)...));
    }

    void destroy(type* pointer) noexcept {
        if (!pointer) return;
        pointer->~type();
        auto* item = reinterpret_cast<node*>(pointer);
        item->next = free;
        free = item;
    }

    void destroy(cursor<type> point) noexcept {
        destroy(point.raw());
    }

    void reset() noexcept {
        free = nullptr;
        if (!block) return;
        for (size index = 0; index < count; ++index) {
            auto* item = reinterpret_cast<node*>(block + index * stride);
            item->next = free;
            free = item;
        }
    }

    void dispose() noexcept {
        if (block) {
            operator delete[](block, std::align_val_t{bound});
            block = nullptr;
            free = nullptr;
        }
    }

private:
    std::byte* block{nullptr};
    node* free{nullptr};
};

}