#pragma once

#include "core/alignment.hpp"
#include "core/types.hpp"

#include <new>
#include <utility>

namespace sci::core::memory {

    // Aligned host memory that frees itself: the bytes behind host arrays, arena blocks, CPU device
    // memory and per-thread scratch space. Move-only.
    class Allocation {
    public:
        Allocation() noexcept = default;

        Allocation(const size bytes, const size alignment)
            : alignment(alignment) {
            reserve(bytes);
        }

        ~Allocation() {
            if (pointer) ::operator delete(pointer, std::align_val_t{alignment});
        }

        Allocation(const Allocation&) = delete;
        Allocation& operator=(const Allocation&) = delete;

        Allocation(Allocation&& allocation) noexcept
            : pointer(std::exchange(allocation.pointer, nullptr)),
              bytes(std::exchange(allocation.bytes, 0)),
              alignment(allocation.alignment) {}

        Allocation& operator=(Allocation&& allocation) noexcept {
            std::swap(pointer, allocation.pointer);
            std::swap(bytes, allocation.bytes);
            std::swap(alignment, allocation.alignment);
            return *this;
        }

        // Grows to at least `amount` bytes and returns the memory. Growing does not keep the contents.
        void* reserve(const size amount) {
            if (amount > bytes) {
                if (pointer) ::operator delete(pointer, std::align_val_t{alignment});
                pointer = ::operator new(amount, std::align_val_t{alignment});
                bytes = amount;
            }
            return pointer;
        }

        [[nodiscard]] void* data() const noexcept { return pointer; }
        [[nodiscard]] size capacity() const noexcept { return bytes; }

    private:
        void* pointer{nullptr};
        size bytes{0};
        size alignment{alignment::processor};
    };

}
