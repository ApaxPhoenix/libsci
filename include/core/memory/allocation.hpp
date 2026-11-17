#pragma once

/// @file
/// @brief Aligned host memory that frees itself.

#include "core/alignment.hpp"
#include "core/types.hpp"

#include <new>
#include <utility>

namespace sci::core::memory {

    /// Aligned host memory that frees itself: the bytes behind host arrays, arena blocks, CPU device
    /// memory and per-thread scratch space. Move-only.
    class Allocation {
    public:
        Allocation() noexcept = default;

        /// @brief An aligned allocation.
        /// @param bytes How many bytes.
        /// @param alignment The alignment, a power of two.
        Allocation(const size bytes, const size alignment) : alignment(alignment) { reserve(bytes); }

        ~Allocation() {
            if (pointer) ::operator delete(pointer, std::align_val_t{alignment});
        }

        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;

        /// @brief Takes another allocation's memory, leaving it empty.
        /// @param allocation The allocation moved from.
        Allocation(Allocation &&allocation) noexcept
            : pointer(std::exchange(allocation.pointer, nullptr)), bytes(std::exchange(allocation.bytes, 0)),
              alignment(allocation.alignment) {}

        /// @brief Takes another allocation's memory, leaving it empty.
        /// @param allocation The allocation moved from.
        /// @return This allocation.
        Allocation &operator=(Allocation &&allocation) noexcept {
            std::swap(pointer, allocation.pointer);
            std::swap(bytes, allocation.bytes);
            std::swap(alignment, allocation.alignment);
            return *this;
        }

        /// Grows to at least `amount` bytes and returns the memory. Growing does not keep the contents.
        /// @param amount How many bytes.
        /// @return The memory.
        void *reserve(const size amount) {
            if (amount > bytes) {
                if (pointer) ::operator delete(pointer, std::align_val_t{alignment});
                pointer = ::operator new(amount, std::align_val_t{alignment});
                bytes = amount;
            }
            return pointer;
        }

        /// @brief The memory.
        /// @return A pointer to it.
        [[nodiscard]] void *data() const noexcept { return pointer; }
        /// @brief How many bytes it holds.
        /// @return The capacity.
        [[nodiscard]] size capacity() const noexcept { return bytes; }

    private:
        /// @brief The memory.
        void *pointer{nullptr};
        /// @brief How many bytes.
        size bytes{0};
        /// @brief Its alignment.
        size alignment{alignment::processor};
    };

}
