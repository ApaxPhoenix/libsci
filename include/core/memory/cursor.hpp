#pragma once

/// @file
/// @brief A position in memory that only moves forward.

#include "core/types.hpp"

namespace sci::core::memory {

    /// A position in memory that only moves forward: where an arena block hands out its next piece.
    template <typename type> class Cursor {
    public:
        constexpr Cursor() noexcept = default;

        /// @brief A cursor at a place in memory.
        /// @param origin The memory.
        /// @param offset How many elements in.
        constexpr explicit Cursor(type *origin, size offset = 0) noexcept : pointer(origin + offset) {}

        /// @brief Moves the cursor forward.
        /// @param offset How many elements.
        constexpr void advance(size offset) noexcept { pointer += offset; }

        /// @brief Where the cursor is.
        /// @return A pointer.
        [[nodiscard]] constexpr type *raw() noexcept { return pointer; }

    private:
        /// @brief Where the cursor is.
        type *pointer{nullptr};
    };

}
