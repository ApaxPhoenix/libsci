#pragma once

#include "core/types.hpp"

namespace sci::core::memory {

    // A position in memory that only moves forward: where an arena block hands out its next piece.
    template <typename type>
    class Cursor {
    public:
        constexpr Cursor() noexcept = default;

        constexpr explicit Cursor(type* origin, size offset = 0) noexcept
            : pointer(origin + offset) {}

        constexpr void advance(size offset) noexcept {
            pointer += offset;
        }

        [[nodiscard]] constexpr type* raw() noexcept {
            return pointer;
        }

    private:
        type* pointer{nullptr};
    };

}