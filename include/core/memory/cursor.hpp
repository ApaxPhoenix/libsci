#pragma once

#include "core/types.hpp"

namespace sci::core::memory {

    template <typename type>
    class Cursor {
    public:
        constexpr Cursor() noexcept = default;

        constexpr explicit Cursor(type* origin, size offset = 0) noexcept
            : pointer(origin + offset) {}

        constexpr void advance(size offset) noexcept {
            pointer += offset;
        }

        [[nodiscard]] constexpr type& get() noexcept {
            return *pointer;
        }

        [[nodiscard]] constexpr const type& get() const noexcept {
            return *pointer;
        }

        [[nodiscard]] constexpr type* raw() noexcept {
            return pointer;
        }

    private:
        type* pointer{nullptr};
    };

}