#pragma once

#include "core/types.hpp"

namespace sci::core::memory {

    template <typename type>
    class cursor {
    public:
        constexpr cursor() noexcept = default;

        constexpr explicit cursor(type* origin, size offset = 0) noexcept
            : pointer(origin + offset) {}

        constexpr void advance(size offset) noexcept {
            pointer += offset;
        }

        constexpr void jump(const size index, const size stride) noexcept {
            pointer += index * stride;
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