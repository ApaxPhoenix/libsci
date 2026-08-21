#pragma once

#include "core/types.hpp"

#include <array>

namespace sci::array {

    template <core::size rank>
    struct shape {
        std::array<core::size, rank> extents{};

        [[nodiscard]] constexpr core::size total() const noexcept {
            core::size count = 1;
            for (auto value : extents) {
                count *= value;
            }
            return count;
        }

        constexpr core::size operator[](core::size index) const noexcept {
            return extents[index];
        }

        constexpr core::size& operator[](core::size index) noexcept {
            return extents[index];
        }
    };

}