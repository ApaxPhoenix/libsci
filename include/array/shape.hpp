#pragma once

#include "core/types.hpp"

#include <array>

namespace sci::array {

    // Extent of each dimension, like numpy's shape tuple: Shape<2>{rows, columns}.
    template <core::size rank>
    struct Shape {
        std::array<core::size, rank> extents{};

        // Number of elements (the product of the extents).
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

        constexpr bool operator==(const Shape&) const noexcept = default;
    };

}
