#pragma once

/// @file
/// @brief Extent of each dimension, like numpy's shape tuple.

#include "core/types.hpp"

#include <array>

namespace sci::array {

    /// Extent of each dimension, like numpy's shape tuple: Shape<2>{rows, columns}.
    template <core::size rank> struct Shape {
        /// @brief The extent of each dimension.
        std::array<core::size, rank> extents{};

        /// Number of elements (the product of the extents).
        /// @return The number of elements.
        [[nodiscard]] constexpr core::size total() const noexcept {
            core::size count = 1;
            for (auto value : extents) {
                count *= value;
            }
            return count;
        }

        /// @brief The extent of a dimension.
        /// @param index The dimension.
        /// @return The extent.
        constexpr core::size operator[](core::size index) const noexcept { return extents[index]; }

        /// @brief Whether two shapes have the same extents.
        /// @return Whether they do.
        constexpr bool operator==(const Shape &) const noexcept = default;
    };

}
