#pragma once

/// @file
/// @brief Elements to skip per step along each dimension (numpy's strides, in elements instead of
///       bytes).

#include "array/shape.hpp"
#include "core/types.hpp"

#include <array>

namespace sci::array {

    /// Elements to skip per step along each dimension (numpy's strides, in elements instead of bytes).
    template <core::size rank> struct Stride {
        /// @brief How many elements each dimension steps over.
        std::array<core::size, rank> steps{};

        /// Row-major (C order) strides for a shape.
        /// @param shape The shape.
        /// @return The strides.
        static constexpr Stride contiguous(Shape<rank> shape) noexcept {
            Stride result{};
            core::size step = 1;
            for (core::size index = rank; index > 0; --index) {
                result.steps[index - 1] = step;
                step *= shape[index - 1];
            }
            return result;
        }

        /// @brief How many elements a dimension steps over.
        /// @param index The dimension.
        /// @return The stride.
        constexpr core::size operator[](core::size index) const noexcept { return steps[index]; }

        /// Flat offset of a multi-dimensional index.
        /// @param indices Its index along each dimension.
        /// @return The flat offset.
        constexpr core::size offset(std::array<core::size, rank> indices) const noexcept {
            core::size total = 0;
            for (core::size index = 0; index < rank; ++index) {
                total += indices[index] * steps[index];
            }
            return total;
        }
    };

}
