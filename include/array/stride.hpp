#pragma once

#include "array/shape.hpp"
#include "core/types.hpp"

#include <array>

namespace sci::array {

    // Elements to skip per step along each dimension (numpy's strides, in elements instead of bytes).
    template <core::size rank>
    struct Stride {
        std::array<core::size, rank> steps{};

        // Row-major (C order) strides for a shape.
        static constexpr Stride contiguous(Shape<rank> shape) noexcept {
            Stride result{};
            core::size step = 1;
            for (core::size index = rank; index > 0; --index) {
                result.steps[index - 1] = step;
                step *= shape[index - 1];
            }
            return result;
        }

        constexpr core::size operator[](core::size index) const noexcept {
            return steps[index];
        }

        // Flat offset of a multi-dimensional index.
        constexpr core::size offset(std::array<core::size, rank> indices) const noexcept {
            core::size total = 0;
            for (core::size index = 0; index < rank; ++index) {
                total += indices[index] * steps[index];
            }
            return total;
        }
    };

}
