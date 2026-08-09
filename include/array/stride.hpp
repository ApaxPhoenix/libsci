#pragma once

#include <array>
#include <array/shape.hpp>
#include <core/types.hpp>

namespace sci::array {

    template <core::size rank>
    struct stride {
        std::array<core::size, rank> steps{};

        static constexpr stride contiguous(shape<rank> extents) noexcept {
            stride result{};
            core::size step = 1;
            for (core::size index = rank; index > 0; --index) {
                result.steps[index - 1] = step;
                step *= extents[index - 1];
            }
            return result;
        }

        constexpr core::size offset(std::array<core::size, rank> indices) const noexcept {
            core::size total = 0;
            for (core::size index = 0; index < rank; ++index) {
                total += indices[index] * steps[index];
            }
            return total;
        }
    };

}