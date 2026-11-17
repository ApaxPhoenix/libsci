#pragma once

/// @file
/// @brief A scalar in an expression: a leaf of the tree.

#include "array/shape.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/drivers/device.hpp"
#include "core/types.hpp"

namespace sci::array {

    using core::drivers::Device;

    /// Leaf: a scalar, broadcast to every element.
    template <typename type> struct Literal {
        /// @brief The element type.
        using element = type;
        /// @brief A scalar has no dimensions.
        static constexpr core::size rank = 0;

        /// @brief The scalar.
        type value;

        /// @brief A scalar's shape: no dimensions.
        /// @return The shape.
        [[nodiscard]] static Shape<0> shape() noexcept { return {}; }
        /// @brief A scalar lives on no device.
        /// @return Null.
        [[nodiscard]] static Device *device() noexcept { return nullptr; }
        /// @brief Records the scalar as a constant.
        /// @param chunk The chunk recorded into.
        /// @return The constant's value id.
        [[nodiscard]] core::uint8 record(core::Chunk &chunk) const noexcept {
            return chunk.constant(static_cast<double>(value));
        }
        /// @brief The scalar, whatever the index.
        /// @return The scalar.
        [[nodiscard]] element at(core::size) const noexcept { return value; }
    };

}
