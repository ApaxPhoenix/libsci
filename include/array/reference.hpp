#pragma once

/// @file
/// @brief An array in an expression: a leaf of the tree.

#include "array/deferred.hpp"
#include "array/methods.hpp"
#include "array/shape.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/drivers/device.hpp"
#include "core/types.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace sci::array {

    using core::drivers::Device;

    /// Leaf: an array the expression reads. `pointer` is a plain pointer to an array that outlives the
    /// expression, or a shared pointer to a temporary moved into it, which it keeps alive.
    template <typename pointer> struct Reference : Methods<Reference<pointer>> {
        /// @brief The array's element type.
        using element = std::pointer_traits<pointer>::element_type::element;
        /// @brief The array's number of dimensions.
        static constexpr core::size rank = std::pointer_traits<pointer>::element_type::rank;

        /// @brief The array.
        pointer source;

        /// @brief A leaf over an array.
        /// @param source The array, by plain or shared pointer.
        explicit Reference(pointer source) noexcept : source(std::move(source)) {}

        /// @brief The array's shape.
        /// @return The shape.
        [[nodiscard]] Shape<rank> shape() const noexcept { return source->shape(); }
        /// @brief The device the array lives on.
        /// @return The device, or null.
        [[nodiscard]] Device *device() const noexcept { return source->device(); }
        /// @brief Records the array as an input.
        /// @param chunk The chunk recorded into.
        /// @return The input's value id.
        [[nodiscard]] core::uint8 record(core::Chunk &chunk) const noexcept { return chunk.input(source->data()); }
        /// @brief One of the array's elements.
        /// @param index The element's flat index.
        /// @return The element.
        [[nodiscard]] element at(const core::size index) const noexcept { return source->data()[index]; }
    };

    /// @brief Every reference is a lazy expression.
    template <typename pointer> struct Deferred<Reference<pointer>> : std::true_type {};

}
