#pragma once

#include "array/deferred.hpp"
#include "array/methods.hpp"
#include "array/shape.hpp"
#include "core/drivers/device.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/types.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace sci::array {

    using core::drivers::Device;

    // Leaf: an array the expression reads. `pointer` is a plain pointer to an array that outlives the
    // expression, or a shared pointer to a temporary moved into it, which it keeps alive.
    template <typename pointer>
    struct Reference : Methods<Reference<pointer>> {
        using element = std::pointer_traits<pointer>::element_type::element;
        static constexpr core::size rank = std::pointer_traits<pointer>::element_type::rank;

        pointer source;

        explicit Reference(pointer source) noexcept : source(std::move(source)) {}

        [[nodiscard]] Shape<rank> shape() const noexcept { return source->shape(); }
        [[nodiscard]] Device* device() const noexcept { return source->device(); }
        [[nodiscard]] core::uint8 record(core::Chunk& chunk) const noexcept { return chunk.input(source->data()); }
        [[nodiscard]] element at(const core::size index) const noexcept { return source->data()[index]; }
    };

    template <typename pointer>
    struct Deferred<Reference<pointer>> : std::true_type {};

}
