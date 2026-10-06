#pragma once

#include "array/deferred.hpp"
#include "array/methods.hpp"
#include "array/shape.hpp"
#include "core/drivers/device.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/types.hpp"

#include <type_traits>

namespace sci::array {

    using core::drivers::Device;

    template <typename element, core::size rank>
    class NDArray;

    // Leaf: an array the expression reads.
    template <typename type, core::size order>
    struct Reference : Methods<Reference<type, order>> {
        using element = type;
        static constexpr core::size rank = order;

        const NDArray<type, order>* source;

        explicit Reference(const NDArray<type, order>* source) noexcept : source(source) {}

        [[nodiscard]] Shape<order> shape() const noexcept { return source->shape(); }
        [[nodiscard]] Device* device() const noexcept { return source->device(); }
        [[nodiscard]] core::uint8 record(core::Chunk& chunk) const noexcept { return chunk.input(source->data()); }
        [[nodiscard]] element at(const core::size index) const noexcept { return source->data()[index]; }
    };

    template <typename type, core::size order>
    struct Deferred<Reference<type, order>> : std::true_type {};

}
