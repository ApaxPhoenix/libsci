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

    template <typename element, core::size rank>
    class NDArray;

    // Leaf: a temporary array moved into the expression, kept alive with it.
    template <typename type, core::size order>
    struct Owner : Methods<Owner<type, order>> {
        using element = type;
        static constexpr core::size rank = order;

        std::shared_ptr<const NDArray<type, order>> source;

        explicit Owner(std::shared_ptr<const NDArray<type, order>> source) noexcept : source(std::move(source)) {}

        [[nodiscard]] Shape<order> shape() const noexcept { return source->shape(); }
        [[nodiscard]] Device* device() const noexcept { return source->device(); }
        [[nodiscard]] core::uint8 record(core::Chunk& chunk) const noexcept { return chunk.input(source->data()); }
        [[nodiscard]] element at(const core::size index) const noexcept { return source->data()[index]; }
    };

    template <typename type, core::size order>
    struct Deferred<Owner<type, order>> : std::true_type {};

}
