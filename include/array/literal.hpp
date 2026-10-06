#pragma once

#include "array/shape.hpp"
#include "core/drivers/device.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/types.hpp"

namespace sci::array {

    using core::drivers::Device;

    // Leaf: a scalar, broadcast to every element.
    template <typename type>
    struct Literal {
        using element = type;
        static constexpr core::size rank = 0;

        type value;

        [[nodiscard]] static Shape<0> shape() noexcept { return {}; }
        [[nodiscard]] static Device* device() noexcept { return nullptr; }
        [[nodiscard]] core::uint8 record(core::Chunk& chunk) const noexcept { return chunk.constant(static_cast<double>(value)); }
        [[nodiscard]] element at(core::size) const noexcept { return value; }
    };

}
