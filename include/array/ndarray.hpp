#pragma once

#include <array/shape.hpp>
#include <array/storage.hpp>
#include <array/stride.hpp>
#include <core/types.hpp>

namespace sci::array {

    template <typename element, core::size rank>
    class ndarray {
        storage<element> store{};
        shape<rank> bounds{};
        stride<rank> layout{};

    public:
        ndarray() = default;

        explicit ndarray(shape<rank> extents)
            : store(extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {}

        element* data() noexcept { return store.data(); }
        const element* data() const noexcept { return store.data(); }
        shape<rank> extents() const noexcept { return bounds; }
        stride<rank> strides() const noexcept { return layout; }
        [[nodiscard]] core::size size() const noexcept { return store.size(); }
    };

}