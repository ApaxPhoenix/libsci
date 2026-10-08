#pragma once

#include <array/ndarray.hpp>

#include <cmath>

// Array creation, named as in numpy. The shape is a braced list and the element type defaults to
// float; pass a Device to create the array there:
//
//     auto a = sci::zeros({2, 3});               // NDArray<float, 2>
//     auto b = sci::full<double>({4}, 2.5);
//     auto c = sci::ones({1024}, gpu);
//     auto d = sci::arange(10);                  // 0, 1, ..., 9

namespace sci::operations {

    using array::Device;
    using array::NDArray;
    using array::Shape;

    template <typename element = float, core::size rank>
    NDArray<element, rank> zeros(const core::size (&shape)[rank]) {
        return NDArray<element, rank>(Shape<rank>{std::to_array(shape)});
    }

    template <typename element = float, core::size rank>
    NDArray<element, rank> zeros(const core::size (&shape)[rank], Device& device) {
        return NDArray<element, rank>(device, Shape<rank>{std::to_array(shape)});
    }

    template <typename element = float, core::size rank>
    NDArray<element, rank> ones(const core::size (&shape)[rank]) {
        return NDArray<element, rank>(Shape<rank>{std::to_array(shape)}, element{1});
    }

    template <typename element = float, core::size rank>
    NDArray<element, rank> ones(const core::size (&shape)[rank], Device& device) {
        return NDArray<element, rank>(device, Shape<rank>{std::to_array(shape)}, element{1});
    }

    template <typename element = float, core::size rank>
    NDArray<element, rank> full(const core::size (&shape)[rank], const element value) {
        return NDArray<element, rank>(Shape<rank>{std::to_array(shape)}, value);
    }

    template <typename element = float, core::size rank>
    NDArray<element, rank> full(const core::size (&shape)[rank], const element value, Device& device) {
        return NDArray<element, rank>(device, Shape<rank>{std::to_array(shape)}, value);
    }

    // Contents are unspecified until written, which skips the zero-fill.
    template <typename element = float, core::size rank>
    NDArray<element, rank> empty(const core::size (&shape)[rank]) {
        return NDArray<element, rank>(Shape<rank>{std::to_array(shape)}, array::Uninitialized{});
    }

    template <typename element = float, core::size rank>
    NDArray<element, rank> empty(const core::size (&shape)[rank], Device& device) {
        return NDArray<element, rank>(device, Shape<rank>{std::to_array(shape)}, array::Uninitialized{});
    }

    // start, start + step, ... up to (not including) stop.
    template <typename element = float>
    NDArray<element, 1> arange(const element start, const element stop, const element step = element{1}) {
        const double span = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(step));
        const core::size count = span > 0 ? static_cast<core::size>(span) : 0;
        NDArray<element, 1> result(Shape<1>{count}, array::Uninitialized{});
        element* values = result.data();
        for (core::size index = 0; index < count; ++index) {
            values[index] = static_cast<element>(start + static_cast<element>(index) * step);
        }
        return result;
    }

    template <typename element = float>
    NDArray<element, 1> arange(const element stop) {
        return arange<element>(element{0}, stop);
    }

}
