#pragma once

#include <array/array.hpp>

#include <cmath>

// Array creation, named as in numpy. The shape is a braced list and the element type defaults to
// float; pass a Device to create the array there:
//
//     auto a = sci::zeros({2, 3});               // Array<float, 2>
//     auto b = sci::full<double>({4}, 2.5);
//     auto c = sci::ones({1024}, graphics);
//     auto d = sci::range(10);                   // 0, 1, ..., 9 (numpy.arange)

namespace sci::operations {

    using array::Device;
    using array::Array;
    using array::Shape;

    template <typename element = float, core::size rank>
    Array<element, rank> zeros(const core::size (&shape)[rank]) {
        return Array<element, rank>(Shape<rank>{std::to_array(shape)});
    }

    template <typename element = float, core::size rank>
    Array<element, rank> zeros(const core::size (&shape)[rank], Device& device) {
        return Array<element, rank>(device, Shape<rank>{std::to_array(shape)});
    }

    template <typename element = float, core::size rank>
    Array<element, rank> ones(const core::size (&shape)[rank]) {
        return Array<element, rank>(Shape<rank>{std::to_array(shape)}, element{1});
    }

    template <typename element = float, core::size rank>
    Array<element, rank> ones(const core::size (&shape)[rank], Device& device) {
        return Array<element, rank>(device, Shape<rank>{std::to_array(shape)}, element{1});
    }

    template <typename element = float, core::size rank>
    Array<element, rank> full(const core::size (&shape)[rank], const element value) {
        return Array<element, rank>(Shape<rank>{std::to_array(shape)}, value);
    }

    template <typename element = float, core::size rank>
    Array<element, rank> full(const core::size (&shape)[rank], const element value, Device& device) {
        return Array<element, rank>(device, Shape<rank>{std::to_array(shape)}, value);
    }

    // Contents are unspecified until written, which skips the zero-fill.
    template <typename element = float, core::size rank>
    Array<element, rank> empty(const core::size (&shape)[rank]) {
        return Array<element, rank>(Shape<rank>{std::to_array(shape)}, array::Uninitialized{});
    }

    template <typename element = float, core::size rank>
    Array<element, rank> empty(const core::size (&shape)[rank], Device& device) {
        return Array<element, rank>(device, Shape<rank>{std::to_array(shape)}, array::Uninitialized{});
    }

    // start, start + step, ... up to (not including) stop.
    template <typename element = float>
    Array<element, 1> range(const element start, const element stop, const element step = element{1}) {
        const double span = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(step));
        const core::size count = span > 0 ? static_cast<core::size>(span) : 0;
        Array<element, 1> result(Shape<1>{count}, array::Uninitialized{});
        element* values = result.data();
        for (core::size index = 0; index < count; ++index) {
            values[index] = static_cast<element>(start + static_cast<element>(index) * step);
        }
        return result;
    }

    template <typename element = float>
    Array<element, 1> range(const element stop) {
        return range<element>(element{0}, stop);
    }

}
