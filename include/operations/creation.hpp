#pragma once

#include <array/array.hpp>

#include <cmath>

// Array creation, named as in numpy. The shape is a braced list and the element type defaults to
// float; pass a Device (or an Arena) after the other arguments to create the array there:
//
//     auto alpha = sci::zeros({2, 3});           // Array<float, 2>
//     auto beta  = sci::full<double>({4}, 2.5);
//     auto gamma = sci::linear(0.0f, 1.0f, 5);
//     auto delta = sci::eye(3);
//     auto zeta  = sci::grid(gamma, delta);
//     auto omega = sci::ones({1024}, graphics);
//     auto sigma = sci::range(10);               // 0, 1, ..., 9 (numpy.arange)

namespace sci::operations {

    using array::Array;
    using array::Shape;
    using array::location;

    template <typename element = float, core::size rank, location... place>
    requires (sizeof...(place) <= 1)
    Array<element, rank> zeros(const core::size (&shape)[rank], place&... where) {
        return Array<element, rank>(where..., Shape<rank>{std::to_array(shape)});
    }

    template <typename element = float, core::size rank, location... place>
    requires (sizeof...(place) <= 1)
    Array<element, rank> ones(const core::size (&shape)[rank], place&... where) {
        return Array<element, rank>(where..., Shape<rank>{std::to_array(shape)}, element{1});
    }

    template <typename element = float, core::size rank, location... place>
    requires (sizeof...(place) <= 1)
    Array<element, rank> full(const core::size (&shape)[rank], const element value, place&... where) {
        return Array<element, rank>(where..., Shape<rank>{std::to_array(shape)}, value);
    }

    template <typename element = float, location... place>
    requires (sizeof...(place) <= 1)
    Array<element, 1> linspace(const element start, const element stop, const core::size count, place&... where) {
        Array<element, 1> result(where..., Shape<1>{count}, array::Uninitialized{});
        if (count == 0) {
            return result;
        }

        element* values = result.data();
        if (count == 1) {
            values[0] = start;
            return result;
        }

        const double step = static_cast<double>(stop - start) / (static_cast<double>(count) - 1.0);
        for (core::size index = 0; index < count; ++index) {
            values[index] = static_cast<element>(static_cast<double>(start) + static_cast<double>(index) * step);
        }
        return result;
    }

    template <typename element = float, core::size rank, location... place>
     requires (sizeof...(place) <= 1 && rank == 2)
     Array<element, rank> eye(const core::size (&shape)[rank], place&... where) {
        Array<element, rank> result(where..., Shape<rank>{std::to_array(shape)}, element{0});

        const core::size rows = shape[0];
        const core::size columns = shape[1];
        element* values = result.data();

        const core::size limit = rows < columns ? rows : columns;
        for (core::size index = 0; index < limit; ++index) {
            values[index * columns + index] = element{1};
        }

        return result;
    }

    template <typename element = float, location... place>
    requires (sizeof...(place) <= 1)
    std::pair<Array<element, 2>, Array<element, 2>> grid(const Array<element, 1>& x, const Array<element, 1>& y, place&... where) {
        const core::size nx = x.shape()[0];
        const core::size ny = y.shape()[0];

        Array<element, 2> xx(where..., Shape<2>{ny, nx}, array::Uninitialized{});
        Array<element, 2> yy(where..., Shape<2>{ny, nx}, array::Uninitialized{});

        for (core::size i = 0; i < ny; ++i) {
            for (core::size j = 0; j < nx; ++j) {
                xx(i, j) = x[j];
                yy(i, j) = y[i];
            }
        }

        return {xx, yy};
    }

    // Contents are unspecified until written, which skips the zero-fill.
    template <typename element = float, core::size rank, location... place>
    requires (sizeof...(place) <= 1)
    Array<element, rank> empty(const core::size (&shape)[rank], place&... where) {
        return Array<element, rank>(where..., Shape<rank>{std::to_array(shape)}, array::Uninitialized{});
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
