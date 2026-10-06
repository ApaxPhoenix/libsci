#pragma once

#include "array/ndarray.hpp"
#include "core/assert.hpp"
#include "core/drivers/cpu/dispatch.hpp"
#include "core/drivers/device.hpp"

namespace sci::linalg {

    using array::Device;
    using array::NDArray;
    using array::Shape;

    // Matrix product of row-major matrices (numpy.matmul). GPU operands multiply on the GPU; host
    // float and double use the CPU kernels (Apple's BLAS on macOS); other element types a plain loop.
    template <typename element>
    NDArray<element, 2> matmul(const NDArray<element, 2>& left, const NDArray<element, 2>& right) {
        const core::size rows = left.shape()[0];
        const core::size inner = left.shape()[1];
        const core::size columns = right.shape()[1];

        ASSERT(inner == right.shape()[0]);

        Device* device = left.device() ? left.device() : right.device();
        NDArray<element, 2> output = device
            ? NDArray<element, 2>(*device, Shape<2>{rows, columns}, array::Uninitialized{})
            : NDArray<element, 2>(Shape<2>{rows, columns}, array::Uninitialized{});

        if constexpr (core::native<element>) {
            constexpr auto type = core::Traits<element>::type;
            if (device) {
                if (left.device() != right.device()) {
                    array::synchronize(left.device());
                    array::synchronize(right.device());
                }
                device->matmul(type, left.data(), right.data(), output.data(), rows, columns, inner);
            } else {
                core::drivers::cpu::Dispatch::matmul(type, left.data(), right.data(), output.data(), rows, columns, inner);
            }
        } else {
            array::synchronize(left.device());
            array::synchronize(right.device());
            const element* first = left.data();
            const element* second = right.data();
            element* target = output.data();
            for (core::size row = 0; row < rows; ++row) {
                for (core::size column = 0; column < columns; ++column) {
                    element sum{};
                    for (core::size middle = 0; middle < inner; ++middle) {
                        sum += first[row * inner + middle] * second[middle * columns + column];
                    }
                    target[row * columns + column] = sum;
                }
            }
        }

        return output;
    }

    // Inner product of two vectors or vector expressions (numpy.dot on 1-D arrays), computed in a single
    // pass: the products are never stored.
    template <array::operand left, array::operand right>
    requires (array::rank_v<left> == 1 && array::rank_v<right> == 1)
    array::element_t<left> dot(const left& first, const right& second) {
        return array::reduce(first * second, core::Fold::sum);
    }

}
