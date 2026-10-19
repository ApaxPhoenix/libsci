#pragma once

#include "array/array.hpp"
#include "core/assert.hpp"
#include "core/drivers/processor/dispatch.hpp"
#include "core/drivers/device.hpp"

namespace sci::linear {

    using array::Device;
    using array::Array;
    using array::Shape;

    // Matrix product of row-major matrices (numpy.matmul). GPU operands multiply on the GPU; host
    // float and double use the CPU kernels (Apple's BLAS on macOS); other element types a plain loop.
    template <typename element>
    Array<element, 2> product(const Array<element, 2>& left, const Array<element, 2>& right) {
        const core::size rows = left.shape()[0];
        const core::size inner = left.shape()[1];
        const core::size columns = right.shape()[1];

        ASSERT(inner == right.shape()[0]);

        Device* device = left.device() ? left.device() : right.device();
        Array<element, 2> output = device
            ? Array<element, 2>(*device, Shape<2>{rows, columns}, array::Uninitialized{})
            : Array<element, 2>(Shape<2>{rows, columns}, array::Uninitialized{});

        if constexpr (core::native<element>) {
            constexpr auto precision = core::Traits<element>::precision;
            if (device) {
                // The device orders its own work; operands on another device have to finish first.
                if (left.device() != right.device()) {
                    if (left.device()) left.device()->synchronize();
                    if (right.device()) right.device()->synchronize();
                }
                device->product(precision, left.data(), right.data(), output.data(), rows, columns, inner);
            } else {
                core::drivers::processor::Dispatch::product(precision, left.data(), right.data(), output.data(), rows, columns, inner);
            }
        } else {
            if (left.device()) left.device()->synchronize();
            if (right.device()) right.device()->synchronize();
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

    // Inner product of two vectors or vector expressions (numpy.dot on 1-D arrays), computed in a
    // single pass: the products are never stored.
    template <array::operand left, array::operand right>
    requires (left::rank == 1 && right::rank == 1)
    typename left::element dot(const left& first, const right& second) {
        return array::reduce(first * second, core::Fold::sum);
    }

}
