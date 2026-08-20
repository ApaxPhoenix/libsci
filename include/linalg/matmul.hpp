#pragma once

#include "array/ndarray.hpp"
#include "core/assert.hpp"
#include "core/drivers/device.hpp"

namespace sci::linalg {

    template <typename element>
    array::ndarray<element, 2> matmul(const array::ndarray<element, 2>& left, const array::ndarray<element, 2>& right) {
        const core::size rows = left.extents()[0];
        const core::size inner = left.extents()[1];
        const core::size columns = right.extents()[1];

        ASSERT(inner == right.extents()[0]);

        if (left.handle().target == core::drivers::target::gpu) {
            core::drivers::device* device = left.device();
            ASSERT(device != nullptr);

            array::ndarray<element, 2> output(*device, array::shape<2>{rows, columns});

            const core::drivers::memory bindings[] = {
                left.handle(),
                right.handle(),
                output.handle()
            };

            constexpr std::span<const core::uint32> bytecode = {};
            const auto width = static_cast<core::uint32>((columns + 15) / 16);
            const auto height = static_cast<core::uint32>((rows + 15) / 16);

            device->compose(bytecode, bindings, width, height);
            device->sync();

            return output;
        }

        array::ndarray<element, 2> output(array::shape<2>{rows, columns});

        const element* first = left.data();
        const element* second = right.data();
        element* target = output.data();

        for (core::size index = 0; index < output.size(); ++index) {
            target[index] = static_cast<element>(0);
        }

        for (core::size row = 0; row < rows; ++row) {
            for (core::size middle = 0; middle < inner; ++middle) {
                for (core::size column = 0; column < columns; ++column) {
                    target[row * columns + column] +=
                        first[row * inner + middle] * second[middle * columns + column];
                }
            }
        }

        return output;
    }

}