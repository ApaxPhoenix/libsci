#pragma once

#include <array/array.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace sci::operations {

    using array::Array;
    using array::Shape;

    template <typename element, core::size input, core::size output>
    Array<element, output> reshape(const Array<element, input>& alpha, const core::size (&shape)[output]) {
        const Shape<output> bounds{std::to_array(shape)};
        if (bounds.total() != alpha.size()) {
            throw std::invalid_argument("reshape must preserve the element count");
        }

        Array<element, output> beta(bounds, array::Uninitialized{});
        std::copy(alpha.begin(), alpha.end(), beta.begin());
        return beta;
    }

    template <typename element, core::size rank>
    Array<element, rank> transpose(const Array<element, rank>& input) {
        const auto shape = input.shape();
        Shape<rank> bounds{};
        for (core::size axis = 0; axis < rank; ++axis) {
            bounds.extents[rank - 1 - axis] = shape[axis];
        }

        Array<element, rank> output(bounds, array::Uninitialized{});
        const auto* source = input.begin();
        auto* target = output.data();
        const auto count = input.size();

        for (core::size index = 0; index < count; ++index) {
            core::size remaining = index;
            core::size destination = 0;

            for (core::size axis = 0; axis < rank; ++axis) {
                const auto coordinate = remaining / input.strides()[axis];
                remaining %= input.strides()[axis];
                destination += coordinate * output.strides()[rank - 1 - axis];
            }

            target[destination] = source[index];
        }

        return output;
    }

    template <typename element, core::size rank>
    Array<element, 1> flatten(const Array<element, rank>& input) {
        Array<element, 1> output(Shape<1>{input.size()}, array::Uninitialized{});
        std::copy(input.begin(), input.end(), output.begin());

        return output;
    }

    template <typename element, core::size rank>
    requires (rank > 1)
    Array<element, rank - 1> squeeze(const Array<element, rank>& input, core::size axis) {
        const auto shape = input.shape();
        if (axis >= rank || shape[axis] != 1) {
            throw std::invalid_argument("squeeze requires an axis of length 1");
        }

        core::size bounds[rank - 1]{};
        core::size destination = 0;
        for (core::size source = 0; source < rank; ++source) {
            if (source != axis) {
                bounds[destination++] = shape[source];
            }
        }

        return reshape(input, bounds);
    }

    template <typename element, core::size rank>
    Array<element, rank + 1> expand(const Array<element, rank>& input, core::size axis) {
        if (axis > rank) {
            throw std::invalid_argument("expand axis is out of range");
        }

        const auto shape = input.shape();
        core::size bounds[rank + 1]{};
        const auto first = shape.extents.begin();
        const auto split = first + axis;
        std::copy(first, split, bounds);
        bounds[axis] = 1;
        std::copy(split, shape.extents.end(), bounds + axis + 1);

        return reshape(input, bounds);
    }

}
