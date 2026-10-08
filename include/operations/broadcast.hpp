#pragma once

#include <algorithm>
#include <array/shape.hpp>

namespace sci::operations {

    // Shape two same-rank shapes broadcast to: the larger extent in each dimension.
    template <core::size rank>
    array::Shape<rank> broadcast(const array::Shape<rank>& left, const array::Shape<rank>& right) {
        array::Shape<rank> output{};
        for (core::size index = 0; index < rank; ++index) {
            output[index] = std::max(left[index], right[index]);
        }
        return output;
    }

}
