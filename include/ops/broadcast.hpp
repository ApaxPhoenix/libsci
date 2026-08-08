#pragma once

#include <algorithm>
#include <array/shape.hpp>

namespace sci::ops {

    template <core::size rank>
    array::shape<rank> broadcast(const array::shape<rank>& left, const array::shape<rank>& right) {
        array::shape<rank> output{};
        for (core::size index = 0; index < rank; ++index) {
            output[index] = std::max(left[index], right[index]);
        }
        return output;
    }

}