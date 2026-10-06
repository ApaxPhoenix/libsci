#pragma once

#include <array/ndarray.hpp>

// Whole-array reductions, named as in numpy. Reducing an expression fuses it into the reduction,
// so sum(a * b) never materialises a * b.

namespace sci::ops {

    using array::operand;

    template <operand source>
    array::element_t<source> sum(const source& input) {
        return array::reduce(input, core::Fold::sum);
    }

    template <operand source>
    array::element_t<source> mean(const source& input) {
        const core::size count = array::wrap(input).shape().total();
        return count == 0 ? array::element_t<source>{} : sum(input) / static_cast<array::element_t<source>>(count);
    }

    // NaN anywhere gives NaN, as in numpy.min / numpy.max.
    template <operand source>
    array::element_t<source> min(const source& input) {
        return array::reduce(input, core::Fold::min);
    }

    template <operand source>
    array::element_t<source> max(const source& input) {
        return array::reduce(input, core::Fold::max);
    }

}
