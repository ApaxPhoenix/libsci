#pragma once

#include <array/array.hpp>

// Whole-array reductions, named as in numpy (numpy.min and numpy.max become minimum and maximum
// with one operand). Reducing an expression fuses it into the reduction, so sum(a * b) never
// materialises a * b.

namespace sci::operations {

    using array::operand;

    template <operand source>
    source::element sum(const source& input) {
        return array::reduce(input, core::Fold::sum);
    }

    template <operand source>
    source::element mean(const source& input) {
        using element = source::element;
        const core::size count = input.shape().total();
        return count == 0 ? element{} : sum(input) / static_cast<element>(count);
    }

    // NaN anywhere gives NaN, as in numpy.min / numpy.max.
    template <operand source>
    source::element minimum(const source& input) {
        return array::reduce(input, core::Fold::minimum);
    }

    template <operand source>
    source::element maximum(const source& input) {
        return array::reduce(input, core::Fold::maximum);
    }

}
