#pragma once

#include "array/array.hpp"
#include "core/assert.hpp"
#include "core/types.hpp"
#include "linear/decompose.hpp"

#include <type_traits>
#include <vector>

namespace sci::linear {

    // Solves L U X = P^T B for the `width` columns of B, given factor()'s packed LU and row order:
    // forward substitution through the unit lower triangle, then back substitution through the upper.
    // Rows are updated as whole rows of `width` values, which keeps the inner loops contiguous.
    template <typename element>
    void substitute(const element* factors, const core::size* rows, const element* right, element* solution,
                    const core::size order, const core::size width) {
        for (core::size row = 0; row < order; ++row) {
            const element* source = right + rows[row] * width;
            element* target = solution + row * width;
            for (core::size column = 0; column < width; ++column) target[column] = source[column];
            for (core::size inner = 0; inner < row; ++inner) {
                const element scale = factors[row * order + inner];
                const element* known = solution + inner * width;
                for (core::size column = 0; column < width; ++column) target[column] -= scale * known[column];
            }
        }
        for (core::size row = order; row-- > 0;) {
            element* target = solution + row * width;
            for (core::size inner = row + 1; inner < order; ++inner) {
                const element scale = factors[row * order + inner];
                const element* known = solution + inner * width;
                for (core::size column = 0; column < width; ++column) target[column] -= scale * known[column];
            }
            const element diagonal = factors[row * order + row];
            for (core::size column = 0; column < width; ++column) target[column] /= diagonal;
        }
    }

    // x with matrix * x = vector (numpy.linalg.solve). The result lives where the matrix does.
    template <typename element>
    Array<element, 1> solve(const Array<element, 2>& matrix, const Array<element, 1>& vector) {
        static_assert(std::is_floating_point_v<element>, "solve needs a floating-point matrix");
        const core::size order = vector.shape()[0];
        ASSERT(matrix.shape()[0] == order && matrix.shape()[1] == order);

        Array<element, 2> work = matrix.host();
        const Array<element, 1> right = vector.host();
        std::vector<core::size> rows(order);
        factor(work.data(), rows.data(), order);

        Array<element, 1> solution(Shape<1>{order}, array::Uninitialized{});
        substitute(work.data(), rows.data(), right.data(), solution.data(), order, 1);
        return matrix.device() ? solution.to(*matrix.device()) : std::move(solution);
    }

    // X with matrix * X = right, for several right-hand sides at once (the columns of `right`).
    template <typename element>
    Array<element, 2> solve(const Array<element, 2>& matrix, const Array<element, 2>& right) {
        static_assert(std::is_floating_point_v<element>, "solve needs a floating-point matrix");
        const core::size order = right.shape()[0];
        const core::size width = right.shape()[1];
        ASSERT(matrix.shape()[0] == order && matrix.shape()[1] == order);

        Array<element, 2> work = matrix.host();
        const Array<element, 2> values = right.host();
        std::vector<core::size> rows(order);
        factor(work.data(), rows.data(), order);

        Array<element, 2> solution(Shape<2>{order, width}, array::Uninitialized{});
        substitute(work.data(), rows.data(), values.data(), solution.data(), order, width);
        return matrix.device() ? solution.to(*matrix.device()) : std::move(solution);
    }

}
