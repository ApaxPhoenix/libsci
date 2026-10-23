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

    // x with matrix * x = right (numpy.linalg.solve). `right` is a vector, or a matrix whose columns
    // are several right-hand sides solved at once. The result has right's shape and lives where the
    // matrix does.
    template <typename element, core::size rank>
    requires (rank == 1 || rank == 2)
    Array<element, rank> solve(const Array<element, 2>& matrix, const Array<element, rank>& right) {
        static_assert(std::is_floating_point_v<element>, "solve needs a floating-point matrix");
        const core::size order = right.shape()[0];
        const core::size width = rank == 1 ? 1 : right.shape()[rank - 1];
        ASSERT(matrix.shape()[0] == order && matrix.shape()[1] == order);

        Array<element, 2> work = matrix.host();
        const Array<element, rank> values = right.host();
        std::vector<core::size> rows(order);
        factor(work.data(), rows.data(), order);

        Array<element, rank> solution(right.shape(), array::Uninitialized{});
        substitute(work.data(), rows.data(), values.data(), solution.data(), order, width);
        return matrix.device() ? solution.to(*matrix.device()) : std::move(solution);
    }

}
