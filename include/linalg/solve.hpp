#pragma once

#include "array/ndarray.hpp"
#include "core/assert.hpp"
#include "core/types.hpp"
#include "linalg/decompose.hpp"

#include <type_traits>
#include <vector>

namespace sci::linalg {

    // Solves L U X = P^T B for the k columns of B, given factor()'s packed LU and row order:
    // forward substitution through the unit lower triangle, then back substitution through the upper.
    // Rows are updated as whole rows of k values, which keeps the inner loops contiguous.
    template <typename element>
    void substitute(const element* factors, const core::size* rows, const element* right, element* solution,
                    const core::size n, const core::size k) {
        for (core::size row = 0; row < n; ++row) {
            const element* source = right + rows[row] * k;
            element* target = solution + row * k;
            for (core::size column = 0; column < k; ++column) target[column] = source[column];
            for (core::size inner = 0; inner < row; ++inner) {
                const element scale = factors[row * n + inner];
                const element* known = solution + inner * k;
                for (core::size column = 0; column < k; ++column) target[column] -= scale * known[column];
            }
        }
        for (core::size row = n; row-- > 0;) {
            element* target = solution + row * k;
            for (core::size inner = row + 1; inner < n; ++inner) {
                const element scale = factors[row * n + inner];
                const element* known = solution + inner * k;
                for (core::size column = 0; column < k; ++column) target[column] -= scale * known[column];
            }
            const element diagonal = factors[row * n + row];
            for (core::size column = 0; column < k; ++column) target[column] /= diagonal;
        }
    }

    // x with matrix * x = vector (numpy.linalg.solve). The result lives where the matrix does.
    template <typename element>
    NDArray<element, 1> solve(const NDArray<element, 2>& matrix, const NDArray<element, 1>& vector) {
        static_assert(std::is_floating_point_v<element>, "solve needs a floating-point matrix");
        const core::size n = vector.shape()[0];
        ASSERT(matrix.shape()[0] == n && matrix.shape()[1] == n);

        NDArray<element, 2> work = matrix.cpu();
        const NDArray<element, 1> right = vector.cpu();
        std::vector<core::size> rows(n);
        factor(work.data(), rows.data(), n);

        NDArray<element, 1> solution(Shape<1>{n}, array::Uninitialized{});
        substitute(work.data(), rows.data(), right.data(), solution.data(), n, 1);
        return matrix.device() ? solution.to(*matrix.device()) : std::move(solution);
    }

    // X with matrix * X = right, for several right-hand sides at once (the columns of `right`).
    template <typename element>
    NDArray<element, 2> solve(const NDArray<element, 2>& matrix, const NDArray<element, 2>& right) {
        static_assert(std::is_floating_point_v<element>, "solve needs a floating-point matrix");
        const core::size n = right.shape()[0];
        const core::size k = right.shape()[1];
        ASSERT(matrix.shape()[0] == n && matrix.shape()[1] == n);

        NDArray<element, 2> work = matrix.cpu();
        const NDArray<element, 2> values = right.cpu();
        std::vector<core::size> rows(n);
        factor(work.data(), rows.data(), n);

        NDArray<element, 2> solution(Shape<2>{n, k}, array::Uninitialized{});
        substitute(work.data(), rows.data(), values.data(), solution.data(), n, k);
        return matrix.device() ? solution.to(*matrix.device()) : std::move(solution);
    }

}
