#pragma once

#include "array/ndarray.hpp"
#include "core/assert.hpp"
#include "core/drivers/cpu/pool.hpp"
#include "core/types.hpp"
#include "linalg/factors.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

namespace sci::linalg {

    using array::NDArray;
    using array::Shape;

    // In-place LU with partial pivoting of a row-major n x n matrix. Afterwards the strict lower
    // triangle holds L (its unit diagonal implied), the rest holds U, and rows[i] is the original
    // row now at position i. A zero pivot (a singular matrix) is skipped; solving then gives inf/NaN.
    template <typename element>
    void factor(element* matrix, core::size* rows, const core::size n) {
        for (core::size index = 0; index < n; ++index) {
            rows[index] = index;
        }

        for (core::size step = 0; step < n; ++step) {
            core::size pivot = step;
            for (core::size row = step + 1; row < n; ++row) {
                if (std::abs(matrix[row * n + step]) > std::abs(matrix[pivot * n + step])) pivot = row;
            }
            if (pivot != step) {
                std::swap_ranges(matrix + step * n, matrix + (step + 1) * n, matrix + pivot * n);
                std::swap(rows[step], rows[pivot]);
            }

            const element diagonal = matrix[step * n + step];
            if (diagonal == element{0}) continue;

            // Every row below subtracts a multiple of the pivot row: contiguous, so it vectorises,
            // and independent, so large matrices spread the rows across the pool.
            const element* top = matrix + step * n;
            auto eliminate = [&](const core::size row) {
                element* target = matrix + row * n;
                const element scale = target[step] /= diagonal;
                for (core::size column = step + 1; column < n; ++column) {
                    target[column] -= scale * top[column];
                }
            };

            const core::size below = n - step - 1;
            if (below * (n - step) >= (core::size{1} << 16)) {
                auto& pool = core::drivers::cpu::Pool::global();
                const core::size parts = std::min(below, pool.width() * 4);
                pool.run(parts, [&](const core::size part) {
                    const core::size first = step + 1 + below * part / parts;
                    const core::size last = step + 1 + below * (part + 1) / parts;
                    for (core::size row = first; row < last; ++row) eliminate(row);
                });
            } else {
                for (core::size row = step + 1; row < n; ++row) eliminate(row);
            }
        }
    }

    // LU decomposition with partial pivoting (scipy.linalg.lu). The factors live where the matrix does.
    template <typename element>
    Factors<element> lu(const NDArray<element, 2>& matrix) {
        static_assert(std::is_floating_point_v<element>, "lu needs a floating-point matrix");
        const core::size n = matrix.shape()[0];
        ASSERT(n == matrix.shape()[1]);

        NDArray<element, 2> work = matrix.cpu();
        std::vector<core::size> rows(n);
        factor(work.data(), rows.data(), n);

        NDArray<element, 2> permutation(Shape<2>{n, n});
        NDArray<element, 2> lower(Shape<2>{n, n});
        NDArray<element, 2> upper(Shape<2>{n, n});
        const element* packed = work.data();
        for (core::size row = 0; row < n; ++row) {
            for (core::size column = 0; column < n; ++column) {
                const element value = packed[row * n + column];
                (column < row ? lower : upper).data()[row * n + column] = value;
            }
            lower.data()[row * n + row] = element{1};
            permutation.data()[rows[row] * n + row] = element{1};
        }

        if (array::Device* device = matrix.device()) {
            return {.permutation = permutation.to(*device), .lower = lower.to(*device), .upper = upper.to(*device)};
        }
        return {.permutation = std::move(permutation), .lower = std::move(lower), .upper = std::move(upper)};
    }

}
