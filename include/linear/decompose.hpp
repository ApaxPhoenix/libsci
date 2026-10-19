#pragma once

#include "array/array.hpp"
#include "core/assert.hpp"
#include "core/drivers/processor/pool.hpp"
#include "core/types.hpp"
#include "linear/factors.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

namespace sci::linear {

    using array::Array;
    using array::Shape;

    // In-place LU with partial pivoting of a row-major order x order matrix. Afterwards the strict lower
    // triangle holds L (its unit diagonal implied), the rest holds U, and rows[i] is the original
    // row now at position i. A zero pivot (a singular matrix) is skipped; solving then gives inf/NaN.
    template <typename element>
    void factor(element* matrix, core::size* rows, const core::size order) {
        for (core::size index = 0; index < order; ++index) {
            rows[index] = index;
        }

        for (core::size step = 0; step < order; ++step) {
            core::size pivot = step;
            for (core::size row = step + 1; row < order; ++row) {
                if (std::abs(matrix[row * order + step]) > std::abs(matrix[pivot * order + step])) pivot = row;
            }
            if (pivot != step) {
                std::swap_ranges(matrix + step * order, matrix + (step + 1) * order, matrix + pivot * order);
                std::swap(rows[step], rows[pivot]);
            }

            const element diagonal = matrix[step * order + step];
            if (diagonal == element{0}) continue;

            // Every row below subtracts a multiple of the pivot row: contiguous, so it vectorises,
            // and independent, so large matrices spread the rows across the pool.
            const element* top = matrix + step * order;
            auto eliminate = [&](const core::size row) {
                element* target = matrix + row * order;
                const element scale = target[step] /= diagonal;
                for (core::size column = step + 1; column < order; ++column) {
                    target[column] -= scale * top[column];
                }
            };

            if (const core::size below = order - step - 1; below * (order - step) >= (core::size{1} << 16)) {
                auto& pool = core::drivers::processor::Pool::global();
                const core::size parts = std::min(below, pool.width() * 4);
                pool.run(parts, [&](const core::size part) {
                    const core::size first = step + 1 + below * part / parts;
                    const core::size last = step + 1 + below * (part + 1) / parts;
                    for (core::size row = first; row < last; ++row) eliminate(row);
                });
            } else {
                for (core::size row = step + 1; row < order; ++row) eliminate(row);
            }
        }
    }

    // LU decomposition with partial pivoting (scipy.linalg.lu). The factors live where the matrix does.
    template <typename element>
    Factors<element> decompose(const Array<element, 2>& matrix) {
        static_assert(std::is_floating_point_v<element>, "decompose needs a floating-point matrix");
        const core::size order = matrix.shape()[0];
        ASSERT(order == matrix.shape()[1]);

        Array<element, 2> work = matrix.host();
        std::vector<core::size> rows(order);
        factor(work.data(), rows.data(), order);

        Array<element, 2> permutation(Shape<2>{order, order});
        Array<element, 2> lower(Shape<2>{order, order});
        Array<element, 2> upper(Shape<2>{order, order});
        const element* packed = work.data();
        for (core::size row = 0; row < order; ++row) {
            for (core::size column = 0; column < order; ++column) {
                const element value = packed[row * order + column];
                (column < row ? lower : upper).data()[row * order + column] = value;
            }
            lower.data()[row * order + row] = element{1};
            permutation.data()[rows[row] * order + row] = element{1};
        }

        if (array::Device* device = matrix.device()) {
            return {.permutation = permutation.to(*device), .lower = lower.to(*device), .upper = upper.to(*device)};
        }
        return {.permutation = std::move(permutation), .lower = std::move(lower), .upper = std::move(upper)};
    }

}
