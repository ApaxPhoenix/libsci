#include <sci.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

int main() {
    const sci::Array<double, 2> matrix(sci::Shape<2>{3, 3}, {4, -2, 1, -2, 4, -2, 1, -2, 4});

    // One right-hand side: the solution times the matrix is the right-hand side again.
    const sci::Array<double, 1> right(sci::Shape<1>{3}, {11, -16, 17});
    const sci::Array<double, 1> solution = sci::linear::solve(matrix, right);
    for (std::size_t row = 0; row < 3; ++row) {
        double total = 0.0;
        for (std::size_t column = 0; column < 3; ++column)
            total += matrix[row, column] * solution[column];
        assert((std::abs(total - right[row]) < 1e-9) && "a vector is solved for");
    }

    // Several at once, as the columns of a matrix.
    const sci::Array<double, 2> several(sci::Shape<2>{3, 2}, {11, 1, -16, 0, 17, 2});
    const sci::Array<double, 2> solutions = sci::linear::solve(matrix, several);
    const sci::Array<double, 2> again = sci::product(matrix, solutions);
    for (std::size_t index = 0; index < 6; ++index)
        assert((std::abs(again.data()[index] - several.data()[index]) < 1e-9) && "and every column of a matrix");
    return 0;
}
