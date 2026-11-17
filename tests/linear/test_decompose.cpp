#include <sci.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace tests::linear {
    bool close(const sci::Array<double, 2> &first, const sci::Array<double, 2> &second) {
        if (!(first.shape() == second.shape())) return false;
        for (std::size_t index = 0; index < first.shape().total(); ++index) {
            if (std::abs(first.data()[index] - second.data()[index]) > 1e-9) return false;
        }
        return true;
    }
}

int main() {
    using namespace tests::linear;

    // A matrix whose first pivot is zero, so partial pivoting must exchange rows.
    const sci::Array<double, 2> matrix(sci::Shape<2>{3, 3}, {0, 2, 1, 4, 1, 3, 2, 5, 7});
    const auto [permutation, lower, upper] = sci::linear::decompose(matrix);

    for (std::size_t row = 0; row < 3; ++row) {
        assert((lower[row, row] == 1.0) && "the lower factor has ones on its diagonal");
        for (std::size_t column = 0; column < 3; ++column) {
            if (column > row) assert((lower[row, column] == 0.0) && "and nothing above it");
            if (column < row) assert((upper[row, column] == 0.0) && "and the upper factor nothing below its own");
        }
    }

    // Whichever side the permutation is written for, the factors give the matrix back.
    const sci::Array<double, 2> factored = sci::product(lower, upper);
    const bool left = close(sci::product(permutation, matrix), factored);
    const bool right = close(matrix, sci::product(permutation, factored));
    assert((left || right) && "the permuted matrix is the product of its factors");
    return 0;
}
