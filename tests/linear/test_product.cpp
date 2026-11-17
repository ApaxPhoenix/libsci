#include <sci.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

int main() {
    constexpr std::size_t rows = 37;
    constexpr std::size_t inner = 53;
    constexpr std::size_t columns = 29;

    sci::Array<float, 2> left(sci::Shape<2>{rows, inner});
    sci::Array<float, 2> right(sci::Shape<2>{inner, columns});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < inner; ++column)
            left[row, column] = static_cast<float>((row * 7 + column * 3) % 11) - 5.0f;
    }
    for (std::size_t row = 0; row < inner; ++row) {
        for (std::size_t column = 0; column < columns; ++column)
            right[row, column] = static_cast<float>((row * 5 + column) % 13) * 0.25f;
    }

    // The product agrees with the three loops a textbook writes.
    const sci::Array<float, 2> result = sci::product(left, right);
    assert((result.shape() == sci::Shape<2>{rows, columns}) && "a product's shape is rows by columns");
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            double expected = 0.0;
            for (std::size_t step = 0; step < inner; ++step)
                expected += static_cast<double>(left[row, step]) * static_cast<double>(right[step, column]);
            assert((std::abs(result[row, column] - expected) < 1e-3) && "every element is the row times the column");
        }
    }

    // A dot product of vectors, and of an expression, which is never stored.
    const sci::Array<float, 1> first = sci::linspace(0.0f, 1.0f, 1000);
    const sci::Array<float, 1> second = sci::ones<float>({1000});
    double expected = 0.0;
    for (std::size_t index = 0; index < 1000; ++index)
        expected += first[index];
    assert((std::abs(sci::dot(first, second) - expected) < 1e-3) && "dot is the sum of the products");
    assert((std::abs(sci::dot(first * 2.0f, second) - 2.0 * expected) < 1e-2) && "of expressions too");
    return 0;
}
