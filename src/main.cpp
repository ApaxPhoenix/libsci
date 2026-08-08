#include <iostream>
#include <array/ndarray.hpp>
#include <ops/binary.hpp>
#include <ops/unary.hpp>
#include <ops/reduce.hpp>
#include <linalg/matrix.hpp>

int main() {
    using namespace sci;

    constexpr array::shape<2> bounds{2, 2};

    array::ndarray<float, 2> alpha(bounds);
    array::ndarray<float, 2> beta(bounds);

    float* left = alpha.data();
    float* right = beta.data();

    left[0] = 1.0f;
    left[1] = 2.0f;
    left[2] = 3.0f;
    left[3] = 4.0f;

    right[0] = 5.0f;
    right[1] = 6.0f;
    right[2] = 7.0f;
    right[3] = 8.0f;

    auto sum = ops::add(alpha, beta);
    auto product = ops::mul(alpha, beta);
    auto transformed = ops::sin(alpha);

    std::cout << "Sum aggregate: " << ops::sum(sum) << "\n";
    std::cout << "Mean aggregate: " << ops::mean(product) << "\n";
    std::cout << "Min aggregate: " << ops::min(transformed) << "\n";
    std::cout << "Max aggregate: " << ops::max(transformed) << "\n";

    auto output = linalg::matmul(alpha, beta);

    const float* data = output.data();
    for (std::size_t index = 0; index < output.size(); ++index) {
        std::cout << data[index] << " ";
    }
    std::cout << "\n";

    return 0;
}