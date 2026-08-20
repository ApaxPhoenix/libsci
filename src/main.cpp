#include <chrono>
#include <iostream>
#include <algorithm>
#include <array/ndarray.hpp>
#include <ops/binary.hpp>
#include <ops/unary.hpp>
#include <ops/reduce.hpp>
#include <linalg/matmul.hpp>

int main() {
    using namespace sci;

    const auto start = std::chrono::high_resolution_clock::now();

    constexpr array::shape<2> bounds{2, 2};

    array::ndarray<float, 2> alpha(bounds);
    array::ndarray<float, 2> beta(bounds);

    constexpr float source[] = {1.0f, 2.0f, 3.0f, 4.0f};
    constexpr float target[] = {5.0f, 6.0f, 7.0f, 8.0f};

    std::copy_n(source, 4, alpha.data());
    std::copy_n(target, 4, beta.data());

    const auto sum = ops::add(alpha, beta);
    const auto product = ops::mul(alpha, beta);
    const auto transform = ops::sin(alpha);

    std::cout << "Sum aggregate: " << ops::sum(sum) << "\n";
    std::cout << "Mean aggregate: " << ops::mean(product) << "\n";
    std::cout << "Min aggregate: " << ops::min(transform) << "\n";
    std::cout << "Max aggregate: " << ops::max(transform) << "\n";

    const auto finish = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = finish - start;

    std::cout << "Execution time: " << elapsed.count() << " ms\n";

    return 0;
}