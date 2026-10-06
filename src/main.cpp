#include <sci.hpp>

#include <chrono>
#include <iostream>
#include <ratio>

int main() {
    const auto start = std::chrono::steady_clock::now();

    const sci::NDArray<float, 2> alpha({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const sci::NDArray<float, 2> beta({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    const auto sum = sci::add(alpha, beta);
    const auto product = sci::multiply(alpha, beta);
    const auto transform = sci::sin(alpha);

    const float total = sci::sum(sum);
    const float mean = sci::mean(product);
    const float low = sci::min(transform);
    const float high = sci::max(transform);

    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::micro> elapsed = finish - start;

    std::cout << "Sum aggregate: " << total << "\n";
    std::cout << "Mean aggregate: " << mean << "\n";
    std::cout << "Min aggregate: " << low << "\n";
    std::cout << "Max aggregate: " << high << "\n";
    std::cout << "Compute time: " << elapsed.count() << " us (printing excluded)\n";

    return 0;
}
