#include <sci.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ratio>

int main() {
    const sci::Array<float, 2> alpha({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const sci::Array<float, 2> beta({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    const sci::Array<float, 2> gamma({2, 2}, {0.0f, 0.5f, -0.5f, 1.0f});

    std::chrono::duration<double, std::micro> overall{0};

    // Runs one operation with its reduction, then prints the result and how long it took.
    const auto measure = [&overall](const char* name, const auto& work) {
        const auto start = std::chrono::steady_clock::now();
        const float result = work();
        const auto finish = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::micro> elapsed = finish - start;
        overall += elapsed;
        std::cout << std::left << std::setw(20) << name << std::setw(12) << result << elapsed.count() << " us\n";
    };

    measure("Sum of add", [&] { return sci::sum(sci::add(alpha, beta)); });
    measure("Mean of product", [&] { return sci::mean(sci::multiply(alpha, beta)); });
    measure("Minimum of sine", [&] { return sci::minimum(sci::sine(alpha)); });
    measure("Maximum of sine", [&] { return sci::maximum(sci::sine(alpha)); });
    measure("Sum of arcsine", [&] {
        const sci::Array<float, 2> inverse = sci::arcsine(gamma);
        return sci::sum(inverse);
    });
    measure("Sum of sinh", [&] {
        const sci::Array<float, 2> hyperbolic = sci::hyperbolic::sine(alpha);
        return sci::sum(hyperbolic);
    });
    measure("Sum of asinh", [&] {
        const sci::Array<float, 2> inverse = sci::hyperbolic::arcsine(alpha);
        return sci::sum(inverse);
    });

    std::cout << "Compute time: " << overall.count() << " us (printing excluded)\n";

    return 0;
}
