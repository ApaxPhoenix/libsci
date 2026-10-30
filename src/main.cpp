#include <sci.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ratio>

int main() {
    const sci::Array<float, 2> alpha({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const sci::Array<float, 2> beta({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    const sci::Array<float, 2> gamma({2, 2}, {0.0f, 0.5f, -0.5f, 1.0f});
    const sci::Array<float, 2> delta({2, 2}, {0.25f, 0.5f, 0.75f, 1.0f}); // Valid domain for asech: (0, 1]
    const sci::Array<float, 2> epsilon({2, 2}, {1.5f, 2.0f, 2.5f, 3.0f}); // Valid domain for acoth: |x| > 1

    std::chrono::duration<double, std::micro> overall{0};

    // Runs one operation with its reduction, then prints the result and how long it took.
    const auto measure = [&overall](const char* name, const auto& work) {
        const auto start = std::chrono::steady_clock::now();
        const float result = work();
        const auto finish = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::micro> elapsed = finish - start;
        overall += elapsed;
        std::cout << std::left << std::setw(22) << name << std::setw(12) << result << elapsed.count() << " us\n";
    };

    // Standard operations & reductions
    measure("Sum of add", [&] { return sci::sum(sci::add(alpha, beta)); });
    measure("Mean of product", [&] { return sci::mean(sci::multiply(alpha, beta)); });
    measure("Minimum of sine", [&] { return sci::minimum(sci::sine(alpha)); });
    measure("Maximum of sine", [&] { return sci::maximum(sci::sine(alpha)); });
    measure("Sum of arcsine", [&] {
        const sci::Array<float, 2> inverse = sci::arcsine(gamma);
        return sci::sum(inverse);
    });

    // Existing hyperbolic functions
    measure("Sum of sinh", [&] {
        const sci::Array<float, 2> hyperbolic = sci::hyperbolic::sine(alpha);
        return sci::sum(hyperbolic);
    });
    measure("Sum of asinh", [&] {
        const sci::Array<float, 2> inverse = sci::hyperbolic::arcsine(alpha);
        return sci::sum(inverse);
    });

    // Newly added hyperbolic functions tests
    measure("Sum of csch", [&] {
        return sci::sum(sci::hyperbolic::cosecant(alpha));
    });
    measure("Sum of sech", [&] {
        return sci::sum(sci::hyperbolic::secant(alpha));
    });
    measure("Sum of coth", [&] {
        return sci::sum(sci::hyperbolic::cotangent(alpha));
    });
    measure("Sum of acsch", [&] {
        return sci::sum(sci::hyperbolic::arccosecant(alpha));
    });
    measure("Sum of asech", [&] {
        return sci::sum(sci::hyperbolic::arcsecant(delta));
    });
    measure("Sum of acoth", [&] {
        return sci::sum(sci::hyperbolic::arccotangent(epsilon));
    });

    std::cout << "Compute time: " << overall.count() << " us (printing excluded)\n";

    return 0;
}