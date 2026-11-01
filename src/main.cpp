#include <sci.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ratio>
#include <vector>

int main() {
    constexpr std::size_t elements = 1'000'000;
    constexpr int warmup = 3;
    constexpr int runs = 20;

    const sci::Array<float, 1> ramp = sci::range(0.0f, 1.0f, 1.0f / static_cast<float>(elements));

    const sci::Array<float, 1> alpha   = sci::linspace(3.0f, 4.0f, elements);
    const sci::Array<float, 1> beta    = sci::linspace(3.0f, 8.0f, elements);
    const sci::Array<float, 1> gamma   = sci::linspace(2.0f, 1.0f, elements);
    const sci::Array<float, 1> delta   = sci::linspace(0.75f, 1.0f, elements); // Valid domain for asech: (0, 1]
    const sci::Array<float, 1> epsilon = sci::linspace(1.5f, 3.0f, elements); // Valid domain for acoth: |x| > 1
    const sci::Array<float, 1> zeta    = sci::linspace(1.25f, 0.75f, elements);

    std::chrono::duration<double, std::micro> overall{0};

    // Runs one operation with its reduction, then prints the result and how long it took.
    const auto measure = [&overall](const char* name, const auto& work) {
        float result = 0.0f;
        for (int run = 0; run < warmup; ++run) {
            result = work();
        }

        std::vector<double> times(runs);
        for (double& time : times) {
            const auto start = std::chrono::steady_clock::now();
            result = work();
            const auto finish = std::chrono::steady_clock::now();
            time = std::chrono::duration<double, std::micro>(finish - start).count();
        }

        std::sort(times.begin(), times.end());
        const double median = times[times.size() / 2];
        overall += std::chrono::duration<double, std::micro>(median);
        std::cout << std::left << std::setw(22) << name << std::setw(14) << result << std::setw(12) << median
                  << std::setw(10) << median * 1000.0 / static_cast<double>(elements) << " ns/elem\n";
    };

    std::cout << std::left << std::setw(22) << "Operation" << std::setw(14) << "Result" << std::setw(12) << "Median us"
              << "Per element\n";

    measure("Sum of add", [&] { return sci::sum(sci::add(alpha, beta)); });
    measure("Sum of subtract", [&] { return sci::sum(sci::subtract(alpha, beta)); });
    measure("Sum of multiply", [&] { return sci::sum(sci::multiply(alpha, beta)); });
    measure("Sum of divide", [&] { return sci::sum(sci::divide(alpha, beta)); });
    measure("Sum of minimum", [&] { return sci::sum(sci::minimum(alpha, beta)); });
    measure("Sum of maximum", [&] { return sci::sum(sci::maximum(alpha, beta)); });

    measure("Sum of copy", [&] { return sci::sum(sci::copy(alpha)); });
    measure("Sum of negate", [&] { return sci::sum(sci::negate(alpha)); });
    measure("Sum of absolute", [&] { return sci::sum(sci::absolute(gamma)); });
    measure("Sum of root", [&] { return sci::sum(sci::root(alpha)); });
    measure("Sum of exponential", [&] { return sci::sum(sci::exponential(alpha)); });
    measure("Sum of logarithm", [&] { return sci::sum(sci::logarithm(alpha)); });

    measure("Sum of sin", [&] { return sci::sum(sci::sin(alpha)); });
    measure("Sum of cos", [&] { return sci::sum(sci::cos(alpha)); });
    measure("Sum of tan", [&] { return sci::sum(sci::tan(alpha)); });
    measure("Sum of asin", [&] { return sci::sum(sci::asin(gamma)); });
    measure("Sum of acos", [&] { return sci::sum(sci::acos(gamma)); });
    measure("Sum of atan", [&] { return sci::sum(sci::atan(alpha)); });
    measure("Sum of csc", [&] { return sci::sum(sci::csc(alpha)); });
    measure("Sum of sec", [&] { return sci::sum(sci::sec(alpha)); });
    measure("Sum of cot", [&] { return sci::sum(sci::cot(alpha)); });
    measure("Sum of acsc", [&] { return sci::sum(sci::acsc(alpha)); });
    measure("Sum of asec", [&] { return sci::sum(sci::asec(alpha)); });
    measure("Sum of acot", [&] { return sci::sum(sci::acot(alpha)); });

    measure("Sum", [&] { return sci::sum(alpha); });
    measure("Sum (method)", [&] { return alpha.sum(); });
    measure("Mean", [&] { return sci::mean(alpha); });
    measure("Minimum", [&] { return sci::minimum(alpha); });
    measure("Maximum", [&] { return sci::maximum(alpha); });

    measure("Sum of sinh", [&] { return sci::sum(sci::sinh(alpha)); });
    measure("Sum of cosh", [&] { return sci::sum(sci::cosh(alpha)); });
    measure("Sum of tanh", [&] { return sci::sum(sci::tanh(alpha)); });
    measure("Sum of asinh", [&] { return sci::sum(sci::asinh(alpha)); });
    measure("Sum of acosh", [&] { return sci::sum(sci::acosh(alpha)); });
    measure("Sum of atanh", [&] { return sci::sum(sci::atanh(zeta)); });

    measure("Sum of csch", [&] { return sci::sum(sci::csch(alpha)); });
    measure("Sum of sech", [&] { return sci::sum(sci::sech(alpha)); });
    measure("Sum of coth", [&] { return sci::sum(sci::coth(alpha)); });
    measure("Sum of acsch", [&] { return sci::sum(sci::acsch(alpha)); });
    measure("Sum of asech", [&] { return sci::sum(sci::asech(delta)); });
    measure("Sum of acoth", [&] { return sci::sum(sci::acoth(epsilon)); });

    std::cout << "Compute time: " << overall.count() << " us (printing excluded)\n";

    return 0;
}