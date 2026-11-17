#include <sci.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

int main() {
    // A chunk is interpreted the first time it runs and compiled the second: every run must give
    // the kernels' answers to the bit.
    constexpr std::size_t count = 50'000;
    const sci::Array<float, 1> first = sci::linspace(-2.0f, 2.0f, count);
    const sci::Array<float, 1> second = sci::linspace(0.5f, 1.5f, count);

    const sci::Array<float, 1> interpreted = (first + second) * (first - second) + 0.5f;
    for (int run = 0; run < 3; ++run) {
        const sci::Array<float, 1> compiled = (first + second) * (first - second) + 0.5f;
        assert((compiled == interpreted) && "a compiled chunk agrees with its first, interpreted run");
    }

    for (std::size_t index = 0; index < count; index += 251) {
        const float expected = std::fma(first[index] + second[index], first[index] - second[index], 0.5f);
        assert((std::abs(interpreted[index] - expected) <= 1e-6f * std::max(1.0f, std::abs(expected))) &&
               "and with a plain loop");
    }
    return 0;
}
