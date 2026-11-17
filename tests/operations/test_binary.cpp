#include <sci.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace tests::operations {
    // Large enough to cross the 1,024-element blocks and to be split across the thread pool.
    constexpr std::size_t count = 300'000;

    template <typename expression, typename reference>
    bool matches(const sci::Array<float, 1> &first, const sci::Array<float, 1> &second, const expression &computed,
                 const reference &expected) {
        const sci::Array<float, 1> result = computed;
        for (std::size_t index = 0; index < count; ++index) {
            if (result[index] != expected(first[index], second[index])) return false;
        }
        return true;
    }
}

int main() {
    using namespace tests::operations;

    const sci::Array<float, 1> first = sci::linspace(-3.0f, 5.0f, count);
    const sci::Array<float, 1> second = sci::linspace(2.0f, 7.0f, count);

    assert(matches(first, second, sci::add(first, second), [](float a, float b) { return a + b; }) && "add");
    assert(matches(first, second, sci::subtract(first, second), [](float a, float b) { return a - b; }) && "subtract");
    assert(matches(first, second, sci::multiply(first, second), [](float a, float b) { return a * b; }) && "multiply");
    assert(matches(first, second, sci::divide(first, second), [](float a, float b) { return a / b; }) && "divide");
    assert(matches(first, second, sci::minimum(first, second), [](float a, float b) { return std::min(a, b); }) &&
           "minimum");
    assert(matches(first, second, sci::maximum(first, second), [](float a, float b) { return std::max(a, b); }) &&
           "maximum");
    assert(matches(first, second, first + 2.0f, [](float a, float) { return a + 2.0f; }) && "a scalar on the right");
    assert(matches(first, second, 2.0f * first, [](float a, float) { return 2.0f * a; }) && "and on the left");

    // A multiply followed by an add is fused, so it rounds once, as std::fma does.
    assert(matches(first, second, first * second + first, [](float a, float b) { return std::fma(a, b, a); }) &&
           "a multiply-add is fused into one rounding");

    // NaN wins in minimum and maximum, as in numpy.
    const sci::Array<float, 1> nan(sci::Shape<1>{4}, std::nanf(""));
    const sci::Array<float, 1> one(sci::Shape<1>{4}, 1.0f);
    const sci::Array<float, 1> lowest = sci::minimum(nan, one);
    const sci::Array<float, 1> highest = sci::maximum(one, nan);
    assert(std::ranges::all_of(lowest, [](float value) { return std::isnan(value); }) && "NaN wins a minimum");
    assert(std::ranges::all_of(highest, [](float value) { return std::isnan(value); }) && "and a maximum");
    return 0;
}
