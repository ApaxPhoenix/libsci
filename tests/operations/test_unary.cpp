#include <sci.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace tests::operations {
    constexpr std::size_t count = 100'000;

    // Within a few units in the last place of the C library, relative to the value's size.
    template <typename expression, typename reference>
    bool near(const sci::Array<float, 1> &input, const expression &computed, const reference &expected) {
        const sci::Array<float, 1> result = computed;
        for (std::size_t index = 0; index < count; ++index) {
            const double wanted = expected(static_cast<double>(input[index]));
            const double error = std::abs(static_cast<double>(result[index]) - wanted);
            if (error > 1e-5 * std::max(1.0, std::abs(wanted))) return false;
        }
        return true;
    }
}

int main() {
    using namespace tests::operations;

    const sci::Array<float, 1> wide = sci::linspace(-4.0f, 4.0f, count);
    const sci::Array<float, 1> positive = sci::linspace(0.25f, 9.0f, count);
    const sci::Array<float, 1> unit = sci::linspace(-0.95f, 0.95f, count);
    const sci::Array<float, 1> beyond = sci::linspace(1.25f, 6.0f, count);

    assert(near(wide, sci::negate(wide), [](double x) { return -x; }) && "negate");
    assert(near(wide, sci::absolute(wide), [](double x) { return std::abs(x); }) && "absolute");
    assert(near(positive, sci::root(positive), [](double x) { return std::sqrt(x); }) && "root");
    assert(near(wide, sci::exponential(wide), [](double x) { return std::exp(x); }) && "exponential");
    assert(near(positive, sci::logarithm(positive), [](double x) { return std::log(x); }) && "logarithm");

    assert(near(wide, sci::sin(wide), [](double x) { return std::sin(x); }) && "sin");
    assert(near(wide, sci::cos(wide), [](double x) { return std::cos(x); }) && "cos");
    assert(near(unit, sci::tan(unit), [](double x) { return std::tan(x); }) && "tan");
    assert(near(unit, sci::asin(unit), [](double x) { return std::asin(x); }) && "asin");
    assert(near(unit, sci::acos(unit), [](double x) { return std::acos(x); }) && "acos");
    assert(near(wide, sci::atan(wide), [](double x) { return std::atan(x); }) && "atan");

    assert(near(wide, sci::sinh(wide), [](double x) { return std::sinh(x); }) && "sinh");
    assert(near(wide, sci::cosh(wide), [](double x) { return std::cosh(x); }) && "cosh");
    assert(near(wide, sci::tanh(wide), [](double x) { return std::tanh(x); }) && "tanh");
    assert(near(wide, sci::asinh(wide), [](double x) { return std::asinh(x); }) && "asinh");
    assert(near(beyond, sci::acosh(beyond), [](double x) { return std::acosh(x); }) && "acosh");
    assert(near(unit, sci::atanh(unit), [](double x) { return std::atanh(x); }) && "atanh");
    return 0;
}
