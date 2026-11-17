#include <sci.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

int main() {
    constexpr std::size_t count = 1'000'000;
    const sci::Array<float, 1> ramp = sci::linspace(-1.0f, 2.0f, count);

    double total = 0.0;
    float lowest = ramp[0];
    float highest = ramp[0];
    for (std::size_t index = 0; index < count; ++index) {
        total += ramp[index];
        lowest = std::fmin(lowest, ramp[index]);
        highest = std::fmax(highest, ramp[index]);
    }

    assert((std::abs(sci::sum(ramp) - total) < 1e-3 * count) && "sum agrees with a loop in double");
    assert((std::abs(sci::mean(ramp) - total / count) < 1e-3) && "and mean is it over the count");
    assert((sci::minimum(ramp) == lowest && sci::maximum(ramp) == highest) && "minimum and maximum are exact");

    // A reduction of an expression never stores the expression, and agrees with storing it.
    const sci::Array<float, 1> squared = ramp * ramp;
    assert((std::abs(sci::sum(ramp * ramp) - sci::sum(squared)) < 1e-2f * count) && "a fused reduction agrees");
    assert((std::abs(ramp.sum() - sci::sum(ramp)) < 1e-6f * count) && "the method is the function");
    return 0;
}
