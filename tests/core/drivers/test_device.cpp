#include <sci.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

int main() {
    // A graphics device, or the CPU when the machine has no usable Vulkan device: the same code
    // either way, and the same answers.
    sci::Device graphics(sci::Target::graphics);
    assert((graphics.target() == sci::Target::graphics || graphics.target() == sci::Target::processor) &&
           "a device is the GPU or, without one, the CPU");

    constexpr std::size_t count = 200'000;
    const sci::Array<float, 1> first = sci::linspace(0.0f, 2.0f, count);
    const sci::Array<float, 1> second = sci::linspace(1.0f, 3.0f, count);
    const sci::Array<float, 1> expected = first * second + 1.0f;

    const auto moved = first.to(graphics);
    const auto other = second.to(graphics);
    const sci::Array<float, 1> computed = moved * other + 1.0f;
    const sci::Array<float, 1> read = computed.host();
    for (std::size_t index = 0; index < count; index += 997)
        assert((std::abs(read[index] - expected[index]) < 1e-5f) && "an expression on the device agrees with the host");

    assert((std::abs(sci::sum(moved) - sci::sum(first)) < 1e-1f) && "and so does a reduction");
    return 0;
}
