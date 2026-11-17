#include <sci.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>

int main() {
    // Large enough that the thread pool splits it: every element is written once, by one thread,
    // and none is left out where two ranges meet.
    constexpr std::size_t count = 4'000'003;
    const sci::Array<float, 1> ones = sci::ones<float>({count});
    const sci::Array<float, 1> doubled = ones + ones;
    assert(std::ranges::all_of(doubled, [](float value) { return value == 2.0f; }) && "every element is written");
    assert((sci::sum(doubled) == 2.0f * count) && "and a split reduction counts each once");

    // Again and again, so a race between runs would show.
    for (int run = 0; run < 20; ++run)
        assert((sci::sum(ones * 3.0f) == 3.0f * count) && "the pool is reused safely");
    return 0;
}
