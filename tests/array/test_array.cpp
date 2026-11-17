#include <sci.hpp>

#include <algorithm>
#include <cassert>

int main() {
    // Built with a value, every element holds it.
    sci::Array<float, 2> filled(sci::Shape<2>{3, 4}, 2.5f);
    assert((filled.shape() == sci::Shape<2>{3, 4}) && "an array keeps its shape");
    assert(std::ranges::all_of(filled, [](float value) { return value == 2.5f; }) && "and the value it was given");

    // Built from a list, row-major, the rest zero.
    const sci::Array<int, 2> listed(sci::Shape<2>{2, 3}, {1, 2, 3, 4});
    assert((listed[0, 2] == 3 && listed[1, 0] == 4) && "a list fills rows first");
    assert((listed[1, 2] == 0) && "and what it leaves out is zero");

    // Written through an index, read back through it.
    filled[2, 3] = 9.0f;
    assert((filled[2, 3] == 9.0f && filled[0, 0] == 2.5f) && "an element is written alone");

    // An expression is computed when it is stored.
    const sci::Array<float, 1> first(sci::Shape<1>{1000}, 3.0f);
    const sci::Array<float, 1> second(sci::Shape<1>{1000}, 4.0f);
    const sci::Array<float, 1> third = first * second + 1.0f;
    assert(std::ranges::all_of(third, [](float value) { return value == 13.0f; }) && "a stored expression is computed");

    // Assigning another shape reshapes the array.
    sci::Array<float, 1> reused(sci::Shape<1>{10}, 0.0f);
    reused = first + second;
    assert((reused.shape() == first.shape() && reused[999] == 7.0f) && "assignment takes the expression's shape");

    // Two arrays are equal when their shapes and elements are.
    const sci::Array<int, 2> same(sci::Shape<2>{2, 3}, {1, 2, 3, 4});
    assert((same == listed) && "equal arrays compare equal");
    assert(!(same == sci::Array<int, 2>(sci::Shape<2>{3, 2}, {1, 2, 3, 4})) && "a different shape is not equal");

    // The host's copy of a host array is the array.
    assert((third.host() == third) && "host() of a host array is the same values");
    return 0;
}
