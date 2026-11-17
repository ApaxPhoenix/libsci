#include <sci.hpp>

#include <algorithm>
#include <cassert>

int main() {
    const sci::Array<float, 2> nothing = sci::zeros<float>({3, 5});
    assert((nothing.shape() == sci::Shape<2>{3, 5}) && "zeros takes its shape");
    assert(std::ranges::all_of(nothing, [](float value) { return value == 0.0f; }) && "and holds zeros");

    const sci::Array<float, 1> unity = sci::ones<float>({8});
    assert(std::ranges::all_of(unity, [](float value) { return value == 1.0f; }) && "ones holds ones");

    const sci::Array<int, 2> sevens = sci::full<int>({2, 2}, 7);
    assert(std::ranges::all_of(sevens, [](int value) { return value == 7; }) && "full holds its value");

    const sci::Array<float, 1> spaced = sci::linspace(0.0f, 1.0f, 5);
    assert((spaced[0] == 0.0f && spaced[2] == 0.5f && spaced[4] == 1.0f) && "linspace includes both ends");

    const sci::Array<int, 1> counted = sci::range(5);
    assert((counted.shape() == sci::Shape<1>{5} && counted[0] == 0 && counted[4] == 4) && "range stops before stop");

    const sci::Array<float, 1> stepped = sci::range(1.0f, 2.0f, 0.25f);
    assert((stepped.shape() == sci::Shape<1>{4} && stepped[3] == 1.75f) && "and steps as asked");

    const sci::Array<float, 2> identity = sci::eye<float>(3);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            assert((identity[row, column] == (row == column ? 1.0f : 0.0f)) && "eye is the identity");
        }
    }

    const sci::Array<float, 3> uninitialized = sci::empty<float>({2, 3, 4});
    assert((uninitialized.shape() == sci::Shape<3>{2, 3, 4}) && "empty has its shape, whatever it holds");
    return 0;
}
