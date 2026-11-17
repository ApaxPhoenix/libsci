#include <sci.hpp>

#include <cassert>
#include <stdexcept>

int main() {
    const auto volume = sci::reshape(sci::range(12), {2, 2, 3});
    assert((volume.shape() == sci::Shape<3>{2, 2, 3} && (volume[1, 1, 2] == 11)) && "reshape keeps row-major order");

    bool refused = false;
    try {
        (void)sci::reshape(sci::range(12), {5, 2});
    } catch (const std::invalid_argument &) {
        refused = true;
    }
    assert(refused && "a reshape that changes the count is refused");

    const auto transposed = sci::transpose(volume);
    const sci::Array<int, 3> expected(sci::Shape<3>{3, 2, 2}, {0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11});
    assert((transposed == expected) && "transpose reverses every axis");
    assert((sci::transpose(transposed) == volume) && "and twice is the array again");

    const auto flat = sci::operations::flatten(volume);
    assert((flat.shape() == sci::Shape<1>{12} && flat[7] == 7) && "flatten keeps the order");

    const auto column = sci::reshape(sci::range(4), {4, 1});
    const auto squeezed = sci::operations::squeeze(column, 1);
    assert((squeezed.shape() == sci::Shape<1>{4} && squeezed[3] == 3) && "squeeze drops an axis of one");

    const auto expanded = sci::operations::expand(squeezed, 0);
    assert((expanded.shape() == sci::Shape<2>{1, 4}) && "expand adds one");

    refused = false;
    try {
        (void)sci::operations::squeeze(volume, 0);
    } catch (const std::invalid_argument &) {
        refused = true;
    }
    assert(refused && "and squeeze refuses an axis longer than one");
    return 0;
}
