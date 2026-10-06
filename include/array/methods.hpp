#pragma once

#include "array/elementof.hpp"
#include "core/bytecode/fold.hpp"
#include "core/types.hpp"

namespace sci::array {

    template <typename element, core::size rank>
    class NDArray;

    // Defined in expression.hpp.
    template <typename source>
    element_t<source> reduce(const source& input, core::Fold fold);

    // Array-like members shared by every expression node.
    template <typename derived>
    struct Methods {
        [[nodiscard]] core::size size() const noexcept { return self().shape().total(); }
        [[nodiscard]] auto sum() const { return reduce(self(), core::Fold::sum); }
        [[nodiscard]] auto min() const { return reduce(self(), core::Fold::min); }
        [[nodiscard]] auto max() const { return reduce(self(), core::Fold::max); }

        [[nodiscard]] auto mean() const {
            using element = derived::element;
            return size() == 0 ? element{} : sum() / static_cast<element>(size());
        }

        // Computes the expression into a new array (numpy.copy).
        [[nodiscard]] auto copy() const { return NDArray<typename derived::element, derived::rank>(self()); }

    private:
        Methods() = default;
        friend derived;

        [[nodiscard]] const derived& self() const noexcept { return static_cast<const derived&>(*this); }
    };

}
