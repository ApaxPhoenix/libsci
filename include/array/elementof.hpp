#pragma once

#include "core/types.hpp"

#include <type_traits>

namespace sci::array {

    template <typename element, core::size rank>
    class NDArray;

    template <typename type>
    struct ElementOf {
        using kind = type::element;
    };

    template <typename type, core::size rank>
    struct ElementOf<NDArray<type, rank>> {
        using kind = type;
    };

    // Element type of an array or expression.
    template <typename type>
    using element_t = ElementOf<std::remove_cvref_t<type>>::kind;

}
