#pragma once

#include "core/types.hpp"

#include <type_traits>

namespace sci::array {

    template <typename element, core::size rank>
    class NDArray;

    // Marks NDArray, with its number of dimensions as `order`.
    template <typename type>
    struct Dense : std::false_type {};

    template <typename element, core::size rank>
    struct Dense<NDArray<element, rank>> : std::true_type {
        static constexpr core::size order = rank;
    };

}
