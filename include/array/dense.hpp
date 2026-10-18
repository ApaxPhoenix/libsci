#pragma once

#include "core/types.hpp"

#include <type_traits>

namespace sci::array {

    template <typename type, core::size order>
    class Array;

    // Marks Array, as Deferred marks expressions.
    template <typename type>
    struct Dense : std::false_type {};

    template <typename type, core::size order>
    struct Dense<Array<type, order>> : std::true_type {};

}
