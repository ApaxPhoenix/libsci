#pragma once

#include <concepts>
#include <type_traits>

namespace sci::core {

    template <typename type>
    concept numeric = std::is_arithmetic_v<type>;

}