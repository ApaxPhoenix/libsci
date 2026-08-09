#pragma once

#include <concepts>
#include <limits>
#include <type_traits>

namespace sci::core {

    template <typename type>
    struct scalar {
        static constexpr bool value = std::is_scalar_v<type>;
    };

    template <typename type>
    concept valid = scalar<type>::value;

    template <typename type>
    concept floatable = std::floating_point<type>;

    template <floatable type>
    inline constexpr type epsilon = std::numeric_limits<type>::epsilon();

}