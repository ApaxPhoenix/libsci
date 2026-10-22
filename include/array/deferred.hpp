#pragma once

#include <type_traits>

namespace sci::array {

    // Marks lazy expression types; Node and Reference each specialize it next to their definition.
    template <typename type>
    struct Deferred : std::false_type {};

}
