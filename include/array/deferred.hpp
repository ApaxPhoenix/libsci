#pragma once

#include <type_traits>

namespace sci::array {

    // Marks lazy expression types; Node, Reference and Owner each specialize it next to their definition.
    template <typename type>
    struct Deferred : std::false_type {};

}
