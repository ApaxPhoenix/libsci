#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::processor {

    // Elements [first, last) of a chunk's arrays.
    struct Range {
        size first;
        size last;
    };

}
