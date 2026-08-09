#pragma once

#include <core/types.hpp>

namespace sci::array {

    struct slice {
        core::size start = 0;
        core::size stop = 0;
        core::size step = 1;

        constexpr core::size length() const noexcept {
            if (stop <= start) return 0;
            return (stop - start + step - 1) / step;
        }
    };

}