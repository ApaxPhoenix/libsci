#pragma once

#include "core/types.hpp"

namespace sci::core::memory {

    struct slice {
        size start{0};
        size stop{0};
        size step{1};

        constexpr slice() noexcept = default;

        explicit constexpr slice(const size end) noexcept
            : stop(end) {}

        constexpr slice(const size begin, const size end, const size rate = 1) noexcept
            : start(begin), stop(end), step(rate) {}

        [[nodiscard]] constexpr size length() const noexcept {
            if (start >= stop || step == 0) return 0;
            return (stop - start + step - 1) / step;
        }
    };

}