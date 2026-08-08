#pragma once

#include <core/arch/alignment.hpp>
#include <core/types.hpp>

namespace sci::core {

    constexpr size align(size value) noexcept {
        return (value + arch::boundary - 1) & ~(arch::boundary - 1);
    }

}