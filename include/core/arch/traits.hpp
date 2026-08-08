#pragma once

#include <core/arch/alignment.hpp>
#include <core/types.hpp>

namespace sci::core::arch {

    template <typename type>
    inline constexpr size lanes = boundary / sizeof(type);

}