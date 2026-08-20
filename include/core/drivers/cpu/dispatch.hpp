#pragma once

#include "core/drivers/cpu/context.hpp"
#include "core/memory/cursor.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu {

    class dispatch {
        using kernel = void(*)(memory::cursor<float>, memory::cursor<float>, memory::cursor<float>, size);

    public:
        explicit dispatch(context data = context::probe());

        void add(memory::cursor<float> left, memory::cursor<float> right, memory::cursor<float> yield, size count) const noexcept;

    private:
        kernel routine;
    };

}