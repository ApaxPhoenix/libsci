#include "core/drivers/cpu/kernels/fallback.hpp"

namespace sci::core::drivers::cpu::kernels {

    void fallback::add(memory::cursor<float> left, memory::cursor<float> right, memory::cursor<float> yield, const size count) noexcept {
        for (size index = 0; index < count; ++index) {
            yield.get() = left.get() + right.get();
            left.advance(1);
            right.advance(1);
            yield.advance(1);
        }
    }

}