#include "core/drivers/cpu/kernels/neon.hpp"

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

namespace sci::core::drivers::cpu::kernels {

    void neon::add(memory::cursor<float> left, memory::cursor<float> right, memory::cursor<float> yield, const size count) noexcept {
        size index = 0;
        for (const size limit = count & ~static_cast<size>(3); index < limit; index += 4) {
            const float32x4_t first = vld1q_f32(left.raw() + index);
            const float32x4_t second = vld1q_f32(right.raw() + index);
            vst1q_f32(yield.raw() + index, vaddq_f32(first, second));
        }
        for (; index < count; ++index) {
            yield.raw()[index] = left.raw()[index] + right.raw()[index];
        }
    }

}

#endif