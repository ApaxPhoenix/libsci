#include "core/drivers/cpu/kernels/avx2.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)

#include <immintrin.h>

namespace sci::core::drivers::cpu::kernels {

    void avx2::add(memory::cursor<float> left, memory::cursor<float> right, memory::cursor<float> yield, const size count) noexcept {
        size index = 0;
        for (const size limit = count & ~static_cast<size>(7); index < limit; index += 8) {
            const __m256 first = _mm256_loadu_ps(left.raw() + index);
            const __m256 second = _mm256_loadu_ps(right.raw() + index);
            _mm256_storeu_ps(yield.raw() + index, _mm256_add_ps(first, second));
        }
        for (; index < count; ++index) {
            yield.raw()[index] = left.raw()[index] + right.raw()[index];
        }
    }

}

#endif