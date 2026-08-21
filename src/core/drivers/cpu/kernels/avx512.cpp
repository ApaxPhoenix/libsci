#include "core/drivers/cpu/kernels/avx512.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)

#include <immintrin.h>

namespace sci::core::drivers::cpu::kernels {

    void avx512::add(memory::cursor<float> left, memory::cursor<float> right, memory::cursor<float> yield, const size count) noexcept {
        size index = 0;
        for (const size limit = count & ~static_cast<size>(15); index < limit; index += 16) {
            const __m512 first = _mm512_loadu_ps(left.raw() + index);
            const __m512 second = _mm512_loadu_ps(right.raw() + index);
            _mm512_storeu_ps(yield.raw() + index, _mm512_add_ps(first, second));
        }
        for (; index < count; ++index) {
            yield.raw()[index] = left.raw()[index] + right.raw()[index];
        }
    }

}

#endif