#include "core/drivers/cpu/context.hpp"

namespace sci::core::drivers::cpu {

    context context::probe() noexcept {
        context state{};
        #if defined(__AVX2__)
            state.avx2 = true;
        #endif
        #if defined(__AVX512F__)
            state.avx512 = true;
        #endif
        #if defined(__ARM_NEON) || defined(__ARM_NEON__)
            state.neon = true;
        #endif
        return state;
    }

}