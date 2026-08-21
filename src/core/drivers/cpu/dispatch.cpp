#include "core/drivers/cpu/dispatch.hpp"
#include "core/drivers/cpu/kernels/avx2.hpp"
#include "core/drivers/cpu/kernels/avx512.hpp"
#include "core/drivers/cpu/kernels/fallback.hpp"
#include "core/drivers/cpu/kernels/neon.hpp"

namespace sci::core::drivers::cpu {

    dispatch::dispatch(const context data) {
        if (data.avx512) {
            routine = kernels::avx512::add;
        } else if (data.avx2) {
            routine = kernels::avx2::add;
        } else if (data.neon) {
            routine = kernels::neon::add;
        } else {
            routine = kernels::fallback::add;
        }
    }

    void dispatch::add(const memory::cursor<float> left, const memory::cursor<float> right, const memory::cursor<float> yield, const size count) const noexcept {
        routine(left, right, yield, count);
    }

}