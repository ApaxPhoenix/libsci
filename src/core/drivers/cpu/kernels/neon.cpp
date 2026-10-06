#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include "core/drivers/cpu/kernels/neon.hpp"
#include "core/drivers/cpu/kernels/generic/compose.hpp"
#include "core/drivers/cpu/kernels/generic/scalar.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    // NEON registers: four floats or two doubles.

    template <typename type>
    struct Neon;

    template <>
    struct Neon<float> {
        using element = float;
        using narrow = Scalar<float, uint32, Neon>;
        using value = float32x4_t;
        using bits = uint32x4_t;
        using mask = uint32x4_t;

        static constexpr size width = 4;

        static value load(const float* source) noexcept { return vld1q_f32(source); }
        static void store(float* target, const value item) noexcept { vst1q_f32(target, item); }
        static value splat(const float item) noexcept { return vdupq_n_f32(item); }
        static bits fill(const uint32 item) noexcept { return vdupq_n_u32(item); }

        static value add(const value left, const value right) noexcept { return vaddq_f32(left, right); }
        static value sub(const value left, const value right) noexcept { return vsubq_f32(left, right); }
        static value mul(const value left, const value right) noexcept { return vmulq_f32(left, right); }
        static value div(const value left, const value right) noexcept { return vdivq_f32(left, right); }
        static value fma(const value left, const value right, const value addend) noexcept { return vfmaq_f32(addend, left, right); }
        static value fnma(const value left, const value right, const value addend) noexcept { return vfmsq_f32(addend, left, right); }

        static value neg(const value item) noexcept { return vnegq_f32(item); }
        static value abs(const value item) noexcept { return vabsq_f32(item); }
        static value sqrt(const value item) noexcept { return vsqrtq_f32(item); }
        static value min(const value left, const value right) noexcept { return vminq_f32(left, right); }
        static value max(const value left, const value right) noexcept { return vmaxq_f32(left, right); }

        static mask less(const value left, const value right) noexcept { return vcltq_f32(left, right); }
        static mask greater(const value left, const value right) noexcept { return vcgtq_f32(left, right); }
        static mask equal(const value left, const value right) noexcept { return vceqq_f32(left, right); }
        static mask unordered(const value item) noexcept { return vmvnq_u32(vceqq_f32(item, item)); }
        static mask test(const bits item) noexcept { return vtstq_u32(item, item); }
        static bool any(const mask item) noexcept { return vmaxvq_u32(item) != 0; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return vbslq_f32(choice, left, right);
        }

        static bits tobits(const value item) noexcept { return vreinterpretq_u32_f32(item); }
        static value cast(const bits item) noexcept { return vreinterpretq_f32_u32(item); }

        static bits band(const bits left, const bits right) noexcept { return vandq_u32(left, right); }
        static bits bor(const bits left, const bits right) noexcept { return vorrq_u32(left, right); }
        static bits bxor(const bits left, const bits right) noexcept { return veorq_u32(left, right); }

        template <int count>
        static bits shl(const bits item) noexcept { return vshlq_n_u32(item, count); }

        template <int count>
        static bits shr(const bits item) noexcept { return vshrq_n_u32(item, count); }
    };

    template <>
    struct Neon<double> {
        using element = double;
        using narrow = Scalar<double, uint64, Neon>;
        using value = float64x2_t;
        using bits = uint64x2_t;
        using mask = uint64x2_t;

        static constexpr size width = 2;

        static value load(const double* source) noexcept { return vld1q_f64(source); }
        static void store(double* target, const value item) noexcept { vst1q_f64(target, item); }
        static value splat(const double item) noexcept { return vdupq_n_f64(item); }
        static bits fill(const uint64 item) noexcept { return vdupq_n_u64(item); }

        static value add(const value left, const value right) noexcept { return vaddq_f64(left, right); }
        static value sub(const value left, const value right) noexcept { return vsubq_f64(left, right); }
        static value mul(const value left, const value right) noexcept { return vmulq_f64(left, right); }
        static value div(const value left, const value right) noexcept { return vdivq_f64(left, right); }
        static value fma(const value left, const value right, const value addend) noexcept { return vfmaq_f64(addend, left, right); }
        static value fnma(const value left, const value right, const value addend) noexcept { return vfmsq_f64(addend, left, right); }

        static value neg(const value item) noexcept { return vnegq_f64(item); }
        static value abs(const value item) noexcept { return vabsq_f64(item); }
        static value sqrt(const value item) noexcept { return vsqrtq_f64(item); }
        static value min(const value left, const value right) noexcept { return vminq_f64(left, right); }
        static value max(const value left, const value right) noexcept { return vmaxq_f64(left, right); }

        static mask less(const value left, const value right) noexcept { return vcltq_f64(left, right); }
        static mask greater(const value left, const value right) noexcept { return vcgtq_f64(left, right); }
        static mask equal(const value left, const value right) noexcept { return vceqq_f64(left, right); }
        static mask unordered(const value item) noexcept { return veorq_u64(vceqq_f64(item, item), vdupq_n_u64(~uint64{0})); }
        static mask test(const bits item) noexcept { return vtstq_u64(item, item); }
        static bool any(const mask item) noexcept { return vmaxvq_u32(vreinterpretq_u32_u64(item)) != 0; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return vbslq_f64(choice, left, right);
        }

        static bits tobits(const value item) noexcept { return vreinterpretq_u64_f64(item); }
        static value cast(const bits item) noexcept { return vreinterpretq_f64_u64(item); }

        static bits band(const bits left, const bits right) noexcept { return vandq_u64(left, right); }
        static bits bor(const bits left, const bits right) noexcept { return vorrq_u64(left, right); }
        static bits bxor(const bits left, const bits right) noexcept { return veorq_u64(left, right); }

        template <int count>
        static bits shl(const bits item) noexcept { return vshlq_n_u64(item, count); }

        template <int count>
        static bits shr(const bits item) noexcept { return vshrq_n_u64(item, count); }
    };

    constexpr Table table = compose<Neon>("neon");

    const Table* neon() noexcept {
        return &table;
    }

}

#else

#include "core/drivers/cpu/kernels/neon.hpp"

namespace sci::core::drivers::cpu::kernels {

    const Table* neon() noexcept {
        return nullptr;
    }

}

#endif
