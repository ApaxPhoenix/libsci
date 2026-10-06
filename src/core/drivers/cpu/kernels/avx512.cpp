// Built with -mavx512f (see CMakeLists.txt); selected at runtime only when the CPU and OS support it.
#if defined(__AVX512F__) && (defined(__FMA__) || defined(_MSC_VER))

#include <immintrin.h>

#include "core/drivers/cpu/kernels/avx512.hpp"
#include "core/drivers/cpu/kernels/generic/compose.hpp"
#include "core/drivers/cpu/kernels/generic/scalar.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    // AVX-512 registers: sixteen floats or eight doubles; masks are k-registers.

    template <typename type>
    struct Avx512;

    template <>
    struct Avx512<float> {
        using element = float;
        using narrow = Scalar<float, uint32, Avx512>;
        using value = __m512;
        using bits = __m512i;
        using mask = __mmask16;

        static constexpr size width = 16;

        static value load(const float* source) noexcept { return _mm512_loadu_ps(source); }
        static void store(float* target, const value item) noexcept { _mm512_storeu_ps(target, item); }
        static value splat(const float item) noexcept { return _mm512_set1_ps(item); }
        static bits fill(const uint32 item) noexcept { return _mm512_set1_epi32(static_cast<int>(item)); }

        static value add(const value left, const value right) noexcept { return _mm512_add_ps(left, right); }
        static value sub(const value left, const value right) noexcept { return _mm512_sub_ps(left, right); }
        static value mul(const value left, const value right) noexcept { return _mm512_mul_ps(left, right); }
        static value div(const value left, const value right) noexcept { return _mm512_div_ps(left, right); }
        static value fma(const value left, const value right, const value addend) noexcept { return _mm512_fmadd_ps(left, right, addend); }
        static value fnma(const value left, const value right, const value addend) noexcept { return _mm512_fnmadd_ps(left, right, addend); }

        static value neg(const value item) noexcept { return cast(bxor(tobits(item), fill(0x80000000u))); }
        static value abs(const value item) noexcept { return cast(band(tobits(item), fill(0x7fffffffu))); }
        static value sqrt(const value item) noexcept { return _mm512_sqrt_ps(item); }

        static value min(const value left, const value right) noexcept {
            return _mm512_mask_blend_ps(unordered(left), _mm512_min_ps(left, right), left);
        }

        static value max(const value left, const value right) noexcept {
            return _mm512_mask_blend_ps(unordered(left), _mm512_max_ps(left, right), left);
        }

        static mask less(const value left, const value right) noexcept { return _mm512_cmp_ps_mask(left, right, _CMP_LT_OQ); }
        static mask greater(const value left, const value right) noexcept { return _mm512_cmp_ps_mask(left, right, _CMP_GT_OQ); }
        static mask equal(const value left, const value right) noexcept { return _mm512_cmp_ps_mask(left, right, _CMP_EQ_OQ); }
        static mask unordered(const value item) noexcept { return _mm512_cmp_ps_mask(item, item, _CMP_UNORD_Q); }
        static mask test(const bits item) noexcept { return _mm512_test_epi32_mask(item, item); }
        static bool any(const mask item) noexcept { return item != 0; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return _mm512_mask_blend_ps(choice, right, left);
        }

        static bits tobits(const value item) noexcept { return _mm512_castps_si512(item); }
        static value cast(const bits item) noexcept { return _mm512_castsi512_ps(item); }

        static bits band(const bits left, const bits right) noexcept { return _mm512_and_si512(left, right); }
        static bits bor(const bits left, const bits right) noexcept { return _mm512_or_si512(left, right); }
        static bits bxor(const bits left, const bits right) noexcept { return _mm512_xor_si512(left, right); }

        template <int count>
        static bits shl(const bits item) noexcept { return _mm512_slli_epi32(item, count); }

        template <int count>
        static bits shr(const bits item) noexcept { return _mm512_srli_epi32(item, count); }
    };

    template <>
    struct Avx512<double> {
        using element = double;
        using narrow = Scalar<double, uint64, Avx512>;
        using value = __m512d;
        using bits = __m512i;
        using mask = __mmask8;

        static constexpr size width = 8;

        static value load(const double* source) noexcept { return _mm512_loadu_pd(source); }
        static void store(double* target, const value item) noexcept { _mm512_storeu_pd(target, item); }
        static value splat(const double item) noexcept { return _mm512_set1_pd(item); }
        static bits fill(const uint64 item) noexcept { return _mm512_set1_epi64(static_cast<long long>(item)); }

        static value add(const value left, const value right) noexcept { return _mm512_add_pd(left, right); }
        static value sub(const value left, const value right) noexcept { return _mm512_sub_pd(left, right); }
        static value mul(const value left, const value right) noexcept { return _mm512_mul_pd(left, right); }
        static value div(const value left, const value right) noexcept { return _mm512_div_pd(left, right); }
        static value fma(const value left, const value right, const value addend) noexcept { return _mm512_fmadd_pd(left, right, addend); }
        static value fnma(const value left, const value right, const value addend) noexcept { return _mm512_fnmadd_pd(left, right, addend); }

        static value neg(const value item) noexcept { return cast(bxor(tobits(item), fill(0x8000000000000000ull))); }
        static value abs(const value item) noexcept { return cast(band(tobits(item), fill(0x7fffffffffffffffull))); }
        static value sqrt(const value item) noexcept { return _mm512_sqrt_pd(item); }

        static value min(const value left, const value right) noexcept {
            return _mm512_mask_blend_pd(unordered(left), _mm512_min_pd(left, right), left);
        }

        static value max(const value left, const value right) noexcept {
            return _mm512_mask_blend_pd(unordered(left), _mm512_max_pd(left, right), left);
        }

        static mask less(const value left, const value right) noexcept { return _mm512_cmp_pd_mask(left, right, _CMP_LT_OQ); }
        static mask greater(const value left, const value right) noexcept { return _mm512_cmp_pd_mask(left, right, _CMP_GT_OQ); }
        static mask equal(const value left, const value right) noexcept { return _mm512_cmp_pd_mask(left, right, _CMP_EQ_OQ); }
        static mask unordered(const value item) noexcept { return _mm512_cmp_pd_mask(item, item, _CMP_UNORD_Q); }
        static mask test(const bits item) noexcept { return _mm512_test_epi64_mask(item, item); }
        static bool any(const mask item) noexcept { return item != 0; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return _mm512_mask_blend_pd(choice, right, left);
        }

        static bits tobits(const value item) noexcept { return _mm512_castpd_si512(item); }
        static value cast(const bits item) noexcept { return _mm512_castsi512_pd(item); }

        static bits band(const bits left, const bits right) noexcept { return _mm512_and_si512(left, right); }
        static bits bor(const bits left, const bits right) noexcept { return _mm512_or_si512(left, right); }
        static bits bxor(const bits left, const bits right) noexcept { return _mm512_xor_si512(left, right); }

        template <int count>
        static bits shl(const bits item) noexcept { return _mm512_slli_epi64(item, count); }

        template <int count>
        static bits shr(const bits item) noexcept { return _mm512_srli_epi64(item, count); }
    };

    constexpr Table table = compose<Avx512>("avx512");

    const Table* avx512() noexcept {
        return &table;
    }

}

#else

#include "core/drivers/cpu/kernels/avx512.hpp"

namespace sci::core::drivers::cpu::kernels {

    const Table* avx512() noexcept {
        return nullptr;
    }

}

#endif
