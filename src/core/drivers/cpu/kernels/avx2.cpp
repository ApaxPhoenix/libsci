// Built with -mavx2 -mfma (see CMakeLists.txt); selected at runtime only when the CPU supports both.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))

#include <immintrin.h>

#include "core/drivers/cpu/kernels/avx2.hpp"
#include "core/drivers/cpu/kernels/generic/compose.hpp"
#include "core/drivers/cpu/kernels/generic/scalar.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::cpu::kernels {

    // AVX2 registers: eight floats or four doubles.

    template <typename type>
    struct Avx2;

    template <>
    struct Avx2<float> {
        using element = float;
        using narrow = Scalar<float, uint32, Avx2>;
        using value = __m256;
        using bits = __m256i;
        using mask = __m256;

        static constexpr size width = 8;

        static value load(const float* source) noexcept { return _mm256_loadu_ps(source); }
        static void store(float* target, const value item) noexcept { _mm256_storeu_ps(target, item); }
        static value splat(const float item) noexcept { return _mm256_set1_ps(item); }
        static bits fill(const uint32 item) noexcept { return _mm256_set1_epi32(static_cast<int>(item)); }

        static value add(const value left, const value right) noexcept { return _mm256_add_ps(left, right); }
        static value sub(const value left, const value right) noexcept { return _mm256_sub_ps(left, right); }
        static value mul(const value left, const value right) noexcept { return _mm256_mul_ps(left, right); }
        static value div(const value left, const value right) noexcept { return _mm256_div_ps(left, right); }
        static value fma(const value left, const value right, const value addend) noexcept { return _mm256_fmadd_ps(left, right, addend); }
        static value fnma(const value left, const value right, const value addend) noexcept { return _mm256_fnmadd_ps(left, right, addend); }

        static value neg(const value item) noexcept { return _mm256_xor_ps(item, _mm256_set1_ps(-0.0f)); }
        static value abs(const value item) noexcept { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), item); }
        static value sqrt(const value item) noexcept { return _mm256_sqrt_ps(item); }

        static value min(const value left, const value right) noexcept {
            return _mm256_blendv_ps(_mm256_min_ps(left, right), left, unordered(left));
        }

        static value max(const value left, const value right) noexcept {
            return _mm256_blendv_ps(_mm256_max_ps(left, right), left, unordered(left));
        }

        static mask less(const value left, const value right) noexcept { return _mm256_cmp_ps(left, right, _CMP_LT_OQ); }
        static mask greater(const value left, const value right) noexcept { return _mm256_cmp_ps(left, right, _CMP_GT_OQ); }
        static mask equal(const value left, const value right) noexcept { return _mm256_cmp_ps(left, right, _CMP_EQ_OQ); }
        static mask unordered(const value item) noexcept { return _mm256_cmp_ps(item, item, _CMP_UNORD_Q); }

        static mask test(const bits item) noexcept {
            const __m256i zero = _mm256_cmpeq_epi32(item, _mm256_setzero_si256());
            return _mm256_castsi256_ps(_mm256_xor_si256(zero, _mm256_set1_epi32(-1)));
        }

        static bool any(const mask item) noexcept { return _mm256_movemask_ps(item) != 0; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return _mm256_blendv_ps(right, left, choice);
        }

        static bits tobits(const value item) noexcept { return _mm256_castps_si256(item); }
        static value cast(const bits item) noexcept { return _mm256_castsi256_ps(item); }

        static bits band(const bits left, const bits right) noexcept { return _mm256_and_si256(left, right); }
        static bits bor(const bits left, const bits right) noexcept { return _mm256_or_si256(left, right); }
        static bits bxor(const bits left, const bits right) noexcept { return _mm256_xor_si256(left, right); }

        template <int count>
        static bits shl(const bits item) noexcept { return _mm256_slli_epi32(item, count); }

        template <int count>
        static bits shr(const bits item) noexcept { return _mm256_srli_epi32(item, count); }
    };

    template <>
    struct Avx2<double> {
        using element = double;
        using narrow = Scalar<double, uint64, Avx2>;
        using value = __m256d;
        using bits = __m256i;
        using mask = __m256d;

        static constexpr size width = 4;

        static value load(const double* source) noexcept { return _mm256_loadu_pd(source); }
        static void store(double* target, const value item) noexcept { _mm256_storeu_pd(target, item); }
        static value splat(const double item) noexcept { return _mm256_set1_pd(item); }
        static bits fill(const uint64 item) noexcept { return _mm256_set1_epi64x(static_cast<long long>(item)); }

        static value add(const value left, const value right) noexcept { return _mm256_add_pd(left, right); }
        static value sub(const value left, const value right) noexcept { return _mm256_sub_pd(left, right); }
        static value mul(const value left, const value right) noexcept { return _mm256_mul_pd(left, right); }
        static value div(const value left, const value right) noexcept { return _mm256_div_pd(left, right); }
        static value fma(const value left, const value right, const value addend) noexcept { return _mm256_fmadd_pd(left, right, addend); }
        static value fnma(const value left, const value right, const value addend) noexcept { return _mm256_fnmadd_pd(left, right, addend); }

        static value neg(const value item) noexcept { return _mm256_xor_pd(item, _mm256_set1_pd(-0.0)); }
        static value abs(const value item) noexcept { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), item); }
        static value sqrt(const value item) noexcept { return _mm256_sqrt_pd(item); }

        static value min(const value left, const value right) noexcept {
            return _mm256_blendv_pd(_mm256_min_pd(left, right), left, unordered(left));
        }

        static value max(const value left, const value right) noexcept {
            return _mm256_blendv_pd(_mm256_max_pd(left, right), left, unordered(left));
        }

        static mask less(const value left, const value right) noexcept { return _mm256_cmp_pd(left, right, _CMP_LT_OQ); }
        static mask greater(const value left, const value right) noexcept { return _mm256_cmp_pd(left, right, _CMP_GT_OQ); }
        static mask equal(const value left, const value right) noexcept { return _mm256_cmp_pd(left, right, _CMP_EQ_OQ); }
        static mask unordered(const value item) noexcept { return _mm256_cmp_pd(item, item, _CMP_UNORD_Q); }

        static mask test(const bits item) noexcept {
            const __m256i zero = _mm256_cmpeq_epi64(item, _mm256_setzero_si256());
            return _mm256_castsi256_pd(_mm256_xor_si256(zero, _mm256_set1_epi64x(-1)));
        }

        static bool any(const mask item) noexcept { return _mm256_movemask_pd(item) != 0; }

        static value select(const mask choice, const value left, const value right) noexcept {
            return _mm256_blendv_pd(right, left, choice);
        }

        static bits tobits(const value item) noexcept { return _mm256_castpd_si256(item); }
        static value cast(const bits item) noexcept { return _mm256_castsi256_pd(item); }

        static bits band(const bits left, const bits right) noexcept { return _mm256_and_si256(left, right); }
        static bits bor(const bits left, const bits right) noexcept { return _mm256_or_si256(left, right); }
        static bits bxor(const bits left, const bits right) noexcept { return _mm256_xor_si256(left, right); }

        template <int count>
        static bits shl(const bits item) noexcept { return _mm256_slli_epi64(item, count); }

        template <int count>
        static bits shr(const bits item) noexcept { return _mm256_srli_epi64(item, count); }
    };

    constexpr Table table = compose<Avx2>("avx2");

    const Table* avx2() noexcept {
        return &table;
    }

}

#else

#include "core/drivers/cpu/kernels/avx2.hpp"

namespace sci::core::drivers::cpu::kernels {

    const Table* avx2() noexcept {
        return nullptr;
    }

}

#endif
