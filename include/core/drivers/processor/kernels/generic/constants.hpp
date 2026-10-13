#pragma once

#include "core/types.hpp"

#include <bit>
#include <numbers>

namespace sci::core::drivers::processor::kernels {

    template <typename type>
    struct Constants;

    template <>
    struct Constants<float> {
        static constexpr int mantissa = 23;
        static constexpr int width = 32;
        static constexpr uint32 fraction = 0x007fffffu;
        static constexpr uint32 one = 0x3f800000u;

        static constexpr float round = 12582912.0f;       // 1.5 * 2^23: adding it rounds to an integer
        static constexpr float magnitude = 8388608.0f;    // 2^23
        static constexpr float exponent = 8388735.0f;     // 2^23 + bias

        static constexpr float binary = std::numbers::log2e_v<float>;  // log2(e) = 1 / ln 2
        // ln 2 = natural + remainder; natural is ln 2 to 9 bits, so k * natural is exact.
        static constexpr float natural = 355.0f / 512.0f;
        static constexpr float remainder = -2.12194440e-4f;
        static constexpr float lower = -104.0f;
        static constexpr float upper = 89.0f;
        static constexpr float normal = 126.0f;

        static constexpr float tiny = 1.17549435e-38f;
        static constexpr float lift = 33554432.0f;        // 2^25
        static constexpr float shift = 25.0f;
        static constexpr float root = std::numbers::sqrt2_v<float>;  // sqrt(2)

        // quadrant = 2/pi turns radians into quarter turns, and pi/2 = quarter[0] + ... + quarter[3].
        // With fma the first two steps of the reduction are exact, so the reduced argument carries
        // about one rounding error even next to a zero.
        static constexpr float quadrant = 0.636619772367581343f;
        static constexpr float quarter[4] = {1.5703125f, 4.837512969970703125e-4f, 7.549790126404332e-08f, -1.7151245100058819e-15f};
        static constexpr float reach = 1048576.0f;

        // Degree of each polynomial.
        static constexpr size exponential = 7;
        static constexpr size logarithm = 4;
        static constexpr size sine = 5;
        static constexpr size cosine = 6;
    };

    template <>
    struct Constants<double> {
        static constexpr int mantissa = 52;
        static constexpr int width = 64;
        static constexpr uint64 fraction = 0x000fffffffffffffull;
        static constexpr uint64 one = 0x3ff0000000000000ull;

        static constexpr double round = 6755399441055744.0;      // 1.5 * 2^52
        static constexpr double magnitude = 4503599627370496.0;  // 2^52
        static constexpr double exponent = 4503599627371519.0;   // 2^52 + bias

        static constexpr double binary = std::numbers::log2e;    // log2(e) = 1 / ln 2
        // ln 2 = natural + remainder; natural is ln 2 with its low 21 bits cleared, so k * natural is exact.
        static constexpr double natural = std::bit_cast<double>(std::bit_cast<uint64>(std::numbers::ln2) & 0xFFFFFFFFFFE00000ULL);
        static constexpr double remainder = 1.90821492927058770002e-10;
        static constexpr double lower = -746.0;
        static constexpr double upper = 710.0;
        static constexpr double normal = 1022.0;

        static constexpr double tiny = 2.2250738585072014e-308;
        static constexpr double lift = 18014398509481984.0;      // 2^54
        static constexpr double shift = 54.0;
        static constexpr double root = std::numbers::sqrt2;      // sqrt(2)

        static constexpr double quadrant = 6.36619772367581382433e-01;
        static constexpr double quarter[4] = {1.57079632673412561417e+00, 6.07710050630396597660e-11, 2.02226624871116645580e-21, 8.47842766036889956997e-32};
        static constexpr double reach = 1048576.0;

        // Degree of each polynomial.
        static constexpr size exponential = 13;
        static constexpr size logarithm = 10;
        static constexpr size sine = 9;
        static constexpr size cosine = 10;
    };

}
