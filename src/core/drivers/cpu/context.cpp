#include "core/drivers/cpu/context.hpp"
#include "core/types.hpp"

#if (defined(_M_X64) || defined(_M_IX86)) && defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace sci::core::drivers::cpu {

    Context Context::probe() noexcept {
        Context context{};
        #if defined(__x86_64__) || defined(__i386__)
            __builtin_cpu_init();
            context.avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
            context.avx512 = context.avx2 && __builtin_cpu_supports("avx512f");
        #elif defined(_M_X64) || defined(_M_IX86)
            int info[4]{};
            __cpuid(info, 0);
            const int top = info[0];
            __cpuid(info, 1);
            const bool fma = (info[2] & (1 << 12)) != 0;
            const bool saves = (info[2] & (1 << 27)) != 0;
            const unsigned long long features = saves ? _xgetbv(0) : 0;
            const bool ymm = (features & 0x6) == 0x6;
            const bool zmm = (features & 0xE6) == 0xE6;
            if (top >= 7) {
                __cpuidex(info, 7, 0);
                context.avx2 = ymm && fma && (info[1] & (1 << 5)) != 0;
                context.avx512 = context.avx2 && zmm && (info[1] & (1 << 16)) != 0;
            }
        #elif defined(__aarch64__) || defined(_M_ARM64)
            context.neon = true;
        #endif
        return context;
    }

    size Context::performance() noexcept {
        static const size cores = [] {
            size count = 0;
            #if defined(__APPLE__)
                // perflevel0 is the fastest cluster; a second level means the chip is hybrid.
                int levels = 0;
                size length = sizeof(levels);
                if (sysctlbyname("hw.nperflevels", &levels, &length, nullptr, 0) == 0 && levels > 1) {
                    int found = 0;
                    length = sizeof(found);
                    if (sysctlbyname("hw.perflevel0.logicalcpu", &found, &length, nullptr, 0) == 0 && found > 0) {
                        count = static_cast<size>(found);
                    }
                }
            #endif
            return count;
        }();
        return cores;
    }

}
