#include "core/drivers/processor/context.hpp"
#include "core/types.hpp"

#if (defined(_M_X64) || defined(_M_IX86)) && defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace sci::core::drivers::processor {

    Context Context::probe() noexcept {
        Context context{};
        #if defined(__x86_64__) || defined(__i386__)
            __builtin_cpu_init();
            context.avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
            context.avx512 = context.avx2 && __builtin_cpu_supports("avx512f");
        #elif defined(_M_X64) || defined(_M_IX86)
            int registers[4]{};
            __cpuid(registers, 0);
            const int top = registers[0];
            __cpuid(registers, 1);
            const bool fused = (registers[2] & (1 << 12)) != 0;
            const bool saves = (registers[2] & (1 << 27)) != 0;
            // Which register state the operating system saves: 256-bit (wide) and 512-bit (widest).
            const unsigned long long features = saves ? _xgetbv(0) : 0;
            const bool wide = (features & 0x6) == 0x6;
            const bool widest = (features & 0xE6) == 0xE6;
            if (top >= 7) {
                __cpuidex(registers, 7, 0);
                context.avx2 = wide && fused && (registers[1] & (1 << 5)) != 0;
                context.avx512 = context.avx2 && widest && (registers[1] & (1 << 16)) != 0;
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
