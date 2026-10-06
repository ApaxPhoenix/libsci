#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::cpu {

    // What this CPU offers, detected at runtime.
    struct Context {
        bool avx2{false};
        bool avx512{false};
        bool neon{false};

        // Instruction sets only, so the first array operation does not pay for system calls.
        [[nodiscard]] static Context probe() noexcept;

        // Performance cores on hybrid chips (Apple P/E cores); 0 when unknown or not hybrid. Asking
        // costs a few system calls, so it happens once, the first time a job is big enough to care.
        [[nodiscard]] static size performance() noexcept;
    };

}
