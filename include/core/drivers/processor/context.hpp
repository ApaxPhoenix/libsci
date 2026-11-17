#pragma once

/// @file
/// @brief What this CPU offers, detected at runtime.

#include "core/types.hpp"

namespace sci::core::drivers::processor {

    /// What this CPU offers, detected at runtime.
    struct Context {
        /// @brief Whether the CPU has AVX2 and FMA.
        bool avx2{false};
        /// @brief Whether it has AVX-512.
        bool avx512{false};
        /// @brief Whether it has NEON.
        bool neon{false};

        /// Instruction sets only, so the first array operation does not pay for system calls.
        /// @return The instruction sets.
        [[nodiscard]] static Context probe() noexcept;

        /// Performance cores on hybrid chips (Apple P/E cores); 0 when unknown or not hybrid. Asking
        /// costs a few system calls, so it happens once, the first time a job is big enough to care.
        /// @return How many performance cores.
        [[nodiscard]] static size performance() noexcept;
    };

}
