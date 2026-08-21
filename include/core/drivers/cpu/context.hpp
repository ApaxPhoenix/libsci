#pragma once

namespace sci::core::drivers::cpu {

    struct context {
        bool avx2{false};
        bool avx512{false};
        bool neon{false};

        [[nodiscard]] static context probe() noexcept;
    };

}