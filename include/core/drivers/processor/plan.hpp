#pragma once

#include "core/bytecode/chunk.hpp"
#include "core/bytecode/precision.hpp"
#include "core/drivers/processor/context.hpp"
#include "core/drivers/processor/pool.hpp"
#include "core/drivers/processor/range.hpp"
#include "core/types.hpp"

#include <algorithm>

namespace sci::core::drivers::processor {

    // How a chunk's elements are split across the pool: parts of whole blocks, and how many threads
    // may take part.
    class Plan {
    public:
        // Elements per block. Every temporary of an expression is one block, so its whole working
        // set stays in L1 while it is evaluated (1024 measured best on an M1).
        static constexpr size block = 1024;

        Plan(const Chunk& chunk, const size count)
            : count(count) {
            const size blocks = (count + block - 1) / block;
            const size cost = chunk.cost();
            if (blocks < 2 || count * cost < threshold) return;
            const size width = Pool::global().width();
            if (width < 2) return;

            const size element = chunk.precision == Precision::single ? sizeof(float) : sizeof(double);
            const size touched = count * element * (chunk.arguments + 1);

            // Several parts per thread so faster cores pick up the slack.
            parts = std::min(blocks, width * 8);
            seats = cost <= streaming && touched >= spill ? Context::performance() : 0;
        }

        // The elements of one part, in whole blocks except at the very end.
        [[nodiscard]] Range range(const size part) const noexcept {
            const size blocks = (count + block - 1) / block;
            return {.first = blocks * part / parts * block, .last = std::min(blocks * (part + 1) / parts * block, count)};
        }

        size parts{1};
        size seats{1};

    private:
        // Weighted element count below which forking costs more than it saves.
        static constexpr size threshold = size{1} << 17;

        // Chunks this cheap per element only stream memory...
        static constexpr uint32 streaming = 16;

        // ...and once they touch more than the caches hold, DRAM is the limit: extra slow cores on a
        // hybrid chip only add contention (measured on an M1: 4 threads beat 8 by about 15%).
        static constexpr size spill = size{32} << 20;

        size count;
    };

}
