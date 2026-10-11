#pragma once

#include "core/bytecode/fold.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sci::core {

    // Folds the partial results of a reduction in double precision, starting from the fold's
    // identity (0, +infinity or -infinity). Minimum and maximum let NaN win, as numpy does.
    class Accumulator {
    public:
        explicit Accumulator(const Fold fold) noexcept
            : fold(fold),
              total(fold == Fold::sum ? 0.0 : (fold == Fold::minimum ? infinity : -infinity)) {}

        void add(const double value) noexcept {
            if (fold == Fold::sum || std::isnan(total) || std::isnan(value)) {
                total += value;
            } else {
                total = fold == Fold::minimum ? std::min(total, value) : std::max(total, value);
            }
        }

        [[nodiscard]] double result() const noexcept { return total; }

    private:
        static constexpr double infinity = std::numeric_limits<double>::infinity();

        Fold fold;
        double total;
    };

}
