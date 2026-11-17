#pragma once

/// @file
/// @brief Folds the partial results of a reduction in double precision, starting from the fold's
///       identity (0, +infinity or -infinity).

#include "core/bytecode/fold.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sci::core {

    /// Folds the partial results of a reduction in double precision, starting from the fold's
    /// identity (0, +infinity or -infinity). Minimum and maximum let NaN win, as numpy does.
    class Accumulator {
    public:
        /// @brief An accumulator for a fold, starting from its identity.
        /// @param fold Sum, minimum or maximum.
        explicit Accumulator(const Fold fold) noexcept
            : fold(fold), total(fold == Fold::sum       ? 0.0
                                : fold == Fold::minimum ? infinity
                                                        : -infinity) {}

        /// @brief Folds in a value.
        /// @param value The value.
        void add(const double value) noexcept {
            if (fold == Fold::sum || std::isnan(total) || std::isnan(value)) {
                total += value;
            } else {
                total = fold == Fold::minimum ? std::min(total, value) : std::max(total, value);
            }
        }

        /// @brief The fold so far.
        /// @return The result.
        [[nodiscard]] double result() const noexcept { return total; }

    private:
        /// @brief Infinity, where minimum and maximum start.
        static constexpr double infinity = std::numeric_limits<double>::infinity();

        /// @brief Which fold it is.
        Fold fold;
        /// @brief The fold so far.
        double total;
    };

}
