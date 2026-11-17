#pragma once

/// @file
/// @brief What arrays and expressions share: size, sum, mean, minimum, maximum and copy.

#include "core/bytecode/fold.hpp"
#include "core/types.hpp"

namespace sci::array {

    template <typename type, core::size order> class Array;

    /// Defined in expression.hpp.
    /// @param input The array or expression.
    /// @param fold Sum, minimum or maximum.
    /// @return The result.
    template <typename source> source::element reduce(const source &input, core::Fold fold);

    /// Array-like members shared by Array and every expression node.
    template <typename derived> struct Methods {
        /// @brief How many elements (numpy.ndarray.size).
        /// @return How many elements.
        [[nodiscard]] core::size size() const noexcept { return self().shape().total(); }
        /// @brief The sum of every element (numpy.sum).
        /// @return The sum.
        [[nodiscard]] auto sum() const { return reduce(self(), core::Fold::sum); }
        /// @brief The least element (numpy.min); NaN anywhere gives NaN.
        /// @return The minimum.
        [[nodiscard]] auto minimum() const { return reduce(self(), core::Fold::minimum); }
        /// @brief The greatest element (numpy.max); NaN anywhere gives NaN.
        /// @return The maximum.
        [[nodiscard]] auto maximum() const { return reduce(self(), core::Fold::maximum); }

        /// @brief The mean of every element (numpy.mean).
        /// @return The mean.
        [[nodiscard]] auto mean() const {
            using element = derived::element;
            return size() == 0 ? element{} : sum() / static_cast<element>(size());
        }

        /// A new array holding the values (numpy.copy) on the device they live on.
        /// @return The new array.
        [[nodiscard]] auto copy() const { return Array<typename derived::element, derived::rank>(capture(self())); }

    private:
        Methods() = default;
        /// @brief Only the array or expression itself can construct these members.
        friend derived;

        /// @brief The array or expression these members belong to.
        /// @return It.
        [[nodiscard]] const derived &self() const noexcept { return static_cast<const derived &>(*this); }
    };

}
