#pragma once

/// @file
/// @brief An operation in an expression: an interior node of the tree.

#include "array/calculate.hpp"
#include "array/deferred.hpp"
#include "array/methods.hpp"
#include "array/shape.hpp"
#include "core/assert.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/code.hpp"
#include "core/drivers/device.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sci::array {

    using core::drivers::Device;

    /// Interior node: code applied to its operands.
    template <core::Code code, typename... operands> struct Node : Methods<Node<code, operands...>> {
        /// @brief The element type, the first operand's.
        using element = std::tuple_element_t<0, std::tuple<operands...>>::element;
        /// @brief The most dimensions among the operands.
        static constexpr core::size rank = std::max({operands::rank...});

        static_assert((std::is_same_v<typename operands::element, element> && ...),
                      "operands must share an element type");
        static_assert(((operands::rank == 0 || operands::rank == rank) && ...), "operands must share a rank");
        static_assert(rank > 0, "an expression needs at least one array operand");

        /// @brief The operands.
        std::tuple<operands...> children;

        /// @brief A node over its operands.
        /// @param items The operands.
        explicit Node(operands... items) : children(std::move(items)...) {}

        /// Operands must agree on their shape (no broadcasting yet, apart from scalars): the first array
        /// operand gives it, and every other array operand must match.
        /// @return The shape.
        [[nodiscard]] Shape<rank> shape() const noexcept {
            const Shape<rank> result = std::get<lead>(children).shape();
            // Scalars match any shape.
            ASSERT(std::apply(
                    [&](const auto &...child) {
                        return ([&] {
                            if constexpr (std::remove_cvref_t<decltype(child)>::rank > 0) {
                                return child.shape() == result;
                            } else {
                                return true;
                            }
                        }() && ...);
                    },
                    children));
            return result;
        }

        /// The first device found among the leaves decides where the expression runs.
        /// @return The device, or null for the host.
        [[nodiscard]] Device *device() const noexcept {
            Device *found = nullptr;
            std::apply([&](const auto &...child) { ((found = found ? found : child.device()), ...); }, children);
            return found;
        }

        /// @brief Records the operands, then the code applied to them.
        /// @param chunk The chunk recorded into.
        /// @return The result's value id.
        [[nodiscard]] core::uint8 record(core::Chunk &chunk) const noexcept {
            return std::apply(
                    [&](const auto &...child) {
                        const core::uint8 slots[] = {child.record(chunk)...};
                        if constexpr (sizeof...(operands) == 1) {
                            return chunk.emit(code, slots[0]);
                        } else if constexpr (sizeof...(operands) == 2) {
                            return chunk.emit(code, slots[0], slots[1]);
                        } else {
                            return chunk.emit(code, slots[0], slots[1], slots[2]);
                        }
                    },
                    children);
        }

        /// @brief One element of the result, computed on the host.
        /// @param index The element's flat index.
        /// @return The element.
        [[nodiscard]] element at(const core::size index) const noexcept {
            return std::apply([&](const auto &...child) { return calculate<code, element>(child.at(index)...); },
                              children);
        }

    private:
        /// Index of the first array operand; rank > 0 guarantees there is one.
        static constexpr core::size lead = [] {
            constexpr bool arrays[] = {(operands::rank > 0)...};
            core::size index = 0;
            while (!arrays[index])
                ++index;
            return index;
        }();
    };

    /// @brief Every node is a lazy expression.
    template <core::Code code, typename... operands> struct Deferred<Node<code, operands...>> : std::true_type {};

}
