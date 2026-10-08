#pragma once

#include "array/calculate.hpp"
#include "array/deferred.hpp"
#include "array/methods.hpp"
#include "array/shape.hpp"
#include "core/assert.hpp"
#include "core/drivers/device.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/code.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sci::array {

    using core::drivers::Device;

    // Interior node: code applied to its operands.
    template <core::Code code, typename... operands>
    struct Node : Methods<Node<code, operands...>> {
        using element = std::tuple_element_t<0, std::tuple<operands...>>::element;
        static constexpr core::size rank = std::max({operands::rank...});

        static_assert((std::is_same_v<typename operands::element, element> && ...), "operands must share an element type");
        static_assert(((operands::rank == 0 || operands::rank == rank) && ...), "operands must share a rank");
        static_assert(rank > 0, "an expression needs at least one array operand");

        std::tuple<operands...> children;

        explicit Node(operands... items) : children(std::move(items)...) {}

        // Operands must agree on their shape (no broadcasting yet, apart from scalars): the first array
        // operand gives it, and every other array operand must match.
        [[nodiscard]] Shape<rank> shape() const noexcept {
            const Shape<rank> result = std::get<lead>(children).shape();
            ASSERT(std::apply([&](const auto&... child) { return (matches(child, result) && ...); }, children));
            return result;
        }

        // The first device found among the leaves decides where the expression runs.
        [[nodiscard]] Device* device() const noexcept {
            Device* found = nullptr;
            std::apply([&](const auto&... child) { ((found = found ? found : child.device()), ...); }, children);
            return found;
        }

        [[nodiscard]] core::uint8 record(core::Chunk& chunk) const noexcept {
            return std::apply([&](const auto&... child) {
                const core::uint8 ids[] = {child.record(chunk)...};
                if constexpr (sizeof...(operands) == 1) {
                    return chunk.emit(code, ids[0]);
                } else if constexpr (sizeof...(operands) == 2) {
                    return chunk.emit(code, ids[0], ids[1]);
                } else {
                    return chunk.emit(code, ids[0], ids[1], ids[2]);
                }
            }, children);
        }

        [[nodiscard]] element at(const core::size index) const noexcept {
            return std::apply([&](const auto&... child) { return calculate<code, element>(child.at(index)...); }, children);
        }

    private:
        // Index of the first array operand; rank > 0 guarantees there is one.
        static constexpr core::size lead = [] {
            constexpr bool arrays[] = {(operands::rank > 0)...};
            core::size index = 0;
            while (!arrays[index]) ++index;
            return index;
        }();

        // Scalars match any shape.
        template <typename child>
        static bool matches(const child& item, const Shape<rank>& expected) noexcept {
            if constexpr (child::rank > 0) {
                return item.shape() == expected;
            } else {
                return true;
            }
        }
    };

    template <core::Code code, typename... operands>
    struct Deferred<Node<code, operands...>> : std::true_type {};

}
