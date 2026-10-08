#pragma once

#include "array/calculate.hpp"
#include "array/deferred.hpp"
#include "array/dense.hpp"
#include "array/elementof.hpp"
#include "array/literal.hpp"
#include "array/methods.hpp"
#include "array/node.hpp"
#include "array/owner.hpp"
#include "array/reference.hpp"
#include "core/assert.hpp"
#include "core/drivers/cpu/dispatch.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/target.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/code.hpp"
#include "core/bytecode/traits.hpp"
#include "core/types.hpp"

#include <memory>
#include <type_traits>
#include <utility>

// Lazy elementwise expressions. `a * b + c` builds a small tree instead of temporaries; assigning it
// to an NDArray (or reducing it, or calling copy()) runs the whole tree in a single pass on the
// device the arrays live on. Element types the engine does not cover fall back to a plain loop.
//
// An expression answers the same questions an array does (shape(), size(), sum(), mean(), min(),
// max(), copy()), so `auto c = a * b;` behaves like an array that is computed when it is used.
// It refers to the arrays it was built from, so it must not outlive them; temporary arrays
// (`f() + a`) are moved into the e   xpression and kept alive.

namespace sci::array {

    using core::drivers::Device;
    using core::drivers::Target;

    template <typename element, core::size rank>
    class NDArray;

    // A lazy expression tree (including a bare reference to an array).
    template <typename type>
    concept expression = Deferred<std::remove_cvref_t<type>>::value;

    // An NDArray.
    template <typename type>
    concept tensor = Dense<std::remove_cvref_t<type>>::value;

    // Anything elementwise operations accept as an array argument.
    template <typename type>
    concept operand = expression<type> || tensor<type>;

    template <typename type>
    concept scalar = std::is_arithmetic_v<std::remove_cvref_t<type>>;

    // Number of dimensions of an array or expression.
    template <typename type>
    inline constexpr core::size rank_v = [] {
        if constexpr (tensor<type>) {
            return Dense<std::remove_cvref_t<type>>::order;
        } else {
            return std::remove_cvref_t<type>::rank;
        }
    }();

    // Waits for queued GPU work on a device before the host touches its memory.
    inline void synchronize(Device* device) noexcept {
        if (device && device->target() == Target::gpu) device->synchronize();
    }

    // Turns an operand into an expression leaf: arrays by reference (or by ownership when they are
    // temporaries), expressions by value.
    template <operand type>
    auto wrap(type&& item) {
        using bare = std::remove_cvref_t<type>;
        if constexpr (expression<type>) {
            return bare(std::forward<type>(item));
        } else if constexpr (std::is_lvalue_reference_v<type>) {
            return Reference<element_t<type>, rank_v<type>>(&item);
        } else {
            return Owner<element_t<type>, rank_v<type>>(std::make_shared<const bare>(std::forward<type>(item)));
        }
    }

    template <typename type>
    using wrapped = decltype(wrap(std::declval<type>()));

    template <core::Code code, operand type>
    auto apply(type&& item) {
        return Node<code, wrapped<type>>(wrap(std::forward<type>(item)));
    }

    // Scalars on either side take the element type of the array side.
    template <core::Code code, typename left, typename right>
    auto apply(left&& first, right&& second) {
        if constexpr (scalar<left>) {
            using element = element_t<right>;
            return Node<code, Literal<element>, wrapped<right>>(Literal<element>{static_cast<element>(first)}, wrap(std::forward<right>(second)));
        } else if constexpr (scalar<right>) {
            using element = element_t<left>;
            return Node<code, wrapped<left>, Literal<element>>(wrap(std::forward<left>(first)), Literal<element>{static_cast<element>(second)});
        } else {
            static_assert(std::is_same_v<element_t<left>, element_t<right>>, "operands must share an element type");
            return Node<code, wrapped<left>, wrapped<right>>(wrap(std::forward<left>(first)), wrap(std::forward<right>(second)));
        }
    }

    template <typename left, typename right>
    concept compatible = (operand<left> && (operand<right> || scalar<right>)) || (scalar<left> && operand<right>);

    template <typename left, typename right>
    requires compatible<left, right>
    auto operator+(left&& first, right&& second) {
        return apply<core::Code::add>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto operator-(left&& first, right&& second) {
        return apply<core::Code::sub>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto operator*(left&& first, right&& second) {
        return apply<core::Code::mul>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto operator/(left&& first, right&& second) {
        return apply<core::Code::div>(std::forward<left>(first), std::forward<right>(second));
    }

    template <operand type>
    auto operator-(type&& item) {
        return apply<core::Code::neg>(std::forward<type>(item));
    }

    // Writes the expression into target, which must already have the expression's shape. The target
    // may appear in the expression (every kernel is elementwise). Runs on the target's device.
    template <typename element, core::size rank, typename source>
    void assign(NDArray<element, rank>& target, const source& item) {
        ASSERT(target.shape() == item.shape());
        if constexpr (core::native<element>) {
            core::Chunk chunk(core::Traits<element>::type);
            chunk.output(item.record(chunk));
            if (chunk.valid()) {
                if (Device* device = target.device()) {
                    synchronize(item.device() != device ? item.device() : nullptr);
                    device->execute(chunk, target.data(), target.size());
                } else {
                    synchronize(item.device());
                    core::drivers::cpu::Dispatch::global().execute(chunk, target.data(), target.size());
                }
                return;
            }
        }
        synchronize(target.device());
        synchronize(item.device());
        element* destination = target.data();
        const core::size count = target.size();
        for (core::size index = 0; index < count; ++index) {
            destination[index] = item.at(index);
        }
    }

    // Sum, min or max of an array or expression, in a single pass, on the device its arrays live on.
    template <typename source>
    element_t<source> reduce(const source& input, const core::Fold fold) {
        using element = element_t<source>;
        const auto item = wrap(input);
        const core::size count = item.shape().total();

        if constexpr (core::native<element>) {
            core::Chunk chunk(core::Traits<element>::type);
            chunk.reduce(item.record(chunk));
            if (chunk.valid()) {
                if (Device* device = item.device()) {
                    return static_cast<element>(device->reduce(chunk, fold, count));
                }
                return static_cast<element>(core::drivers::cpu::Dispatch::global().reduce(chunk, fold, count));
            }
        }

        synchronize(item.device());
        if (count == 0) return element{};
        element result = item.at(0);
        for (core::size index = 1; index < count; ++index) {
            const element value = item.at(index);
            switch (fold) {
                case core::Fold::sum:
                    result += value;
                    break;
                case core::Fold::min:
                    result = calculate<core::Code::min, element>(result, value);
                    break;
                default:
                    result = calculate<core::Code::max, element>(result, value);
                    break;
            }
        }
        return result;
    }

}
