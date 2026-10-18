#pragma once

#include "array/calculate.hpp"
#include "array/deferred.hpp"
#include "array/dense.hpp"
#include "array/literal.hpp"
#include "array/methods.hpp"
#include "array/node.hpp"
#include "array/owner.hpp"
#include "array/reference.hpp"
#include "core/assert.hpp"
#include "core/drivers/processor/dispatch.hpp"
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
// to an Array (or reducing it, or calling copy()) runs the whole tree in a single pass on the
// device the arrays live on. Element types the engine does not cover fall back to a plain loop.
//
// An expression answers the same questions an array does (shape(), size(), sum(), mean(),
// minimum(), maximum(), copy()), so `auto c = a * b;` behaves like an array that is computed when it
// is used.
// It refers to the arrays it was built from, so it must not outlive them; temporary arrays
// (`f() + a`) are moved into the expression and kept alive.

namespace sci::array {

    using core::drivers::Device;
    using core::drivers::Target;

    template <typename type, core::size order>
    class Array;

    // A lazy expression tree (including a bare reference to an array).
    template <typename type>
    concept expression = Deferred<std::remove_cvref_t<type>>::value;

    // An Array.
    template <typename type>
    concept tensor = Dense<std::remove_cvref_t<type>>::value;

    // Anything elementwise operations accept as an array argument.
    template <typename type>
    concept operand = expression<type> || tensor<type>;

    template <typename type>
    concept scalar = std::is_arithmetic_v<std::remove_cvref_t<type>>;

    // Turns an operand into an expression leaf: arrays by reference (or by ownership when they are
    // temporaries), expressions by value.
    template <operand type>
    auto capture(type&& item) {
        using bare = std::remove_cvref_t<type>;
        if constexpr (expression<type>) {
            return bare(std::forward<type>(item));
        } else if constexpr (std::is_lvalue_reference_v<type>) {
            return Reference<typename bare::element, bare::rank>(&item);
        } else {
            return Owner<typename bare::element, bare::rank>(std::make_shared<const bare>(std::forward<type>(item)));
        }
    }

    template <typename type>
    using captured = decltype(capture(std::declval<type>()));

    template <core::Code code, operand type>
    auto apply(type&& item) {
        return Node<code, captured<type>>(capture(std::forward<type>(item)));
    }

    // Scalars on either side take the element type of the array side.
    template <core::Code code, typename left, typename right>
    auto apply(left&& first, right&& second) {
        if constexpr (scalar<left>) {
            using element = std::remove_cvref_t<right>::element;
            return Node<code, Literal<element>, captured<right>>(Literal<element>{static_cast<element>(first)}, capture(std::forward<right>(second)));
        } else if constexpr (scalar<right>) {
            using element = std::remove_cvref_t<left>::element;
            return Node<code, captured<left>, Literal<element>>(capture(std::forward<left>(first)), Literal<element>{static_cast<element>(second)});
        } else {
            static_assert(std::is_same_v<typename std::remove_cvref_t<left>::element, typename std::remove_cvref_t<right>::element>, "operands must share an element type");
            return Node<code, captured<left>, captured<right>>(capture(std::forward<left>(first)), capture(std::forward<right>(second)));
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
        return apply<core::Code::subtract>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto operator*(left&& first, right&& second) {
        return apply<core::Code::multiply>(std::forward<left>(first), std::forward<right>(second));
    }

    template <typename left, typename right>
    requires compatible<left, right>
    auto operator/(left&& first, right&& second) {
        return apply<core::Code::divide>(std::forward<left>(first), std::forward<right>(second));
    }

    template <operand type>
    auto operator-(type&& item) {
        return apply<core::Code::negate>(std::forward<type>(item));
    }

    // Writes the expression into target, which must already have the expression's shape. The target
    // may appear in the expression (every kernel is elementwise). Runs on the target's device.
    template <typename element, core::size rank, typename source>
    void assign(Array<element, rank>& target, const source& item) {
        ASSERT(target.shape() == item.shape());
        if constexpr (core::native<element>) {
            core::Chunk chunk(core::Traits<element>::precision);
            chunk.output(item.record(chunk));
            if (chunk.valid()) {
                // The device that runs the chunk orders its own work; another device the inputs live on
                // has to finish first.
                Device* origin = item.device();
                if (Device* device = target.device()) {
                    if (origin && origin != device) origin->synchronize();
                    device->execute(chunk, target.data(), target.size());
                } else {
                    if (origin) origin->synchronize();
                    core::drivers::processor::Dispatch::global().execute(chunk, target.data(), target.size());
                }
                return;
            }
        }
        if (Device* device = target.device()) device->synchronize();
        if (Device* device = item.device()) device->synchronize();
        element* destination = target.data();
        const core::size count = target.size();
        for (core::size index = 0; index < count; ++index) {
            destination[index] = item.at(index);
        }
    }

    // Sum, minimum or maximum of an array or expression, in a single pass, on the device its arrays live on.
    template <typename source>
    typename source::element reduce(const source& input, const core::Fold fold) {
        using element = source::element;
        const auto item = capture(input);
        const core::size count = item.shape().total();

        if constexpr (core::native<element>) {
            core::Chunk chunk(core::Traits<element>::precision);
            chunk.reduce(item.record(chunk));
            if (chunk.valid()) {
                if (Device* device = item.device()) {
                    return static_cast<element>(device->reduce(chunk, fold, count));
                }
                return static_cast<element>(core::drivers::processor::Dispatch::global().reduce(chunk, fold, count));
            }
        }

        if (Device* device = item.device()) device->synchronize();
        if (count == 0) return element{};
        element result = item.at(0);
        for (core::size index = 1; index < count; ++index) {
            const element value = item.at(index);
            switch (fold) {
                case core::Fold::sum:
                    result += value;
                    break;
                case core::Fold::minimum:
                    result = calculate<core::Code::minimum, element>(result, value);
                    break;
                default:
                    result = calculate<core::Code::maximum, element>(result, value);
                    break;
            }
        }
        return result;
    }

}
