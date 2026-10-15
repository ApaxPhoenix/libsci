#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    template <typename pack, typename operation>
    void unary(const typename pack::element* const* sources, typename pack::element* target, const size count) noexcept {
        using narrow = pack::narrow;
        const auto* first = sources[0];
        size index = 0;
        for (; index + pack::width <= count; index += pack::width) {
            pack::store(target + index, operation::template apply<pack>(pack::load(first + index)));
        }
        for (; index < count; ++index) {
            target[index] = operation::template apply<narrow>(first[index]);
        }
    }

    template <typename pack, typename operation>
    void binary(const typename pack::element* const* sources, typename pack::element* target, const size count) noexcept {
        using narrow = pack::narrow;
        const auto* first = sources[0];
        const auto* second = sources[1];
        size index = 0;
        for (; index + 2 * pack::width <= count; index += 2 * pack::width) {
            const auto low = operation::template apply<pack>(pack::load(first + index), pack::load(second + index));
            const auto high = operation::template apply<pack>(pack::load(first + index + pack::width), pack::load(second + index + pack::width));
            pack::store(target + index, low);
            pack::store(target + index + pack::width, high);
        }
        for (; index + pack::width <= count; index += pack::width) {
            pack::store(target + index, operation::template apply<pack>(pack::load(first + index), pack::load(second + index)));
        }
        for (; index < count; ++index) {
            target[index] = operation::template apply<narrow>(first[index], second[index]);
        }
    }

    template <typename pack>
    void ternary(const typename pack::element* const* sources, typename pack::element* target, const size count) noexcept {
        using narrow = pack::narrow;
        const auto* first = sources[0];
        const auto* second = sources[1];
        const auto* third = sources[2];
        size index = 0;
        for (; index + 2 * pack::width <= count; index += 2 * pack::width) {
            const auto low = pack::fuse(pack::load(first + index), pack::load(second + index), pack::load(third + index));
            const auto high = pack::fuse(pack::load(first + index + pack::width), pack::load(second + index + pack::width), pack::load(third + index + pack::width));
            pack::store(target + index, low);
            pack::store(target + index + pack::width, high);
        }
        for (; index + pack::width <= count; index += pack::width) {
            pack::store(target + index, pack::fuse(pack::load(first + index), pack::load(second + index), pack::load(third + index)));
        }
        for (; index < count; ++index) {
            target[index] = narrow::fuse(first[index], second[index], third[index]);
        }
    }

}
