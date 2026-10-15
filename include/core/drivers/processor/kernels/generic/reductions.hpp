#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    // Four independent accumulators hide the add latency and give a partly pairwise summation order.
    template <typename pack>
    pack::element sum(const typename pack::element* source, const size count) noexcept {
        using type = pack::element;
        constexpr size width = pack::width;

        auto first = pack::splat(type{0});
        auto second = first;
        auto third = first;
        auto fourth = first;

        size index = 0;
        for (; index + 4 * width <= count; index += 4 * width) {
            first = pack::add(first, pack::load(source + index));
            second = pack::add(second, pack::load(source + index + width));
            third = pack::add(third, pack::load(source + index + 2 * width));
            fourth = pack::add(fourth, pack::load(source + index + 3 * width));
        }
        for (; index + width <= count; index += width) {
            first = pack::add(first, pack::load(source + index));
        }
        first = pack::add(pack::add(first, second), pack::add(third, fourth));

        type lanes[width];
        pack::store(lanes, first);
        type result = type{0};
        for (size lane = 0; lane < width; ++lane) {
            result += lanes[lane];
        }
        for (; index < count; ++index) {
            result += source[index];
        }
        return result;
    }

    // sum(left * right) with four fused multiply-add accumulators: the multiply never leaves registers.
    template <typename pack>
    pack::element dot(const typename pack::element* left, const typename pack::element* right, const size count) noexcept {
        using type = pack::element;
        using narrow = pack::narrow;
        constexpr size width = pack::width;

        auto first = pack::splat(type{0});
        auto second = first;
        auto third = first;
        auto fourth = first;

        size index = 0;
        for (; index + 4 * width <= count; index += 4 * width) {
            first = pack::fuse(pack::load(left + index), pack::load(right + index), first);
            second = pack::fuse(pack::load(left + index + width), pack::load(right + index + width), second);
            third = pack::fuse(pack::load(left + index + 2 * width), pack::load(right + index + 2 * width), third);
            fourth = pack::fuse(pack::load(left + index + 3 * width), pack::load(right + index + 3 * width), fourth);
        }
        for (; index + width <= count; index += width) {
            first = pack::fuse(pack::load(left + index), pack::load(right + index), first);
        }
        first = pack::add(pack::add(first, second), pack::add(third, fourth));

        type lanes[width];
        pack::store(lanes, first);
        type result = type{0};
        for (size lane = 0; lane < width; ++lane) {
            result += lanes[lane];
        }
        for (; index < count; ++index) {
            result = narrow::fuse(left[index], right[index], result);
        }
        return result;
    }

    template <typename pack, typename operation>
    pack::element extreme(const typename pack::element* source, const size count) noexcept {
        using type = pack::element;
        using narrow = pack::narrow;
        constexpr size width = pack::width;

        type result = source[0];
        size index = 0;
        if (count >= 4 * width) {
            auto first = pack::load(source);
            auto second = pack::load(source + width);
            auto third = pack::load(source + 2 * width);
            auto fourth = pack::load(source + 3 * width);
            for (index = 4 * width; index + 4 * width <= count; index += 4 * width) {
                first = operation::template apply<pack>(first, pack::load(source + index));
                second = operation::template apply<pack>(second, pack::load(source + index + width));
                third = operation::template apply<pack>(third, pack::load(source + index + 2 * width));
                fourth = operation::template apply<pack>(fourth, pack::load(source + index + 3 * width));
            }
            first = operation::template apply<pack>(operation::template apply<pack>(first, second), operation::template apply<pack>(third, fourth));

            type lanes[width];
            pack::store(lanes, first);
            for (size lane = 0; lane < width; ++lane) {
                result = operation::template apply<narrow>(result, lanes[lane]);
            }
        }
        for (; index < count; ++index) {
            result = operation::template apply<narrow>(result, source[index]);
        }
        return result;
    }

}
