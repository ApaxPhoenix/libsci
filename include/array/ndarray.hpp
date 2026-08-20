#pragma once

#include "array/shape.hpp"
#include "array/storage.hpp"
#include "array/stride.hpp"
#include "core/assert.hpp"
#include "core/drivers/device.hpp"
#include "core/memory/arena.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <initializer_list>
#include <ostream>

namespace sci::array {

    template <typename element, core::size rank>
    class ndarray {
        storage<element> store{};
        shape<rank> bounds{};
        stride<rank> layout{};

    public:
        ndarray() = default;

        explicit ndarray(shape<rank> extents)
            : store(extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {}

        ndarray(core::drivers::device& device, shape<rank> extents)
            : store(device, extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {}

        ndarray(core::memory::arena& arena, shape<rank> extents)
            : store(arena, extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {}

        ndarray(shape<rank> extents, const element& value)
            : store(extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {
            std::fill(begin(), end(), value);
        }

        ndarray(core::drivers::device& device, shape<rank> extents, const element& value)
            : store(device, extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {
            if (device.target() == core::drivers::target::cpu) {
                std::fill(begin(), end(), value);
            }
        }

        ndarray(shape<rank> extents, std::initializer_list<element> values)
            : store(extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {
            std::copy(values.begin(), values.end(), begin());
        }

        ndarray(core::memory::arena& arena, shape<rank> extents, std::initializer_list<element> values)
            : store(arena, extents.total()), bounds(extents), layout(stride<rank>::contiguous(extents)) {
            std::copy(values.begin(), values.end(), begin());
        }

        [[nodiscard]] element* data() noexcept { return store.data(); }
        [[nodiscard]] const element* data() const noexcept { return store.data(); }

        [[nodiscard]] shape<rank> extents() const noexcept { return bounds; }
        [[nodiscard]] stride<rank> strides() const noexcept { return layout; }

        [[nodiscard]] core::size size() const noexcept { return store.size(); }
        [[nodiscard]] bool empty() const noexcept { return store.size() == 0; }
        [[nodiscard]] const core::drivers::memory& handle() const noexcept { return store.handle(); }
        [[nodiscard]] core::drivers::device* device() noexcept { return store.device(); }
        [[nodiscard]] const core::drivers::device* device() const noexcept { return store.device(); }
        [[nodiscard]] core::drivers::target target() const noexcept { return store.target(); }

        element* begin() noexcept {
            ASSERT(target() == core::drivers::target::cpu);
            return store.data();
        }

        const element* begin() const noexcept {
            ASSERT(target() == core::drivers::target::cpu);
            return store.data();
        }

        element* end() noexcept {
            ASSERT(target() == core::drivers::target::cpu);
            return store.data() + store.size();
        }

        const element* end() const noexcept {
            ASSERT(target() == core::drivers::target::cpu);
            return store.data() + store.size();
        }

        template <typename... arguments>
        requires (sizeof...(arguments) == rank)
        [[nodiscard]] element& operator[](arguments... indices) {
            ASSERT(target() == core::drivers::target::cpu);
            core::size list[] = { static_cast<core::size>(indices)... };
            core::size offset = 0;
            for (core::size position = 0; position < rank; ++position) {
                offset += list[position] * layout[position];
            }
            return store.data()[offset];
        }

        template <typename... arguments>
        requires (sizeof...(arguments) == rank)
        [[nodiscard]] const element& operator[](arguments... indices) const {
            ASSERT(target() == core::drivers::target::cpu);
            core::size list[] = { static_cast<core::size>(indices)... };
            core::size offset = 0;
            for (core::size position = 0; position < rank; ++position) {
                offset += list[position] * layout[position];
            }
            return store.data()[offset];
        }

        [[nodiscard]] element& operator[](core::size index) {
            ASSERT(target() == core::drivers::target::cpu);
            return store.data()[index];
        }

        [[nodiscard]] const element& operator[](core::size index) const {
            ASSERT(target() == core::drivers::target::cpu);
            return store.data()[index];
        }

        ndarray to(core::drivers::device& target_device) const {
            ndarray result(target_device, bounds);
            if (target() == core::drivers::target::cpu && target_device.target() == core::drivers::target::cpu) {
                std::copy(begin(), end(), result.begin());
            }
            return result;
        }

        ndarray to(core::memory::arena& arena) const {
            ndarray result(arena, bounds);
            if (target() == core::drivers::target::cpu) {
                std::copy(begin(), end(), result.begin());
            }
            return result;
        }

        ndarray& operator+=(const ndarray& other) {
            ASSERT(target() == other.target());
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const element* __restrict src = other.store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] += src[position];
            }
            return *this;
        }

        ndarray& operator+=(const element& scalar) {
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] += scalar;
            }
            return *this;
        }

        ndarray& operator-=(const ndarray& other) {
            ASSERT(target() == other.target());
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const element* __restrict src = other.store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] -= src[position];
            }
            return *this;
        }

        ndarray& operator-=(const element& scalar) {
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] -= scalar;
            }
            return *this;
        }

        ndarray& operator*=(const ndarray& other) {
            ASSERT(target() == other.target());
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const element* __restrict src = other.store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] *= src[position];
            }
            return *this;
        }

        ndarray& operator*=(const element& scalar) {
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] *= scalar;
            }
            return *this;
        }

        ndarray& operator/=(const ndarray& other) {
            ASSERT(target() == other.target());
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const element* __restrict src = other.store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] /= src[position];
            }
            return *this;
        }

        ndarray& operator/=(const element& scalar) {
            ASSERT(target() == core::drivers::target::cpu);
            element* __restrict dst = store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] /= scalar;
            }
            return *this;
        }

        ndarray operator-() const {
            ASSERT(target() == core::drivers::target::cpu);
            ndarray result(bounds);
            const element* __restrict src = store.data();
            element* __restrict dst = result.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                dst[position] = -src[position];
            }
            return result;
        }

        ndarray operator+() const {
            return *this;
        }

        [[nodiscard]] element sum() const {
            ASSERT(target() == core::drivers::target::cpu);
            element result{};
            const element* __restrict src = store.data();
            const core::size total = store.size();

            #if defined(__clang__) || defined(__GNUG__)
            #pragma clang loop vectorize(enable) interleave(enable)
            #endif
            for (core::size position = 0; position < total; ++position) {
                result += src[position];
            }
            return result;
        }

        [[nodiscard]] element mean() const {
            ASSERT(target() == core::drivers::target::cpu);
            return store.size() == 0 ? element{} : sum() / static_cast<element>(store.size());
        }

        [[nodiscard]] element min() const {
            ASSERT(target() == core::drivers::target::cpu);
            return *std::min_element(begin(), end());
        }

        [[nodiscard]] element max() const {
            ASSERT(target() == core::drivers::target::cpu);
            return *std::max_element(begin(), end());
        }
    };

    template <typename element, core::size rank>
    ndarray<element, rank> operator+(ndarray<element, rank> left, const ndarray<element, rank>& right) {
        left += right;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator+(ndarray<element, rank> left, const element& scalar) {
        left += scalar;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator+(const element& scalar, ndarray<element, rank> right) {
        right += scalar;
        return right;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator-(ndarray<element, rank> left, const ndarray<element, rank>& right) {
        left -= right;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator-(ndarray<element, rank> left, const element& scalar) {
        left -= scalar;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator-(const element& scalar, const ndarray<element, rank>& right) {
        ASSERT(right.target() == core::drivers::target::cpu);
        ndarray<element, rank> result(right.extents());
        const element* __restrict src = right.data();
        element* __restrict dst = result.data();
        const core::size total = right.size();

        #if defined(__clang__) || defined(__GNUG__)
        #pragma clang loop vectorize(enable) interleave(enable)
        #endif
        for (core::size position = 0; position < total; ++position) {
            dst[position] = scalar - src[position];
        }
        return result;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator*(ndarray<element, rank> left, const ndarray<element, rank>& right) {
        left *= right;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator*(ndarray<element, rank> left, const element& scalar) {
        left *= scalar;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator*(const element& scalar, ndarray<element, rank> right) {
        right *= scalar;
        return right;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator/(ndarray<element, rank> left, const ndarray<element, rank>& right) {
        left /= right;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator/(ndarray<element, rank> left, const element& scalar) {
        left /= scalar;
        return left;
    }

    template <typename element, core::size rank>
    ndarray<element, rank> operator/(const element& scalar, const ndarray<element, rank>& right) {
        ASSERT(right.target() == core::drivers::target::cpu);
        ndarray<element, rank> result(right.extents());
        const element* __restrict src = right.data();
        element* __restrict dst = result.data();
        const core::size total = right.size();

        #if defined(__clang__) || defined(__GNUG__)
        #pragma clang loop vectorize(enable) interleave(enable)
        #endif
        for (core::size position = 0; position < total; ++position) {
            dst[position] = scalar / src[position];
        }
        return result;
    }

    template <typename element, core::size rank>
    bool operator==(const ndarray<element, rank>& left, const ndarray<element, rank>& right) {
        ASSERT(left.target() == right.target());
        ASSERT(left.target() == core::drivers::target::cpu);
        if (left.size() != right.size()) {
            return false;
        }
        for (core::size position = 0; position < left.size(); ++position) {
            if (left[position] != right[position]) {
                return false;
            }
        }
        return true;
    }

    template <typename element, core::size rank>
    bool operator!=(const ndarray<element, rank>& left, const ndarray<element, rank>& right) {
        return !(left == right);
    }

    template <typename element, core::size rank>
    std::ostream& operator<<(std::ostream& stream, const ndarray<element, rank>& array) {
        ASSERT(array.target() == core::drivers::target::cpu);
        stream << "[";
        for (core::size position = 0; position < array.size(); ++position) {
            stream << array[position] << (position + 1 == array.size() ? "" : ", ");
        }
        return stream << "]";
    }

}