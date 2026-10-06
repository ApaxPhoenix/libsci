#pragma once

#include "array/expression.hpp"
#include "array/shape.hpp"
#include "array/storage.hpp"
#include "array/stride.hpp"
#include "array/uninitialized.hpp"
#include "core/assert.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/target.hpp"
#include "core/memory/arena.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>

namespace sci::array {

    // An n-dimensional, contiguous, row-major array, in the spirit of numpy.ndarray:
    //
    //     NDArray<float, 2> a({2, 3});           // zeros
    //     NDArray<float, 2> b({2, 3}, 1.5f);     // filled
    //     NDArray<float, 2> c = a * b + 1.0f;    // computed in a single loop
    //     float total = c.sum();
    //
    // Arrays can live on a Device (a.to(gpu)); expressions then run there, and reading an element on
    // the host waits for the GPU first, so a GPU array reads like any other array.
    template <typename element, core::size rank>
    class NDArray {
        Storage<element> store{};
        Shape<rank> bounds{};
        Stride<rank> layout{};

    public:
        NDArray() = default;

        // Zeros.
        explicit NDArray(Shape<rank> shape)
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        NDArray(Device& device, Shape<rank> shape)
            : store(device, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        NDArray(core::memory::Arena& arena, Shape<rank> shape)
            : store(arena, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        // Every element set to value.
        NDArray(Shape<rank> shape, const element& value)
            : store(shape.total(), Uninitialized{}), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            std::fill_n(data(), size(), value);
        }

        NDArray(Device& device, Shape<rank> shape, const element& value)
            : store(device, shape.total(), Uninitialized{}), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            std::fill_n(data(), size(), value);
        }

        // Values in row-major order; missing trailing values are zero.
        NDArray(Shape<rank> shape, std::initializer_list<element> values)
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            ASSERT(values.size() <= size());
            std::copy(values.begin(), values.end(), data());
        }

        NDArray(core::memory::Arena& arena, Shape<rank> shape, std::initializer_list<element> values)
            : store(arena, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            ASSERT(values.size() <= size());
            std::copy(values.begin(), values.end(), data());
        }

        // Uninitialised (see sci::empty).
        NDArray(Shape<rank> shape, Uninitialized)
            : store(shape.total(), Uninitialized{}), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        NDArray(Device& device, Shape<rank> shape, Uninitialized)
            : store(device, shape.total(), Uninitialized{}), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        // Computes an expression in a single pass, on the device its arrays live on.
        template <expression source>
        requires (std::is_same_v<element_t<source>, element> && rank_v<source> == rank)
        NDArray(const source& item)
            : store(allot(item)), bounds(item.shape()), layout(Stride<rank>::contiguous(item.shape())) {
            assign(*this, item);
        }

        NDArray(NDArray&&) noexcept = default;
        NDArray& operator=(NDArray&&) noexcept = default;

        // Same shape: computed in place (the array may appear in the expression). Otherwise, the
        // array is replaced by a new one.
        template <expression source>
        requires (std::is_same_v<element_t<source>, element> && rank_v<source> == rank)
        NDArray& operator=(const source& item) {
            if (store.size() == 0 || bounds != item.shape()) {
                NDArray result(item);
                *this = std::move(result);
            } else {
                assign(*this, item);
            }
            return *this;
        }

        [[nodiscard]] Shape<rank> shape() const noexcept { return bounds; }
        [[nodiscard]] Stride<rank> strides() const noexcept { return layout; }
        [[nodiscard]] core::size size() const noexcept { return store.size(); }
        [[nodiscard]] static constexpr core::size ndim() noexcept { return rank; }
        [[nodiscard]] bool empty() const noexcept { return store.size() == 0; }

        // The device the array lives on (null for plain host memory); the array does not own it.
        [[nodiscard]] Device* device() const noexcept { return store.device(); }
        [[nodiscard]] Target target() const noexcept { return store.target(); }

        // Raw element pointer. For GPU arrays, call device()->synchronize() before using it on the host.
        [[nodiscard]] element* data() noexcept { return store.data(); }
        [[nodiscard]] const element* data() const noexcept { return store.data(); }

        element* begin() {
            wait();
            return data();
        }

        const element* begin() const {
            wait();
            return data();
        }

        element* end() {
            wait();
            return data() + size();
        }

        const element* end() const {
            wait();
            return data() + size();
        }

        // a[i, j] (multi-dimensional) or a[i] (flat).
        template <typename... arguments>
        requires (sizeof...(arguments) == rank && rank > 1)
        [[nodiscard]] element& operator[](arguments... indices) {
            wait();
            return data()[layout.offset({static_cast<core::size>(indices)...})];
        }

        template <typename... arguments>
        requires (sizeof...(arguments) == rank && rank > 1)
        [[nodiscard]] const element& operator[](arguments... indices) const {
            wait();
            return data()[layout.offset({static_cast<core::size>(indices)...})];
        }

        [[nodiscard]] element& operator[](const core::size index) {
            wait();
            return data()[index];
        }

        [[nodiscard]] const element& operator[](const core::size index) const {
            wait();
            return data()[index];
        }

        // A copy on another device (or from the GPU back to a CPU device).
        [[nodiscard]] NDArray to(Device& destination) const {
            NDArray result(destination, bounds, Uninitialized{});
            wait();
            std::copy_n(data(), size(), result.data());
            return result;
        }

        [[nodiscard]] NDArray to(core::memory::Arena& arena) const {
            NDArray result(arena, bounds);
            wait();
            std::copy_n(data(), size(), result.data());
            return result;
        }

        // A copy in plain host memory, wherever this array lives (like torch.Tensor.cpu()).
        [[nodiscard]] NDArray cpu() const {
            NDArray result(bounds, Uninitialized{});
            wait();
            std::copy_n(data(), size(), result.data());
            return result;
        }

        // A copy on the same device (numpy.ndarray.copy).
        [[nodiscard]] NDArray copy() const {
            return NDArray(wrap(*this));
        }

        // Sets every element (numpy.ndarray.fill).
        void fill(const element& value) {
            wait();
            std::fill_n(data(), size(), value);
        }

        template <typename source>
        requires compatible<NDArray&, source>
        NDArray& operator+=(source&& other) {
            assign(*this, *this + std::forward<source>(other));
            return *this;
        }

        template <typename source>
        requires compatible<NDArray&, source>
        NDArray& operator-=(source&& other) {
            assign(*this, *this - std::forward<source>(other));
            return *this;
        }

        template <typename source>
        requires compatible<NDArray&, source>
        NDArray& operator*=(source&& other) {
            assign(*this, *this * std::forward<source>(other));
            return *this;
        }

        template <typename source>
        requires compatible<NDArray&, source>
        NDArray& operator/=(source&& other) {
            assign(*this, *this / std::forward<source>(other));
            return *this;
        }

        [[nodiscard]] element sum() const { return reduce(*this, core::Fold::sum); }
        [[nodiscard]] element min() const { return reduce(*this, core::Fold::min); }
        [[nodiscard]] element max() const { return reduce(*this, core::Fold::max); }

        [[nodiscard]] element mean() const {
            return size() == 0 ? element{} : sum() / static_cast<element>(size());
        }

    private:
        // Lets the host touch the memory: waits for queued GPU work on the array's device.
        void wait() const noexcept {
            synchronize(device());
        }

        template <typename source>
        static Storage<element> allot(const source& item) {
            const core::size total = item.shape().total();
            if (Device* device = item.device()) {
                return Storage<element>(*device, total, Uninitialized{});
            }
            return Storage<element>(total, Uninitialized{});
        }
    };

    // Same shape and every element equal.
    template <typename element, core::size rank>
    bool operator==(const NDArray<element, rank>& left, const NDArray<element, rank>& right) {
        if (left.shape() != right.shape()) return false;
        return std::equal(left.begin(), left.end(), right.begin());
    }

    template <typename element, core::size rank>
    bool operator!=(const NDArray<element, rank>& left, const NDArray<element, rank>& right) {
        return !(left == right);
    }

    // Prints the elements in row-major order: [1, 2, 3].
    template <typename element, core::size rank>
    std::ostream& operator<<(std::ostream& stream, const NDArray<element, rank>& array) {
        stream << "[";
        for (core::size position = 0; position < array.size(); ++position) {
            stream << array[position] << (position + 1 == array.size() ? "" : ", ");
        }
        return stream << "]";
    }

}
