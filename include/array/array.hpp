#pragma once

#include "array/expression.hpp"
#include "array/methods.hpp"
#include "array/shape.hpp"
#include "array/storage.hpp"
#include "array/stride.hpp"
#include "array/uninitialized.hpp"
#include "core/assert.hpp"
#include "core/drivers/device.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>

namespace sci::array {

    // An n-dimensional, contiguous, row-major array, in the spirit of numpy.ndarray:
    //
    //     Array<float, 2> a({2, 3});           // zeros
    //     Array<float, 2> b({2, 3}, 1.5f);     // filled
    //     Array<float, 2> c = a * b + 1.0f;    // computed in a single loop
    //     float total = c.sum();
    //
    // Arrays can live on a Device (a.to(graphics)) or in an Arena; expressions then run there, and
    // reading an element on the host waits for the GPU first, so a GPU array reads like any other
    // array. size(), sum(), mean(), minimum(), maximum() and copy() come from Methods, which
    // expressions share.
    template <typename type, core::size order>
    class Array : public Methods<Array<type, order>> {
    public:
        using element = type;

        // Number of dimensions (numpy's ndim).
        static constexpr core::size rank = order;

    private:
        Storage<element> store{};
        Shape<rank> bounds{};
        Stride<rank> layout{};

    public:
        Array() = default;

        // Every element set to value, zeros unless one is given.
        explicit Array(Shape<rank> shape, const element& value = element{})
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            std::fill_n(data(), store.size(), value);
        }

        // The same on a Device or in an Arena.
        template <location place>
        Array(place& where, Shape<rank> shape, const element& value = element{})
            : store(where, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            std::fill_n(data(), store.size(), value);
        }

        // Values in row-major order; missing trailing values are zero.
        Array(Shape<rank> shape, std::initializer_list<element> values)
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            ASSERT(values.size() <= store.size());
            std::fill(std::copy(values.begin(), values.end(), data()), data() + store.size(), element{});
        }

        template <location place>
        Array(place& where, Shape<rank> shape, std::initializer_list<element> values)
            : store(where, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            ASSERT(values.size() <= store.size());
            std::fill(std::copy(values.begin(), values.end(), data()), data() + store.size(), element{});
        }

        // Uninitialised (see sci::empty).
        Array(Shape<rank> shape, Uninitialized)
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        template <location place>
        Array(place& where, Shape<rank> shape, Uninitialized)
            : store(where, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        // Computes an expression in a single pass, on the device its arrays live on. Implicit on
        // purpose, so `Array<float, 2> c = a * b;` works as it does in numpy.
        template <expression source>
        requires (std::is_same_v<typename source::element, element> && source::rank == rank)
        Array(const source& item) // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
            : store(item.device() ? Storage<element>(*item.device(), item.shape().total()) : Storage<element>(item.shape().total())),
              bounds(item.shape()), layout(Stride<rank>::contiguous(item.shape())) {
            assign(*this, item);
        }

        Array(Array&&) noexcept = default;
        Array& operator=(Array&&) noexcept = default;

        // Same shape: computed in place (the array may appear in the expression). Otherwise, the
        // array is replaced by a new one.
        template <expression source>
        requires (std::is_same_v<typename source::element, element> && source::rank == rank)
        Array& operator=(const source& item) {
            if (store.size() == 0 || bounds != item.shape()) {
                Array result(item);
                *this = std::move(result);
            } else {
                assign(*this, item);
            }
            return *this;
        }

        [[nodiscard]] Shape<rank> shape() const noexcept { return bounds; }
        [[nodiscard]] Stride<rank> strides() const noexcept { return layout; }

        // The device the array lives on (null for host and arena memory); the array does not own it.
        [[nodiscard]] Device* device() const noexcept { return store.device(); }

        // Raw element pointer. For GPU arrays, call device()->synchronize() before using it on the host.
        [[nodiscard]] element* data() noexcept { return store.data(); }
        [[nodiscard]] const element* data() const noexcept { return store.data(); }

        element* begin() {
            store.synchronize();
            return data();
        }

        const element* begin() const {
            store.synchronize();
            return data();
        }

        element* end() {
            store.synchronize();
            return data() + store.size();
        }

        const element* end() const {
            store.synchronize();
            return data() + store.size();
        }

        // a[i, j] (multi-dimensional) or a[i] (flat).
        template <typename... arguments>
        requires (sizeof...(arguments) == rank && rank > 1)
        [[nodiscard]] element& operator[](arguments... indices) {
            store.synchronize();
            return data()[layout.offset({static_cast<core::size>(indices)...})];
        }

        template <typename... arguments>
        requires (sizeof...(arguments) == rank && rank > 1)
        [[nodiscard]] const element& operator[](arguments... indices) const {
            store.synchronize();
            return data()[layout.offset({static_cast<core::size>(indices)...})];
        }

        [[nodiscard]] element& operator[](const core::size index) {
            store.synchronize();
            return data()[index];
        }

        [[nodiscard]] const element& operator[](const core::size index) const {
            store.synchronize();
            return data()[index];
        }

        // A copy on a Device (or from the GPU back to a CPU device), or in an Arena.
        template <location place>
        [[nodiscard]] Array to(place& where) const {
            Array result(where, bounds, Uninitialized{});
            store.synchronize();
            std::copy_n(data(), store.size(), result.data());
            return result;
        }

        // A copy in plain host memory, wherever this array lives (like torch.Tensor.cpu()).
        [[nodiscard]] Array host() const {
            Array result(bounds, Uninitialized{});
            store.synchronize();
            std::copy_n(data(), store.size(), result.data());
            return result;
        }

        // Sets every element (numpy.ndarray.fill).
        void fill(const element& value) {
            store.synchronize();
            std::fill_n(data(), store.size(), value);
        }

        template <typename source>
        requires compatible<Array&, source>
        Array& operator+=(source&& other) {
            assign(*this, *this + std::forward<source>(other));
            return *this;
        }

        template <typename source>
        requires compatible<Array&, source>
        Array& operator-=(source&& other) {
            assign(*this, *this - std::forward<source>(other));
            return *this;
        }

        template <typename source>
        requires compatible<Array&, source>
        Array& operator*=(source&& other) {
            assign(*this, *this * std::forward<source>(other));
            return *this;
        }

        template <typename source>
        requires compatible<Array&, source>
        Array& operator/=(source&& other) {
            assign(*this, *this / std::forward<source>(other));
            return *this;
        }

    };

    // Same shape and every element equal; `!=` is derived from it.
    template <typename element, core::size rank>
    bool operator==(const Array<element, rank>& left, const Array<element, rank>& right) {
        if (left.shape() != right.shape()) return false;
        return std::equal(left.begin(), left.end(), right.begin());
    }

    // Prints the elements in row-major order: [1, 2, 3].
    template <typename element, core::size rank>
    std::ostream& operator<<(std::ostream& stream, const Array<element, rank>& array) {
        stream << "[";
        for (core::size position = 0; position < array.size(); ++position) {
            stream << array[position] << (position + 1 == array.size() ? "" : ", ");
        }
        return stream << "]";
    }

}
