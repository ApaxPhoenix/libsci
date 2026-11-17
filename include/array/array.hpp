#pragma once

/// @file
/// @brief An n-dimensional, contiguous, row-major array, in the spirit of numpy.ndarray.

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

    /// An n-dimensional, contiguous, row-major array, in the spirit of numpy.ndarray:
    ///
    ///     Array<float, 2> a({2, 3});           // zeros
    ///     Array<float, 2> b({2, 3}, 1.5f);     // filled
    ///     Array<float, 2> c = a * b + 1.0f;    // computed in a single loop
    ///     float total = c.sum();
    ///
    /// Arrays can live on a Device (a.to(graphics)) or in an Arena; expressions then run there, and
    /// reading an element on the host waits for the GPU first, so a GPU array reads like any other
    /// array. size(), sum(), mean(), minimum(), maximum() and copy() come from Methods, which
    /// expressions share.
    template <typename type, core::size order> class Array : public Methods<Array<type, order>> {
    public:
        /// @brief The element type.
        using element = type;

        /// Number of dimensions (numpy's ndim).
        static constexpr core::size rank = order;

    private:
        /// @brief The elements.
        Storage<element> store{};
        /// @brief The extent of each dimension.
        Shape<rank> bounds{};
        /// @brief How many elements each dimension steps over.
        Stride<rank> layout{};

    public:
        Array() = default;

        /// Every element set to value, zeros unless one is given.
        /// @param shape The extent of each dimension.
        /// @param value What every element is set to.
        explicit Array(Shape<rank> shape, const element &value = element{})
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            std::fill_n(data(), store.size(), value);
        }

        /// The same on a Device or in an Arena.
        /// @param where The Device or Arena it lives on.
        /// @param shape The extent of each dimension.
        /// @param value What every element is set to.
        template <location place>
        Array(place &where, Shape<rank> shape, const element &value = element{})
            : store(where, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            std::fill_n(data(), store.size(), value);
        }

        /// Values in row-major order; missing trailing values are zero.
        /// @param shape The extent of each dimension.
        /// @param values The values, in row-major order.
        Array(Shape<rank> shape, std::initializer_list<element> values)
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            ASSERT(values.size() <= store.size());
            std::fill(std::copy(values.begin(), values.end(), data()), data() + store.size(), element{});
        }

        /// @brief Values in row-major order on a Device or in an Arena; missing trailing values are zero.
        /// @param where The Device or Arena it lives on.
        /// @param shape The extent of each dimension.
        /// @param values The values, in row-major order.
        template <location place>
        Array(place &where, Shape<rank> shape, std::initializer_list<element> values)
            : store(where, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {
            ASSERT(values.size() <= store.size());
            std::fill(std::copy(values.begin(), values.end(), data()), data() + store.size(), element{});
        }

        /// Uninitialised (see sci::empty).
        /// @param shape The extent of each dimension.
        Array(Shape<rank> shape, Uninitialized /*unused*/)
            : store(shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        /// @brief Uninitialised, on a Device or in an Arena.
        /// @param where The Device or Arena it lives on.
        /// @param shape The extent of each dimension.
        template <location place>
        Array(place &where, Shape<rank> shape, Uninitialized /*unused*/)
            : store(where, shape.total()), bounds(shape), layout(Stride<rank>::contiguous(shape)) {}

        /// Computes an expression in a single pass, on the device its arrays live on. Implicit on
        /// purpose, so `Array<float, 2> c = a * b;` works as it does in numpy.
        /// @param item The expression.
        template <expression source>
            requires(std::is_same_v<typename source::element, element> && source::rank == rank)
        Array(const source &item) // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
            : store(item.device() ? Storage<element>(*item.device(), item.shape().total())
                                  : Storage<element>(item.shape().total())),
              bounds(item.shape()), layout(Stride<rank>::contiguous(item.shape())) {
            assign(*this, item);
        }

        /// @brief Takes another array's elements, leaving it empty.
        Array(Array &&) noexcept = default;
        /// @brief Takes another array's elements, leaving it empty.
        /// @return This array.
        Array &operator=(Array &&) noexcept = default;

        /// Same shape: computed in place (the array may appear in the expression). Otherwise, the
        /// array is replaced by a new one.
        /// @param item The expression.
        /// @return This array.
        template <expression source>
            requires(std::is_same_v<typename source::element, element> && source::rank == rank)
        Array &operator=(const source &item) {
            if (store.size() == 0 || bounds != item.shape()) {
                Array result(item);
                *this = std::move(result);
            } else {
                assign(*this, item);
            }
            return *this;
        }

        /// @brief The extent of each dimension.
        /// @return The shape.
        [[nodiscard]] Shape<rank> shape() const noexcept { return bounds; }
        /// @brief How many elements each dimension steps over.
        /// @return The strides.
        [[nodiscard]] Stride<rank> strides() const noexcept { return layout; }

        /// The device the array lives on (null for host and arena memory); the array does not own it.
        /// @return The device, or null.
        [[nodiscard]] Device *device() const noexcept { return store.device(); }

        /// Raw element pointer. For GPU arrays, call device()->synchronize() before using it on the host.
        /// @return The first element.
        [[nodiscard]] element *data() noexcept { return store.data(); }
        /// @brief Raw element pointer, read-only.
        /// @return The first element.
        [[nodiscard]] const element *data() const noexcept { return store.data(); }

        /// @brief The first element, for range-for.
        /// @return A pointer to it.
        element *begin() {
            store.synchronize();
            return data();
        }

        /// @brief The first element, read-only.
        /// @return A pointer to it.
        const element *begin() const {
            store.synchronize();
            return data();
        }

        /// @brief Past the last element.
        /// @return A pointer past it.
        element *end() {
            store.synchronize();
            return data() + store.size();
        }

        /// @brief Past the last element, read-only.
        /// @return A pointer past it.
        const element *end() const {
            store.synchronize();
            return data() + store.size();
        }

        /// a[i, j] (multi-dimensional) or a[i] (flat).
        /// @param indices Its index along each dimension.
        /// @return The element.
        template <typename... arguments>
            requires(sizeof...(arguments) == rank && rank > 1)
        [[nodiscard]] element &operator[](arguments... indices) {
            store.synchronize();
            return data()[layout.offset({static_cast<core::size>(indices)...})];
        }

        /// @brief a[i, j], read-only.
        /// @param indices Its index along each dimension.
        /// @return The element.
        template <typename... arguments>
            requires(sizeof...(arguments) == rank && rank > 1)
        [[nodiscard]] const element &operator[](arguments... indices) const {
            store.synchronize();
            return data()[layout.offset({static_cast<core::size>(indices)...})];
        }

        /// @brief a[i]: an element by its flat index.
        /// @param index The element's index.
        /// @return The element.
        [[nodiscard]] element &operator[](const core::size index) {
            store.synchronize();
            return data()[index];
        }

        /// @brief a[i], read-only.
        /// @param index The element's index.
        /// @return The element.
        [[nodiscard]] const element &operator[](const core::size index) const {
            store.synchronize();
            return data()[index];
        }

        /// A copy on a Device (or from the GPU back to a CPU device), or in an Arena.
        /// @param where The Device or Arena it goes to.
        /// @return The copy.
        template <location place> [[nodiscard]] Array to(place &where) const {
            Array result(where, bounds, Uninitialized{});
            store.synchronize();
            std::copy_n(data(), store.size(), result.data());
            return result;
        }

        /// A copy in plain host memory, wherever this array lives (like torch.Tensor.cpu()).
        /// @return The copy.
        [[nodiscard]] Array host() const {
            Array result(bounds, Uninitialized{});
            store.synchronize();
            std::copy_n(data(), store.size(), result.data());
            return result;
        }

        /// Sets every element (numpy.ndarray.fill).
        /// @param value The value.
        void fill(const element &value) {
            store.synchronize();
            std::fill_n(data(), store.size(), value);
        }

        /// @brief Adds an array or expression, element by element, in place.
        /// @param other The array or expression.
        /// @return This array.
        template <typename source>
            requires compatible<Array &, source>
        Array &operator+=(source &&other) {
            assign(*this, *this + std::forward<source>(other));
            return *this;
        }

        /// @brief Subtracts an array or expression, element by element, in place.
        /// @param other The array or expression.
        /// @return This array.
        template <typename source>
            requires compatible<Array &, source>
        Array &operator-=(source &&other) {
            assign(*this, *this - std::forward<source>(other));
            return *this;
        }

        /// @brief Multiplies by an array or expression, element by element, in place.
        /// @param other The array or expression.
        /// @return This array.
        template <typename source>
            requires compatible<Array &, source>
        Array &operator*=(source &&other) {
            assign(*this, *this * std::forward<source>(other));
            return *this;
        }

        /// @brief Divides by an array or expression, element by element, in place.
        /// @param other The array or expression.
        /// @return This array.
        template <typename source>
            requires compatible<Array &, source>
        Array &operator/=(source &&other) {
            assign(*this, *this / std::forward<source>(other));
            return *this;
        }
    };

    /// Same shape and every element equal; `!=` is derived from it.
    /// @param left The left operand.
    /// @param right The right operand.
    /// @return Whether they are equal.
    template <typename element, core::size rank>
    bool operator==(const Array<element, rank> &left, const Array<element, rank> &right) {
        if (left.shape() != right.shape()) return false;
        return std::equal(left.begin(), left.end(), right.begin());
    }

    /// Prints the elements in row-major order: [1, 2, 3].
    /// @param stream The stream.
    /// @param array The array.
    /// @return The stream.
    template <typename element, core::size rank>
    std::ostream &operator<<(std::ostream &stream, const Array<element, rank> &array) {
        stream << "[";
        for (core::size position = 0; position < array.size(); ++position) {
            stream << array[position] << (position + 1 == array.size() ? "" : ", ");
        }
        return stream << "]";
    }

}
