#pragma once

#include "core/bytecode/code.hpp"
#include "core/drivers/processor/kernels/generic/exponential.hpp"
#include "core/drivers/processor/kernels/generic/hyperbolic.hpp"
#include "core/drivers/processor/kernels/generic/logarithm.hpp"
#include "core/drivers/processor/kernels/generic/trigonometry.hpp"

namespace sci::core::drivers::processor::kernels {

    // What one code does to a pack. The loops in elementwise.hpp and reductions.hpp call apply()
    // with the ISA's pack, and with its one-lane Scalar pack for the tail.
    template <Code code>
    struct Operation;

    template <>
    struct Operation<Code::copy> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return item; }
    };

    template <>
    struct Operation<Code::negate> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return pack::negate(item); }
    };

    template <>
    struct Operation<Code::absolute> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return pack::absolute(item); }
    };

    template <>
    struct Operation<Code::root> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return pack::root(item); }
    };

    template <>
    struct Operation<Code::exponential> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Exponential<pack>::exponential(item); }
    };

    template <>
    struct Operation<Code::logarithm> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Logarithm<pack>::logarithm(item); }
    };

    template <>
    struct Operation<Code::sine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::sine(item); }
    };

    template <>
    struct Operation<Code::cosine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::cosine(item); }
    };

    template <>
    struct Operation<Code::tangent> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::tangent(item); }
    };

    template <>
    struct Operation<Code::arcsine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::arcsine(item); }
    };

    template <>
    struct Operation<Code::arccosine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::arccosine(item); }
    };

    template <>
    struct Operation<Code::arctangent> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::arctangent(item); }
    };

    template <>
    struct Operation<hyperbolic::sine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Hyperbolic<pack>::sine(item); }
    };

    template <>
    struct Operation<hyperbolic::cosine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Hyperbolic<pack>::cosine(item); }
    };

    template <>
    struct Operation<hyperbolic::tangent> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Hyperbolic<pack>::tangent(item); }
    };

    template <>
    struct Operation<hyperbolic::arcsine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Hyperbolic<pack>::arcsine(item); }
    };

    template <>
    struct Operation<hyperbolic::arccosine> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Hyperbolic<pack>::arccosine(item); }
    };

    template <>
    struct Operation<hyperbolic::arctangent> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Hyperbolic<pack>::arctangent(item); }
    };

    template <>
    struct Operation<Code::add> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::add(left, right); }
    };

    template <>
    struct Operation<Code::subtract> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::subtract(left, right); }
    };

    template <>
    struct Operation<Code::multiply> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::multiply(left, right); }
    };

    template <>
    struct Operation<Code::divide> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::divide(left, right); }
    };

    template <>
    struct Operation<Code::minimum> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::minimum(left, right); }
    };

    template <>
    struct Operation<Code::maximum> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::maximum(left, right); }
    };

}
