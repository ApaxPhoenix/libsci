#pragma once

#include "core/bytecode/code.hpp"
#include "core/drivers/cpu/kernels/generic/exponential.hpp"
#include "core/drivers/cpu/kernels/generic/logarithm.hpp"
#include "core/drivers/cpu/kernels/generic/trigonometry.hpp"

namespace sci::core::drivers::cpu::kernels {

    // What one opcode does to a pack. The loops in elementwise.hpp and reductions.hpp call apply()
    // with the ISA's pack, and with its one-lane Scalar pack for the tail.
    template <Code code>
    struct Operation;

    template <>
    struct Operation<Code::copy> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return item; }
    };

    template <>
    struct Operation<Code::neg> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return pack::neg(item); }
    };

    template <>
    struct Operation<Code::abs> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return pack::abs(item); }
    };

    template <>
    struct Operation<Code::sqrt> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return pack::sqrt(item); }
    };

    template <>
    struct Operation<Code::exp> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Exponential<pack>::exp(item); }
    };

    template <>
    struct Operation<Code::log> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Logarithm<pack>::log(item); }
    };

    template <>
    struct Operation<Code::sin> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::sin(item); }
    };

    template <>
    struct Operation<Code::cos> {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept { return Trigonometry<pack>::cos(item); }
    };

    template <>
    struct Operation<Code::add> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::add(left, right); }
    };

    template <>
    struct Operation<Code::sub> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::sub(left, right); }
    };

    template <>
    struct Operation<Code::mul> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::mul(left, right); }
    };

    template <>
    struct Operation<Code::div> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::div(left, right); }
    };

    template <>
    struct Operation<Code::min> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::min(left, right); }
    };

    template <>
    struct Operation<Code::max> {
        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept { return pack::max(left, right); }
    };

}
