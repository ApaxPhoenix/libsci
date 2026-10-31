#pragma once

#include "core/bytecode/code.hpp"
#include "core/drivers/processor/kernels/generic/exponential.hpp"
#include "core/drivers/processor/kernels/generic/library.hpp"
#include "core/drivers/processor/kernels/generic/logarithm.hpp"
#include "core/drivers/processor/kernels/generic/trigonometry.hpp"

// The C functions (::asinf, ::sinh, ...), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::processor::kernels {

    // What one code does to a pack. The loops in elementwise.hpp and reductions.hpp call apply()
    // with the ISA's pack, and with its one-lane Scalar pack for the tail. The functions with no
    // polynomial of their own, the inverse circular and the hyperbolic ones, go through the C library.
    template <Code code>
    struct Operation {
        template <typename pack>
        static pack::value apply(const pack::value item) noexcept {
            if constexpr (code == Code::copy) return item;
            else if constexpr (code == Code::negate) return pack::negate(item);
            else if constexpr (code == Code::absolute) return pack::absolute(item);
            else if constexpr (code == Code::root) return pack::root(item);
            else if constexpr (code == Code::exponential) return Exponential<pack>::exponential(item);
            else if constexpr (code == Code::logarithm) return Logarithm<pack>::logarithm(item);
            else if constexpr (code == Code::sine) return Trigonometry<pack>::template evaluate<false>(item);
            else if constexpr (code == Code::cosine) return Trigonometry<pack>::template evaluate<true>(item);
            else if constexpr (code == Code::tangent) {
                return pack::divide(Trigonometry<pack>::template evaluate<false>(item), Trigonometry<pack>::template evaluate<true>(item));
            }
            else if constexpr (code == Code::arcsine) return Library<pack>::template apply<&asinf, &::asin>(item);
            else if constexpr (code == Code::arccosine) return Library<pack>::template apply<&acosf, &::acos>(item);
            else if constexpr (code == Code::arctangent) return Library<pack>::template apply<&atanf, &::atan>(item);
            else if constexpr (code == hyperbolic::sine) return Library<pack>::template apply<&sinhf, &::sinh>(item);
            else if constexpr (code == hyperbolic::cosine) return Library<pack>::template apply<&coshf, &::cosh>(item);
            else if constexpr (code == hyperbolic::tangent) return Library<pack>::template apply<&tanhf, &::tanh>(item);
            else if constexpr (code == hyperbolic::arcsine) return Library<pack>::template apply<&asinhf, &::asinh>(item);
            else if constexpr (code == hyperbolic::arccosine) return Library<pack>::template apply<&acoshf, &::acosh>(item);
            else if constexpr (code == hyperbolic::cosecant) {
                return pack::divide(pack::splat(typename pack::element(1)), Operation<hyperbolic::sine>::apply<pack>(item));
            }
            else if constexpr (code == hyperbolic::secant) {
                return pack::divide(pack::splat(typename pack::element(1)), Operation<hyperbolic::cosine>::apply<pack>(item));
            }
            else if constexpr (code == hyperbolic::cotangent) {
                return pack::divide(pack::splat(typename pack::element(1)), Operation<hyperbolic::tangent>::apply<pack>(item));
            }
            else if constexpr (code == hyperbolic::arccosecant) {
                return Operation<hyperbolic::arcsine>::apply<pack>(pack::divide(pack::splat(typename pack::element(1)), item));
            }
            else if constexpr (code == hyperbolic::arcsecant) {
                return Operation<hyperbolic::arccosine>::apply<pack>(pack::divide(pack::splat(typename pack::element(1)), item));
            }
            else {
                static_assert(code == hyperbolic::arctangent, "every one-operand code needs a case here");
                return Library<pack>::template apply<&::atanhf, &::atanh>(item);
            }
        }

        template <typename pack>
        static pack::value apply(const pack::value left, const pack::value right) noexcept {
            if constexpr (code == Code::add) return pack::add(left, right);
            else if constexpr (code == Code::subtract) return pack::subtract(left, right);
            else if constexpr (code == Code::multiply) return pack::multiply(left, right);
            else if constexpr (code == Code::divide) return pack::divide(left, right);
            else if constexpr (code == Code::minimum) return pack::minimum(left, right);
            else {
                static_assert(code == Code::maximum, "every two-operand code needs a case here");
                return pack::maximum(left, right);
            }
        }
    };

}