#pragma once

#include "core/drivers/processor/kernels/generic/library.hpp"

// The C functions (::sinhf, ::cosh, ...), not <cmath>: its std:: overloads are inline functions the
// linker could share between ISA files (see compose.hpp).
#include <math.h> // NOLINT(modernize-deprecated-headers)

namespace sci::core::drivers::processor::kernels {

    // The hyperbolic functions and their inverses over a whole pack, through the C library.
    template <typename pack>
    struct Hyperbolic {
        using value = pack::value;

        static value sine(const value input) noexcept {
            return Library<pack>::template apply<&::sinhf, &::sinh>(input);
        }

        static value cosine(const value input) noexcept {
            return Library<pack>::template apply<&::coshf, &::cosh>(input);
        }

        static value tangent(const value input) noexcept {
            return Library<pack>::template apply<&::tanhf, &::tanh>(input);
        }

        static value arcsine(const value input) noexcept {
            return Library<pack>::template apply<&::asinhf, &::asinh>(input);
        }

        static value arccosine(const value input) noexcept {
            return Library<pack>::template apply<&::acoshf, &::acosh>(input);
        }

        static value arctangent(const value input) noexcept {
            return Library<pack>::template apply<&::atanhf, &::atanh>(input);
        }
    };

}
