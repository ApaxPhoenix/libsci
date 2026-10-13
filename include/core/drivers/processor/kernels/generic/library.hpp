#pragma once

#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    // Runs a C library function on every lane of a pack, the float or the double version to match
    // the pack: the functions with no polynomial of their own (see Trigonometry and Hyperbolic).
    template <typename pack>
    struct Library {
        using type = pack::element;
        using value = pack::value;

        template <float (*narrow)(float), double (*wide)(double)>
        static value apply(const value input) noexcept {
            type inputs[pack::width];
            type outputs[pack::width];
            pack::store(inputs, input);
            for (size lane = 0; lane < pack::width; ++lane) {
                if constexpr (sizeof(type) == 4) {
                    outputs[lane] = narrow(inputs[lane]);
                } else {
                    outputs[lane] = wide(inputs[lane]);
                }
            }
            return pack::load(outputs);
        }
    };

}
