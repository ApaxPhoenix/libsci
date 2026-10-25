#include "core/drivers/processor/kernels/fallback.hpp"
#include "core/drivers/processor/kernels/generic/compose.hpp"
#include "core/drivers/processor/kernels/generic/scalar.hpp"
#include "core/types.hpp"

namespace sci::core::drivers::processor::kernels {

    // Plain scalar code: any CPU, and the reference the vector paths are tested against.
    template <typename type>
    struct Fallback;

    template <>
    struct Fallback<float> : Scalar<float, uint32, Fallback<float>> {};

    template <>
    struct Fallback<double> : Scalar<double, uint64, Fallback<double>> {};

    constexpr Table table = compose<Fallback>("fallback");

    const Table* const fallback = &table;

}
