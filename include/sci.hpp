#pragma once

// libsci in one header, laid out like numpy: everything you normally need is in `sci::`.
//
//     #include <sci.hpp>
//
//     auto a = sci::arange(0.0f, 1.0f, 0.001f);     // NDArray<float, 1>
//     auto b = sci::ones({1000});
//     sci::NDArray<float, 1> c = a * b + 2.0f;       // computed in a single loop
//     float total = sci::sum(sci::sin(c));            // or c.sum()
//
//     sci::Device gpu(sci::Target::gpu);             // like torch.device / cupy
//     auto d = c.to(gpu);                            // expressions on d run on the GPU
//     auto m = sci::matmul(x.to(gpu), y.to(gpu));
//
// Expressions are lazy: `a * b + 2.0f` is computed when it is stored in an NDArray, reduced, or
// copied, in a single loop with no temporary arrays. Reading a GPU array on the host waits for the GPU.

#include "array/ndarray.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/target.hpp"
#include "linalg/decompose.hpp"
#include "linalg/matmul.hpp"
#include "linalg/solve.hpp"
#include "ops/binary.hpp"
#include "ops/creation.hpp"
#include "ops/reduce.hpp"
#include "ops/unary.hpp"

namespace sci {

    using array::NDArray;
    using array::Shape;
    using core::drivers::Device;
    using core::drivers::Target;

    using ops::arange;
    using ops::empty;
    using ops::full;
    using ops::ones;
    using ops::zeros;

    using ops::add;
    using ops::copy;
    using ops::divide;
    using ops::maximum;
    using ops::minimum;
    using ops::multiply;
    using ops::subtract;

    using ops::abs;
    using ops::cos;
    using ops::exp;
    using ops::log;
    using ops::negative;
    using ops::sin;
    using ops::sqrt;

    using ops::max;
    using ops::mean;
    using ops::min;
    using ops::sum;

    using linalg::dot;
    using linalg::matmul;

}
