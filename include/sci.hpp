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
#include "operations/binary.hpp"
#include "operations/creation.hpp"
#include "operations/reduce.hpp"
#include "operations/unary.hpp"

namespace sci {

    using array::NDArray;
    using array::Shape;
    using core::drivers::Device;
    using core::drivers::Target;

    using operations::arange;
    using operations::empty;
    using operations::full;
    using operations::ones;
    using operations::zeros;

    using operations::add;
    using operations::divide;
    using operations::multiply;
    using operations::subtract;

    using operations::sin;
    using operations::cos;
    using operations::tan;
    using operations::asin;
    using operations::acos;
    using operations::atan;
    using operations::sinh;
    using operations::cosh;
    using operations::tanh;
    using operations::asinh;
    using operations::acosh;
    using operations::atanh;





















    using operations::copy;
    using operations::maximum;
    using operations::minimum;



    using operations::abs;
    using operations::exp;
    using operations::log;
    using operations::negative;
    using operations::sqrt;

    using operations::max;
    using operations::mean;
    using operations::min;
    using operations::sum;

    using linalg::dot;
    using linalg::matmul;

}
