#pragma once

// libsci in one header, laid out like numpy: everything you normally need is in `sci::`.
//
//     #include <sci.hpp>
//
//     auto first = sci::range(0.0f, 1.0f, 0.001f);           // Array<float, 1>
//     auto second = sci::ones({1000});
//     sci::Array<float, 1> third = first * second + 2.0f;    // computed in a single loop
//     float total = sci::sum(sci::sine(third));              // or third.sum()
//
//     sci::Device graphics(sci::Target::graphics);           // like torch.device / cupy
//     auto fourth = third.to(graphics);                      // expressions on it run on the GPU
//     auto matrix = sci::product(left.to(graphics), right.to(graphics));
//
// Expressions are lazy: `first * second + 2.0f` is computed when it is stored in an Array, reduced,
// or copied, in a single loop with no temporary arrays. Reading a GPU array on the host waits for
// the GPU.

#include "array/array.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/target.hpp"
#include "linear/decompose.hpp"
#include "linear/product.hpp"
#include "linear/solve.hpp"
#include "operations/binary.hpp"
#include "operations/creation.hpp"
#include "operations/reduce.hpp"
#include "operations/unary.hpp"

namespace sci {

    using array::Array;
    using array::Shape;
    using core::drivers::Device;
    using core::drivers::Target;

    using operations::empty;
    using operations::full;
    using operations::ones;
    using operations::range;
    using operations::zeros;

    using operations::add;
    using operations::divide;
    using operations::multiply;
    using operations::subtract;

    using operations::copy;

    // With two operands elementwise (numpy.minimum), with one a reduction (numpy.min).
    using operations::maximum;
    using operations::minimum;

    using operations::absolute;
    using operations::exponential;
    using operations::logarithm;
    using operations::negative;
    using operations::root;

    using operations::sine;
    using operations::cosine;
    using operations::tangent;
    using operations::arcsine;
    using operations::arccosine;
    using operations::arctangent;

    // sci::hyperbolic::sine (sinh), ::cosine, ::tangent, ::arcsine, ::arccosine, ::arctangent.
    namespace hyperbolic = operations::hyperbolic;

    using operations::mean;
    using operations::sum;

    using linear::dot;
    using linear::product;

}
