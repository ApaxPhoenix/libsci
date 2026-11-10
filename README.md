<h1 align="center">
  libsci
</h1>

<p align="center">
  <b>A numpy-style array library for modern C++, on SIMD CPUs and Vulkan GPUs.</b>
</p>

<p align="center">
  <a href="https://en.cppreference.com/w/cpp/compiler_support">
    <img alt="C++ Standard" src="https://img.shields.io/badge/C%2B%2B-23-blue?logo=c%2B%2B"
  ></a>
  <a href="https://github.com/ApaxPhoenix/libsci">
    <img alt="Platforms" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey"
  ></a>
  <a href="https://github.com/ApaxPhoenix/libsci/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/ApaxPhoenix/libsci?color=239dad&label=stars"
  ></a>
  <a href="https://github.com/ApaxPhoenix/libsci/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/ApaxPhoenix/libsci?color=5865F2&label=issues"
  ></a>
  <a href="https://github.com/ApaxPhoenix/libsci/pulls">
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-A561FF"
  ></a>
  <a href="https://github.com/ApaxPhoenix/libsci/commits/dev">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/ApaxPhoenix/libsci/dev?color=success"
  ></a>
  <a href="https://github.com/ApaxPhoenix/libsci/blob/main/LICENSE">
    <img alt="AGPL v3 License" src="https://img.shields.io/badge/license-AGPL%20v3-brightgreen"
  ></a>
</p>

libsci brings numpy's way of working to C++: arrays, elementwise math and reductions with the
names you already know, in one header. Behind it is an engine that treats every expression as a
small program. `a * b + c` is not computed when you write it. It is recorded as bytecode and runs
in a single pass when the result is needed, with no temporary arrays in between.

The same bytecode runs everywhere: on SIMD kernels chosen for the CPU at runtime, as machine code
from a JIT, and on the GPU through Vulkan. The code you write doesn't change; where the arrays
live decides where it runs.

```cpp
#include <sci.hpp>

int main() {
    const sci::Array<float, 1> first = sci::linspace(0.0f, 1.0f, 1'000'000);
    const sci::Array<float, 1> second = sci::ones({1'000'000});

    sci::Array<float, 1> third = first * second + 2.0f;   // one loop, no temporaries
    const float total = sci::sum(sci::sin(third));        // the sine never gets stored

    sci::Device graphics(sci::Target::graphics);          // falls back to the CPU without a GPU
    const auto moved = third.to(graphics);
    sci::Array<float, 1> fourth = sci::exponential(moved) * 0.5f;   // runs on the GPU
    return fourth.sum() > total ? 0 : 1;
}
```

## Highlights

- **Lazy and fused.** Expressions are computed in one pass over the data, block by block so
  their working set stays in cache. A multiply followed by an add becomes one fused multiply-add,
  and `sum(a * b)` never builds `a * b`.
- **Fast on every CPU.** SIMD kernels for NEON, AVX2 and AVX-512, plus a scalar fallback, picked
  at runtime from what the CPU supports. Large arrays are split across a thread pool whose hot path takes no locks,
  and memory-bound work on Apple chips stays on the performance cores.
- **Two JITs.** On the CPU, an expression of two or more instructions compiles into one
  machine-code loop with every value in a register (asmjit, NEON and AVX2). On the GPU, every
  expression gets a shader of its own, compiled by Slang on a background thread while the
  interpreter runs it.
- **GPU arrays.** `array.to(device)` moves an array to the GPU. Expressions, reductions and
  matrix products on it run there asynchronously, and reading it on the host waits for the
  result.
- **Accurate.** The vector exponential and logarithm are within 1 ulp, and sine and cosine
  within 2. `minimum` and `maximum` let NaN win, as numpy does.
- **numpy's vocabulary.** `zeros`, `ones`, `full`, `empty`, `range`, `linspace` and `eye`;
  `add`, `subtract`, `multiply` and `divide`; `absolute`, `root`, `exponential`, `logarithm`;
  every circular and hyperbolic function and its inverse (`sin` … `acot`, `sinh` … `acoth`);
  `sum`, `mean`, `minimum` and `maximum`; `product`, `dot`, LU decomposition and `solve`.

## Status

libsci is pre-release software (version 0.1.0). The array engine, the CPU kernels and both JITs
are complete. The GPU runs float32; float64 falls back to the CPU. Broadcasting, reshaping and
reductions along one dimension are next. See [CHANGELOG.md](CHANGELOG.md) for what has been done.

## Building from source

### Requirements

- CMake 3.20 or later.
- A C++23 compiler: Apple clang 16 or later, GCC 13 or later, or a recent MSVC.
- The Vulkan loader and headers. On macOS, the Vulkan SDK with MoltenVK.
- Slang and asmjit are downloaded into `vendors/` the first time CMake configures.

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build
```

This builds the `sci` static library and the demo in `src/main.cpp` as `bin/libsci`. To use
libsci in your own project, link `sci` and include `<sci.hpp>`. Add `-DSCI_NATIVE=ON` to tune
every file for the machine you build on.

## Settings

| Variable      | What it does                                                    |
|---------------|-----------------------------------------------------------------|
| `SCI_ISA`     | Forces the CPU kernels: `fallback`, `neon`, `avx2` or `avx512`. |
| `SCI_JIT`     | `0` turns both JITs off, `trace` reports every compile.         |
| `SCI_THREADS` | Size of the thread pool. The default is one thread per core.    |

## Layout

| Folder                                         | What is in it                                        |
|------------------------------------------------|------------------------------------------------------|
| `include/array/`                               | `Array`, its storage and the lazy expression nodes   |
| `include/operations/`                          | The numpy-style functions                            |
| `include/linear/`                              | Matrix product, LU decomposition and solving         |
| `include/core/bytecode/`                       | The bytecode every engine runs                       |
| `include/core/drivers/processor/`              | SIMD kernels, thread pool and CPU JIT                |
| `include/core/drivers/vulkan/`, `shaders/`     | The GPU path and its Slang shaders                   |

## License

libsci is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
