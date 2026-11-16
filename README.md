# libsci

**numpy for C++. One header, one pass over the data, on any CPU or GPU.**

libsci is an array library for people who think in numpy and need to ship C++. You write
`a * b + c` with the names you already know (`zeros`, `linspace`, `sum`, `exponential`, `sin`),
and libsci turns it into a single loop with no temporary arrays. The same code runs on SIMD
CPUs, through a JIT, or on the GPU, depending only on where your arrays live.

libsci is built for anyone who wants numerical code that is fast by default, without learning a
new way to write it.

> **Status: pre-release (0.1.0-alpha.1).** libsci is changing quickly. The engine, the CPU
> kernels and both JITs work today, and the API may still change before 1.0.

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

## Contents

[Highlights](#highlights) ·
[How it works](#how-it-works) ·
[Features](#features) ·
[How it compares](#how-it-compares) ·
[Roadmap](#roadmap) ·
[Platforms](#platforms) ·
[Building from source](#building-from-source) ·
[Usage](#usage) ·
[Documentation](#documentation) ·
[Project layout](#project-layout) ·
[Security](#security) ·
[Contributing](#contributing) ·
[License](#license)

## Highlights

- **numpy you can compile.** Arrays, elementwise math, reductions and linear algebra, named the
  way numpy names them, in one header with no Python anywhere.
- **Fast without trying.** Expressions fuse into one pass, pick the best SIMD instructions for
  the CPU they run on, spread across cores, and compile to machine code when they repeat.
- **One code path, every device.** Move an array to the GPU with `to(device)` and the same
  expressions run there. No kernels to write, no second API to learn.
- **Correct to the bit.** Every path, from the scalar fallback to the GPU JIT, follows the same
  rules: NaN propagates as in numpy, and the vector math stays within a couple of ulp of the C
  library.

## How it works

1. **Expressions are recorded, not computed.** `a * b + c` builds a small tree of nodes. Nothing
   runs until the result is stored, reduced or copied.
2. **The tree becomes bytecode.** When it is needed, the tree is recorded into a `Chunk`: a short
   list of instructions over input arrays, constants and temporaries. A multiply followed by an
   add becomes one fused multiply-add, and `sum(a * b)` becomes a dot product that never stores
   `a * b`.
3. **The CPU runs it block by block.** The chunk runs over 1,024 elements at a time, so every
   temporary stays in the L1 cache. The SIMD kernels (NEON, AVX2, AVX-512 or a scalar fallback)
   are chosen once from what the CPU supports, and large arrays are split across a thread pool.
4. **Repeated chunks become machine code.** The second time a chunk runs, or the first time it
   is large, the CPU JIT (asmjit) compiles it into one loop that keeps every value in a register.
5. **GPU arrays run on the GPU.** For arrays on a Vulkan device, the chunk goes to an
   interpreter shader straight away, while a background thread compiles a shader of its own with
   Slang. Work is batched into command buffers and runs asynchronously.
6. **Reading waits for the result.** Touching a GPU array from the host waits for the queued
   work first, so a GPU array reads like any other.

## Features

**Arrays**
- `sci::Array<type, rank>`: contiguous, row-major, on the host, on a `Device` or in an `Arena`.
- Composed with `zeros`, `ones`, `full`, `empty`, `range`, `linspace` and `eye`, on the host or
  straight on a device: `sci::ones({1024}, graphics)`.
- Indexing with `a[i, j]`, iteration, `fill`, `copy`, `to(device)` and `host()`.
- Reshaping: `reshape`, `transpose`, `flatten`, `squeeze` and `expand`.

**Math**
- `+ - * /` and `add`, `subtract`, `multiply`, `divide`, with arrays or scalars on either side.
- `absolute`, `root`, `exponential`, `logarithm` and `negate`.
- Every circular function and its inverse: `sin`, `cos`, `tan`, `csc`, `sec`, `cot`, `asin` …
  `acot`.
- Every hyperbolic function and its inverse: `sinh` … `coth`, `asinh` … `acoth`.
- Elementwise `minimum` and `maximum`, where NaN wins as in numpy.

**Reductions and linear algebra**
- `sum`, `mean`, `minimum` and `maximum`, fused with the expression they reduce.
- `product` (matrix product, Apple's Accelerate on macOS), `dot`, `linear::decompose` (LU with
  partial pivoting) and `linear::solve`.

**Performance**
- SIMD kernels for NEON, AVX2 and AVX-512, picked at runtime.
- A thread pool whose hot path takes no locks. Memory-bound work on Apple chips stays on the
  performance cores.
- A CPU JIT for NEON and AVX2, and a GPU JIT that compiles in the background.
- The vector exponential and logarithm are within 1 ulp, and sine and cosine within 2.

**GPU**
- Vulkan 1.2 on any capable GPU, including Apple GPUs through MoltenVK.
- Expressions, reductions and matrix products run on the device, split so no single dispatch
  trips the macOS GPU watchdog.
- Without a usable GPU, a `Device` falls back to the CPU and everything still works.

## How it compares

|  | libsci | NumPy | Eigen | xtensor |
| --- | :---: | :---: | :---: | :---: |
| C++ library | ✅ | ❌ | ✅ | ✅ |
| numpy's names | ✅ | ✅ | ❌ | ✅ |
| Lazy, fused expressions | ✅ | ❌ | ✅ | ✅ |
| SIMD chosen at runtime | ✅ | ✅ | ❌ | ❌ |
| JIT-compiled expressions | ✅ | ❌ | ❌ | ❌ |
| Multithreaded elementwise math | ✅ | ❌ | ❌ | Optional |
| GPU arrays | ✅ Vulkan | ❌ | ❌ | ❌ |
| Reshape and transpose | ✅ Copies | ✅ | ✅ | ✅ |
| Broadcasting | Planned | ✅ | ❌ | ✅ |
| Views and slicing without copies | Planned | ✅ | ✅ | ✅ |
| Reductions along an axis | Planned | ✅ | ✅ | ✅ |

This table is the project's best reading of each library's public documentation and may fall out
of date. If something in it is wrong, please open an issue.

## Roadmap

### Done in 0.1.0-alpha.1
- [x] Lazy expressions fused into one pass, as bytecode
- [x] SIMD kernels for NEON, AVX2 and AVX-512, chosen at run time
- [x] A CPU JIT with asmjit, and a GPU JIT with Slang over Vulkan
- [x] `reshape`, `transpose`, `flatten`, `squeeze` and `expand`

### Next
- [ ] Broadcasting, so `(3, 1)` and `(1, 4)` arrays combine into `(3, 4)`
- [ ] Views: `reshape`, `transpose` and slicing without copies
- [ ] Reductions along one dimension: `sum(a, axis)` and the rest
- [ ] More elementwise functions: `power`, `floor`, `ceil`, `round`, `clip` and `where`
- [ ] A test suite, and CI on Linux x86-64, Windows and macOS

### Later
- [ ] Reading and writing `.npy` files
- [ ] Random numbers
- [ ] Integer arrays on the fast path
- [ ] Vector versions of the functions that still call the C library
- [ ] float64 on GPUs that support it

Every release's notes are in [releases/](releases/), and [CHANGELOG.md](CHANGELOG.md) collects
them all.

## Platforms

| Platform | Status |
| --- | --- |
| macOS, Apple silicon | Works. This is where development happens; the GPU runs through MoltenVK. |
| Linux, x86-64 | Builds. The AVX2 and AVX-512 kernels and the x86 JIT are checked by the compiler but not yet tested on hardware. |
| Windows, x86-64 | Configures with `clang-cl`; not yet tested on hardware. |

## Building from source

### Requirements

- CMake 3.25 or later and Ninja.
- A C++26 compiler: Clang 17 or later, GCC 14 or later, or Apple clang 16. On Windows, use
  `clang-cl`, which the build selects if your IDE profile names MSVC (`cl`) or MinGW.
- The Vulkan loader and headers; on macOS, the Vulkan SDK with MoltenVK.
- Nothing else: the first configure downloads Slang and asmjit into `vendors/`.

### Build and test

```bash
git clone https://github.com/ApaxPhoenix/libsci.git
```

```bash
cmake --preset release
```

```bash
cmake --build --preset release
```

```bash
ctest --preset release
```

This builds the `sci` static library, and the demonstration in `src/main.cpp` as `bin/libsci`.

### Presets

| Preset | What it builds |
| --- | --- |
| `release` | The optimised build. |
| `debug` | The same with assertions on and no optimisation. |
| `sanitize` | Everything under AddressSanitizer and UndefinedBehaviorSanitizer. |
| `analyze` | clang-tidy and the Clang Static Analyzer over libsci's own sources. |

What belongs to your machine alone -- a compiler path, a toolchain -- goes in
`CMakeUserPresets.json`, which is never committed.

## Usage

Link `sci` and include `<sci.hpp>`:

```cmake
add_subdirectory(libsci)
target_link_libraries(program PRIVATE sci::sci)
```

These settings are read when the program runs, or, for `SCI_NATIVE`, when it is configured:

| Setting | What it does |
| --- | --- |
| `SCI_ISA` | Forces the CPU kernels: `fallback`, `neon`, `avx2` or `avx512`. |
| `SCI_JIT` | `0` turns both JITs off, `trace` reports every compile. |
| `SCI_THREADS` | Size of the thread pool. The default is one thread per core. |
| `SCI_NATIVE` | CMake option that tunes every file for the build machine. |

## Documentation

The API reference is generated from the sources with Doxygen:

```bash
cmake --build --preset release --target docs
```

It is written to `cmake-build-release/docs/html`.

## Project layout

| Path | Contents |
| --- | --- |
| `include/`, `src/` | libsci: `array` (arrays, shapes, strides, storage and deferred expressions), `operations` (the numpy-style functions), `linear` (products, factorizations and solving), `core` (bytecode, memory, and the CPU and GPU drivers) |
| `shaders/` | The Slang compute shaders, compiled into the library |
| `cmake/` | The scripts that turn shaders into headers |
| `releases/` | Every release's notes |
| `bin/`, `build/` | What a build leaves, and scratch files; neither is committed |

## Security

Found a vulnerability? Please don't report it in a public issue. Report it privately, with
**Report a vulnerability** on the repository's **Security** tab or by email to
andromedeyz@hotmail.com, and read [SECURITY.md](SECURITY.md) before disclosing anything publicly.

## Contributing

Contributions of every size are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which
covers the [Contributor License Agreement](CLA.md), and please read the
[Code of Conduct](CODE_OF_CONDUCT.md). Open an issue to report a bug or to discuss a feature
before starting on a large change. Pull requests should:

- build without warnings;
- check every operation they change against a plain scalar loop;
- add an entry to [CHANGELOG.md](CHANGELOG.md).

libsci is maintained by Andres Hernandez ([\@ApaxPhoenix](https://github.com/ApaxPhoenix)).

## License

Copyright (C) 2026 Andres Hernandez.

libsci is dual-licensed. It is free software under the
[GNU Affero General Public License v3.0](LICENSE), and is also available under a commercial
license for use the AGPL does not allow, such as in closed-source products: see
[LICENSING.md](LICENSING.md) and [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). Contributions
are accepted under the [Contributor License Agreement](CLA.md) described in
[CONTRIBUTING.md](CONTRIBUTING.md).

The libraries libsci builds on keep their own licenses: see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Acknowledgements

libsci compiles through [asmjit](https://asmjit.com), by Petr Kobalicek, and its shaders through
[Slang](https://shader-slang.org). It runs on the GPU through [Vulkan](https://www.vulkan.org)
and, on Apple hardware, [MoltenVK](https://github.com/KhronosGroup/MoltenVK). Its names and its
rules for NaN follow [NumPy](https://numpy.org).
