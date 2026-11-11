# libsci

**numpy for C++. One header, one pass over the data, on any CPU or GPU.**

libsci is an array library for people who think in numpy and need to ship C++. You write
`a * b + c` with the names you already know (`zeros`, `linspace`, `sum`, `exponential`, `sin`),
and libsci turns it into a single loop with no temporary arrays. The same code runs on SIMD
CPUs, through a JIT, or on the GPU, depending only on where your arrays live.

libsci is built for anyone who wants numerical code that is fast by default, without learning a
new way to write it.

> **Status: early development.** libsci is at version 0.1.0 and changing quickly. The engine,
> the CPU kernels and both JITs work today, and the API may still change before 1.0.

## What it's going to be

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
- Created with `zeros`, `ones`, `full`, `empty`, `range`, `linspace` and `eye`, on the host or
  straight on a device: `sci::ones({1024}, graphics)`.
- Indexing with `a[i, j]`, iteration, `fill`, `copy`, `to(device)` and `host()`.

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
| Broadcasting | Planned | ✅ | ❌ | ✅ |
| Views, reshape and slicing | Planned | ✅ | ✅ | ✅ |
| Reductions along an axis | Planned | ✅ | ✅ | ✅ |

This table is our best reading of each library's public documentation and may fall out of date.
If something is wrong, please open an issue.

## Roadmap

**Next**
- Broadcasting, so `(3, 1)` and `(1, 4)` arrays combine into `(3, 4)`
- Views: `reshape`, `transpose` and slicing without copies
- Reductions along one dimension: `sum(a, axis)` and friends
- More elementwise functions: `power`, `floor`, `ceil`, `round`, `clip` and `where`
- A test suite and CI on Linux x86-64, Windows and macOS

**Later**
- Reading and writing `.npy` files
- Random numbers
- Integer arrays on the fast path
- Vector versions of the functions that still call the C library
- float64 on GPUs that support it

## Platforms

| Platform | Status |
| --- | --- |
| macOS, Apple silicon | Works. This is where development happens; the GPU runs through MoltenVK. |
| Linux, Windows, x86-64 | Builds. The AVX2 and AVX-512 kernels and the x86 JIT are checked by the compiler but not yet tested on hardware. |

## Building

You need CMake 3.25 or later, a C++26 compiler (Apple clang 16, Clang 17 or GCC 14 or later),
and the Vulkan loader and headers (on macOS, the Vulkan SDK with MoltenVK). Slang and asmjit are
downloaded into `vendors/` the first time CMake configures.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build
```

This builds the `sci` static library and the demo in `src/main.cpp` as `bin/libsci`. To use
libsci in your own project, link `sci` and include `<sci.hpp>`:

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

### Settings

| Variable | What it does |
| --- | --- |
| `SCI_ISA` | Forces the CPU kernels: `fallback`, `neon`, `avx2` or `avx512`. |
| `SCI_JIT` | `0` turns both JITs off, `trace` reports every compile. |
| `SCI_THREADS` | Size of the thread pool. The default is one thread per core. |
| `SCI_NATIVE` | CMake option that tunes every file for the build machine. |

## License

libsci is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
