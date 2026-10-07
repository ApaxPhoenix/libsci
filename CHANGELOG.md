# Changelog

All notable changes to libsci are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Nothing has been tagged yet; the
version in `CMakeLists.txt` is 0.1.0.

## [Unreleased]

### Added

- `sci.hpp`, a single header with a numpy-style API in the `sci` namespace: `NDArray`, `Shape`,
  `Device` and `Target`; `add`, `subtract`, `multiply`, `divide`, `minimum`, `maximum`, `negative`,
  `abs`, `sqrt`, `exp`, `log`, `sin`, `cos` and `copy`; `sum`, `mean`, `min` and `max`; `zeros`,
  `ones`, `full`, `empty` and `arange`; `dot` and `matmul`.
- Lazy expressions. `a * b + c` is recorded as bytecode (`core::Chunk`) and computed in a single
  pass, with no temporary arrays, when it is stored, reduced or copied. A multiply followed by an
  add becomes one fused multiply-add, and `sum(a * b)` never builds `a * b`.
- SIMD CPU kernels for every operation, chosen at runtime: scalar fallback, NEON, AVX2 with FMA,
  and AVX-512. The `SCI_ISA` environment variable forces one. The vector `exp`, `log`, `sin` and
  `cos` are within a few ulp of the C library.
- Large arrays are split across a fork-join thread pool (`SCI_THREADS` sets its size).
  Memory-bound work on Apple chips uses only the performance cores.
- GPU arrays: `sci::Device gpu(sci::Target::gpu)` and `array.to(gpu)`. Expressions, reductions
  and `matmul` on GPU arrays run there asynchronously, and reading a GPU array on the host waits
  for the result. Without a usable Vulkan device, the device falls back to the CPU.
- Slang compute shaders (`shaders/interpreter.slang`, `shaders/matmul.slang`), compiled with
  `slangc` at build time and embedded as SPIR-V by `cmake/embed.cmake`.
- GPU work is batched into command buffers (`vulkan::Stream`), and GPU buffers are recycled
  through a size-keyed cache (`vulkan::Allocator`).
- `matmul` uses Apple's Accelerate on macOS.
- `SCI_NATIVE` CMake option to tune every file for the build machine (`-march=native` /
  `-mcpu=native`).
- CPU JIT (`cpu::Jit`, with [asmjit](https://asmjit.com)): an expression of two or more
  instructions compiles into one machine-code loop that keeps every value in a register, for NEON
  on ARM64 and AVX2 on x86-64. Results are the kernel tables' to the bit, apart from which NaN
  comes out when two NaNs meet. A chunk compiles the second time it runs, or the first time when
  it is large. Expressions with `exp`, `log`, `sin` or `cos` keep running on the kernels for now.
  About 2× faster for arithmetic chains that fit in cache.
- GPU JIT (`vulkan::Jit`, with Slang's runtime library): every expression run on the GPU gets a
  shader of its own, compiled on a background thread; it runs on the interpreter until the shader
  is ready. Up to 2.2× faster for long expressions on 16M-element arrays.
- `SCI_JIT` environment variable: `0` turns both JITs off, `trace` reports every compile.
- `Chunk::signature()`, the key both JITs cache compiled code under.
- `CHANGELOG.md` and `SECURITY.md`.

### Changed

- Class names are capitalised: `NDArray`, `Shape`, `Device`, `Target` (previously `ndarray`,
  `shape`, `device`, `target`).
- Operations use numpy's names, such as `multiply` instead of `mul`.
- `linalg::lu` pivots and returns `Factors` (`permutation`, `lower`, `upper`), as
  `scipy.linalg.lu` does, instead of a pair of lower and upper factors. `linalg::solve` also
  accepts several right-hand sides as a matrix.
- `vulkan::dispatch` is replaced by `vulkan::Stream`.
- Every class has its own header, named after the class, and `include/` mirrors `src/` (no
  headers under `src/`). The shared CPU kernel code lives in
  `include/core/drivers/cpu/kernels/generic/`, one file per class (`Scalar`, `Polynomial`,
  `Exponential`, `Logarithm`, `Trigonometry`, `Operation`, ...), and each instruction set has its
  own header (`fallback.hpp`, `neon.hpp`, `avx2.hpp`, `avx512.hpp`).
- Helper types that were nested inside other classes are now classes of their own:
  `memory::Block` (formerly `Arena::Chunk`), `vulkan::Batch` and `vulkan::Region`.
- Variables and parameters are named after their type (`Batch& batch`, `Buffer& buffer`).
- No anonymous namespaces. File-local helpers are `static`, and each instruction set's SIMD
  wrapper has its own name (`Fallback`, `Neon`, `Avx2`, `Avx512`), so the kernels of one
  instruction set can never be linked in place of another's.
- The interpreter shader's shared parts moved to `shaders/common/` (`layout.slang`,
  `entries.slang`), which the GPU JIT's shaders reuse; `cmake/sources.cmake` embeds them as text.
- asmjit is fetched at a pinned commit and checked against its SHA-256; Slang's runtime library is
  now linked.

### Removed

- Unused headers: `core/meta/traits.hpp`, `core/meta/concepts.hpp` and `core/memory/pool.hpp`.
- The test suite (`tests/main.cpp` and the `sci_tests` target), for now.

### Fixed

- Clang-Tidy and CLion warnings: `const` on Vulkan handle typedefs, constant arguments to the
  stream's memory-barrier helper, `Stream::complete` changing a batch it received by reference,
  and a flag in `Node::shape()` that looked unused.
- A GPU device destroyed its pipelines before waiting for queued GPU work that might still use
  them; the command stream now finishes first.

### Performance

- The first array operation no longer asks the operating system for the number of performance
  cores. The count is queried once, the first time a large multithreaded job needs it, which cut
  the demo's first run from about 15 µs to about 6 µs.

## 2026-08-20

### Added

- GPU support through Vulkan: context, buffer allocator, pipelines and a dispatcher, with the
  compute shaders embedded as SPIR-V bytes.
- CPU driver with runtime instruction-set detection, a thread pool, and SIMD `add` kernels for
  AVX2, AVX-512, NEON and a scalar fallback.
- Memory arena, cursor, pool and slice utilities.
- `matmul` (replacing `linalg/matrix.hpp`).
- Slang is fetched when CMake configures the project, for compiling shaders later.

## 2026-08-08

### Added

- Initial commit: `ndarray` with shapes, strides, slices and storage; elementwise, broadcasting
  and reduction operations; LU decomposition and `solve`.
