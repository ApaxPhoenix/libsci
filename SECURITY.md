# Security Policy

libsci computes on arrays that programs build, often from data they did not
write, and generates machine code and shaders while it runs. This page
explains which versions receive security fixes, how to report a
vulnerability, and what the library does and does not protect against.

## Supported versions

libsci is pre-release software. Security fixes go to the `main` branch and
ship in the next release.

| Version           | Supported          |
|-------------------|--------------------|
| 0.1.x (`main`)    | :white_check_mark: |
| Earlier snapshots | :x:                |

## Reporting a vulnerability

**Please do not report security issues in public issues, pull requests or
discussions.**

Report them privately in one of two ways:

- through GitHub's private vulnerability reporting, using **Report a
  vulnerability** on the repository's **Security** tab; or
- by email to andromedeyz@hotmail.com, with `libsci security` in the
  subject line.

Please include:

- the version or commit you tested, and your platform: operating system,
  CPU, GPU and driver and compiler;
- the smallest program that reproduces the issue;
- what an attacker gains: code execution, reading or writing memory outside
  what was given, denial of service or something else.

### What to expect

| Step                                                | Target                                      |
|-----------------------------------------------------|---------------------------------------------|
| Acknowledgement of your report                      | within 3 business days                      |
| Initial assessment and severity                     | within 10 business days                     |
| Fix for a confirmed critical or high-severity issue | as soon as possible, usually within 30 days |

You will be kept informed while the issue is being fixed. Please allow up to
90 days for a fix before disclosing the issue publicly. Unless you ask to
stay anonymous, the fix's entry in [CHANGELOG.md](CHANGELOG.md) will credit
you.

## Security model

### What the library guarantees

- **Generated code computes the expression and nothing else.** The CPU JIT's
  machine code and the GPU JIT's shaders are generated from an expression's
  bytecode alone, never from text a program hands in.
- **No files and no network.** libsci reads no file and opens no connection
  of its own; its shaders are compiled into the library.
- **GPU work finishes before the host reads it.** An array's memory reaches
  the host only after every dispatch writing it has completed.

### What you need to handle yourself

Shape and size checks in libsci are assertions (`ASSERT`), and release builds (`NDEBUG`) leave
them out. In a release build:

- an expression that mixes arrays of different shapes is not rejected and can read past the end
  of the smaller array;
- `NDArray(shape, {values...})` with more values than the shape holds writes past the end of the
  array.

If shapes or values come from an untrusted source, check them before building arrays or
expressions from them.

## Scope

In scope: problems in libsci's own code, for example:

- out-of-bounds reads or writes, use-after-free, or integer overflow in size calculations in
  `NDArray`, `Storage`, `Arena` or the expression engine;
- CPU kernels reading or writing outside the arrays they were given;
- GPU dispatches reading or writing outside their buffers, or GPU memory reaching the host before
  pending work has finished;
- the JITs: machine code (`cpu::Jit`) or shaders (`vulkan::Jit`) generated for an expression that
  do anything other than compute that expression, or executable memory left writable.

Out of scope, please report these to their own projects:

- the Vulkan loader, MoltenVK and GPU drivers;
- Slang (`slangc` and its runtime library), which CMake downloads into `vendors/slang`;
- asmjit, which CMake downloads into `vendors/asmjit`;
- the C++ standard library and the compiler.

Results that are wrong but cannot be used to attack anything, such as a numerical accuracy issue,
belong in a normal issue.
