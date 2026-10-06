# Security Policy

## Supported versions

libsci is before 1.0 and has no tagged releases yet. Security fixes go into the `main` branch.

| Version        | Supported |
|----------------|-----------|
| `main` (0.1.x) | Yes       |
| Older commits  | No        |

## Reporting a vulnerability

Please do not open a public issue, pull request or discussion for a security problem.

Report it privately through GitHub instead: open the repository's **Security** tab and choose
**Report a vulnerability**, or go straight to
<https://github.com/ApaxPhoenix/libsci/security/advisories/new>.

A useful report includes:

- what is affected (file, function or API) and the commit you tested;
- the smallest program or steps that reproduce the problem;
- what someone could do with it, such as crash the process or read or write memory outside an
  array;
- your platform: operating system, CPU, GPU and driver, and compiler.

The report stays private while it is looked into. Once a fix is on `main`, the advisory is
published, and you are credited in it unless you would rather not be.

## Scope

In scope: problems in libsci's own code, for example:

- out-of-bounds reads or writes, use-after-free, or integer overflow in size calculations in
  `NDArray`, `Storage`, `Arena` or the expression engine;
- CPU kernels reading or writing outside the arrays they were given;
- GPU dispatches reading or writing outside their buffers, or GPU memory reaching the host before
  pending work has finished.

Out of scope, please report these to their own projects:

- the Vulkan loader, MoltenVK and GPU drivers;
- Slang (`slangc`), which CMake downloads into `vendors/slang`;
- the C++ standard library and the compiler.

Results that are wrong but cannot be used to attack anything, such as a numerical accuracy issue,
belong in a normal issue.

## Things to know when handling untrusted input

Shape and size checks in libsci are assertions (`ASSERT`), and release builds (`NDEBUG`) leave
them out. In a release build:

- an expression that mixes arrays of different shapes is not rejected and can read past the end
  of the smaller array;
- `NDArray(shape, {values...})` with more values than the shape holds writes past the end of the
  array.

If shapes or values come from an untrusted source, check them before building arrays or
expressions from them.
