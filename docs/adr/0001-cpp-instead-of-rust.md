# ADR 0001: C++17 for the rule core instead of Rust

## Status
Accepted.

## Context
Spec §17 recommends Rust or C++ for the rule core and search, and Rust was the
initially preferred option.

The development sandbox for this work could not obtain a Rust toolchain:
`static.rust-lang.org`, `crates.io`, `index.crates.io` and GitHub release
assets (`release-assets.githubusercontent.com`) are all unreachable from it,
and no distro package, container registry or mirror was available either. Only
PyPI, npm and the GitHub API/codeload endpoints respond.

Writing Rust anyway would have meant shipping a rule core that had never been
compiled, let alone tested — directly at odds with the goal of making M0/M1
*robust*, where spec §18.1 states that rule consistency is the single most
important requirement.

## Decision
Implement the rule core in C++17, which spec §17 explicitly permits, using the
`g++ 12` toolchain available locally. Keep the code dependency-free so it
builds anywhere with a standard compiler.

## Consequences
- The full test suite (117 tests, ~518k assertions) actually runs, under
  `-Werror`, AddressSanitizer and UBSan. Several real bugs were caught this way
  that static review would have missed.
- No `cargo`, so the build is a hand-written `Makefile` with header dependency
  tracking. Adequate for a header-mostly project.
- If the project later moves to Rust, the C++ implementation doubles as a
  reference oracle: the property and determinism tests can be run against both
  and diffed, which is a stronger position than a direct rewrite.
- Memory safety is not guaranteed by the language, so sanitizers are part of CI
  rather than optional.
