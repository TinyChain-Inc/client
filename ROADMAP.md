# Client repo roadmap

This directory is intended to become the standalone TinyChain client repository.

Today, parts of the client tree (especially the optional PyO3 backend) assume a
parent TinyChain runtime checkout is present (e.g., to build WASM artifacts or
to compile the kernel as a Rust dependency). This document tracks the work
required to make `client/` self-contained.

## Goal: standalone builds

From a standalone `client` checkout, users should be able to:

- Install and use the pure-Python client (`py/`) with no Rust toolchain.
- Optionally build and install `tinychain-local` (`rust/`) via `maturin`
  without requiring a parent runtime repository layout.

## Current blockers

- `tinychain-local` depends on the TinyChain kernel crate (`tinychain`, currently
  provided by the runtime repo) as a workspace-relative path dependency.
- The runtime crates it depends on (`tinychain` / `tc-ir` / related crates) are
  not yet consumed as published crates in this client tree, so the client cannot
  pin them by version/revision in a standalone build.

## Planned work

1. Publish the runtime crates needed by `tinychain-local`
   - Publish (or otherwise make consumable via versioned dependencies) the Rust
     crates required to build the kernel with PyO3 support.
   - Ensure features used by `tinychain-local` remain minimal (e.g., avoid
     bringing in an HTTP server when building an in-process backend).

2. Replace workspace-relative dependencies in `rust/`
   - Switch `rust/` from `path = ...` to a versioned dependency on the
     published kernel crate (or a pinned git revision, if that is the interim
     distribution mechanism).
   - Keep the dependency surface aligned with the kernel feature model:
     `default-features = false` + explicit features for PyO3 and WASM execution
     as required by the Python tests and installer workflows.

3. Make Python workflows repo-local
   - Ensure docs and scripts in `py/` do not assume they run from a parent
     runtime repo root.
   - Keep any workflows which *do* require the runtime checkout explicitly
     labeled as such (building example WASM artifacts, running end-to-end tests
     against a runtime).

4. Add minimal CI coverage for the split
   - Pure-Python lint/test path for `py/`.
   - Optional `maturin` build path for `rust/` gated on Rust + Python
     toolchains.
   - Verify import ergonomics remain stable (`import tinychain as tc` with and
     without the optional local backend installed).

## Non-goals

- Duplicating the TinyChain runtime source tree inside the client repository.
- Adding new transport protocols or bespoke HTTP clients which bypass kernel
  dependency/egress enforcement.
