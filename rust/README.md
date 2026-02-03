## tc-local (PyO3 local backend)

This crate builds the optional Python extension module `tinychain_local` (distributed as the
`tinychain-local` wheel). It is intentionally a thin wrapper around the Rust kernel crate
(`tc-server` / `tinychain`) so the base `tinychain` Python package can remain pure-Python.

The Python package `tinychain` will import `tinychain_local` when installed and re-export
`KernelHandle`, `KernelRequest`, etc. at the top level for v1-style ergonomics (`import tinychain as tc`).

Today, `tinychain-local` depends on the TinyChain runtime crate via a workspace-relative
path, so building the extension assumes this repo is vendored into a runtime checkout.

If you are developing without that parent checkout, you will need to either:

- check out the TinyChain runtime repository so the relative dependency paths in
  `Cargo.toml` resolve, or
- temporarily adjust the dependency paths in `Cargo.toml` to match your local
  layout.
