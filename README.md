# TinyChain clients

This directory contains the TinyChain client SDKs and related tooling.

It is designed to be split into its own repository and vendored into the main
TinyChain runtime workspace as the `client/` Git submodule.

## Layout

- `py/` – Python client (`tinychain`) plus tests and tooling.
- `js/` – Node.js client notes/roadmap plus a reference UX kit under `js/ux/`.
- `rust/` – Rust-side client tooling, including the PyO3 local backend (`tinychain-local`).

## Cross-repo references

Some docs and developer workflows (PyO3 local backend, building WASM example
artifacts) depend on the TinyChain runtime sources (e.g., `tc-server`, `tc-wasm`,
and build scripts). When this repo is vendored as a submodule, those live in the
parent checkout; otherwise you will need a separate runtime checkout.
