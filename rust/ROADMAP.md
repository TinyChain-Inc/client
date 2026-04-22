# Rust client roadmap

## Active deliverables

1. **Standalone `tinychain-local` build path**
   - Replace workspace-relative runtime dependencies with versioned (or pinned
     git) dependencies so `client/rust` can build outside a parent runtime
     checkout.
   - Keep dependency features minimal (`default-features = false`) and enable
     only the PyO3/WASM surfaces required by client workflows.

2. **Automatic PyO3 surface for user-facing runtime types**
   - Design and implement a binding strategy so user-facing types like `State`
     and `Library` can be exposed through PyO3 automatically.
   - Avoid requiring hand-written PyO3 implementations for each individual type
     in `tc-server`.
   - Keep generated/exposed bindings aligned with the canonical runtime type
     model and client-facing docs.

3. **Client-local packaging and CI**
   - Keep `maturin` workflows runnable from the `client` repo.
   - Add/maintain CI coverage for Rust build correctness and Python import
     ergonomics with and without the optional local backend.
