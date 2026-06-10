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
  - Implementation phases:
    1. Introduce shared bridge traits for extraction/projection and keep impls
      near type-family definitions instead of central adapter files.
    2. Migrate the Python local backend to consume those traits through a
      small registration/composition layer.
    3. Add extension fixtures proving new `Value`/`Collection` variants can be
      exposed with localized impls and no adapter-wide rewrites.
    4. Add perf regression gates to ensure the bridge migration does not add
      avoidable serialization or allocation overhead on native calls.

3. **Client-local packaging and CI**
   - Keep `maturin` workflows runnable from the `client` repo.
   - Add/maintain CI coverage for Rust build correctness and Python import
     ergonomics with and without the optional local backend.
