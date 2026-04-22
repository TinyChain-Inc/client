# TinyChain clients

This directory contains the TinyChain client SDKs and related tooling.

It is designed to be split into its own repository and vendored into the main
TinyChain runtime workspace as the `client/` Git submodule.

## Layout

- `py/` – Python client (`tinychain`) plus tests and tooling.
- `js/` – Node.js client package (`@tinychain/js`) plus documentation.
- `web/` – web integration layer (Express + SSR/hydration examples).
- `rust/` – Rust-side client tooling, including the PyO3 local backend (`tinychain-local`).

See [`CONTRIBUTING.md`](/home/haydn/Documents/tcv2/client/CONTRIBUTING.md) for
client contribution guidelines and legal terms.

## Version alignment

- Keep client package versions aligned with the `tinychain` crate version in
  `tc-server` unless a package explicitly documents a reason to diverge.
- For JavaScript, `@tinychain/js` should track that same version line.

## Release and version bump workflow

When bumping versions, update the runtime and clients together in one change:

1. Update the `tinychain` crate version in `tc-server` (source of truth).
2. Update `client/js/package.json` to the same version line.
3. Update `client/web` defaults/docs that mention the service/client version
   (for example `.env.example`, SSR defaults, and README snippets).
4. Refresh lockfiles after any package.json changes.
5. Run the client validation surface before merge:
   - `cd client/web && npm run test:all` (Node 20.x)
   - `client/py` and `client/rust` tests relevant to changed code paths

## Cross-repo references

Some docs and developer workflows (PyO3 local backend, building WASM example
artifacts) depend on the TinyChain runtime sources (e.g., `tc-server`, `tc-wasm`,
and build scripts). When this repo is vendored as a submodule, those live in the
parent checkout; otherwise you will need a separate runtime checkout.
