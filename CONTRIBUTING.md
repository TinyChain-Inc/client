# Contributing to TinyChain Clients

This directory contains the TinyChain client SDK surfaces (`py`, `js`, `web`,
and `rust` integrations). Keep contributions focused on shared client behavior,
clear runtime boundaries, and minimal public API surface area.

## Scope and structure

- `py/` contains the Python client and Python-specific contribution details.
- `js/` contains the JavaScript client package (`@tinychain/js`).
- `web/` contains the Express + React Native Web SSR/hydration reference
  integration that consumes `@tinychain/js`.
- `rust/` contains Rust-side client tooling and local backend integration.

When changing a single client, prefer adding client-specific notes in that
client directory rather than expanding this top-level guide.

## Versioning policy

- Keep client package versions aligned with the `tinychain` crate version in
  `tc-server` unless a specific client documents a justified divergence.
- For JavaScript surfaces, keep `@tinychain/js` aligned with that same version
  line.

## Style and quality gates

- Follow the shared repository style guide in
  [`CODE_STYLE.md`](/home/haydn/Documents/tcv2/CODE_STYLE.md).
- Keep changes minimal and avoid adding fallback or transitional code paths.
- Validate only the relevant client surfaces you changed (for example:
  `client/web` uses `npm run test:all`, `client/py` uses `pytest`).

## Documentation expectations

- Update the affected client README when behavior, configuration, or API shape
  changes.
- Keep examples aligned with production behavior and avoid stale fallback
  guidance.

## Rights and licensing

By contributing to this package you represent that (a) you authored the work
(or have the right to contribute it) and (b) you transfer and assign all right,
title, and interest in the contribution to the TinyChain Open-Source Project
for distribution under the TinyChain open-source license (Apache 2.0, see the
root `LICENSE`). Contributions must be free of third-party claims or
encumbrances.
