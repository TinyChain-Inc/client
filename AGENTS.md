# TinyChain client repo Agent Notes

This tree is expected to become its own repository and be vendored into the main
TinyChain workspace as a Git submodule. Keep client code and docs usable in that
standalone context.

## Design expectations

- Keep each client thin and protocol-aligned: no bespoke verbs, no alternate
  request/response envelopes, and no adapter-specific transaction lifecycles.
- Favor a small set of broadly reusable primitives (URI builders, request
  envelopes, capability tokens) over one-off convenience APIs.
- Avoid cross-repo coupling: when docs or tooling depend on the TinyChain runtime
  sources, call that out explicitly instead of assuming a monorepo layout.
- Minimize dependencies and keep runtimes lightweight (especially for Node.js and
  browser targets).

## Testing

- Prefer focused tests that cover IR/URI/contracts and the most critical client
  behaviors (auth propagation, routing, install workflows).
- Keep long-running workflows modeled as TinyChain queues; client helpers should
  not hide synchronous work beyond the 3-second budget.
