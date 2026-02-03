# Agent Notes (Node.js client)

- Mirror Python client ergonomics where possible: expose the same TinyChain standard library structures (e.g., `Table`, `BTree`, `Tensor`, `Op`, `Txn`) and IR envelopes so examples stay portable across languages.
- Keep the Node runtime **single-threaded** and minimal—no implicit native bindings or PyO3 equivalents. Use streaming/async primitives that preserve Node’s event-loop friendliness and avoid hidden worker pools unless explicitly configured.
- Route cross-host shards in the client layer (parity with Python): shard-aware helpers should live entirely in JavaScript/TypeScript with hosts remaining shard-local.
- Browser/edge integration should rely on WASM bundles and browser-friendly fetch transports; don’t add browser-only UX components here (those belong in downstream UI kits like React Native wrappers).
- Apply the repo-wide documentation hygiene: no calendar dates in docs; clarify divergences from v1 explicitly and prefer reuse when unsure.
- Keep roadmap items implementable and scoped (tests, examples, and compatibility notes) before coding, but commit decisively once scoped.
