# Agent Notes (Node.js client)

- Mirror Python client ergonomics where possible: expose the same TinyChain standard library structures (e.g., `Table`, `BTree`, `Tensor`, `Op`, `Txn`) and IR envelopes so examples stay portable across languages.
- Keep the Node runtime **single-threaded** and minimal—no implicit native bindings or PyO3 equivalents. Use streaming/async primitives that preserve Node’s event-loop friendliness and avoid hidden worker pools unless explicitly configured.
- Keep symbolic graphs, route arguments, manifests, and local wrapper calls in
  their canonical JavaScript/TypeScript representation. `JSON.stringify` and
  `JSON.parse` belong only in the outermost remote transport, explicit artifact
  export/import, or another documented ABI boundary; they must not implement
  equality, hashing, cloning, URI/reference construction, validation, caching,
  or delegation between client modules.
- A remote request is encoded exactly once by the fetch transport and its
  response is decoded exactly once before entering the client. Preserve streamed
  bodies and async iterators rather than buffering and re-encoding them.
- Preserve transport backpressure through async iteration and fetch body streams.
  Do not add unbounded prefetch, request fan-out, retry queues, or eager collection
  materialization; every concurrency option must be finite and explicit.
- Route cross-host shards in the client layer (parity with Python): shard-aware helpers should live entirely in JavaScript/TypeScript with hosts remaining shard-local.
- Browser/edge integration should rely on WASM bundles and browser-friendly fetch transports; don’t add browser-only UX components here (those belong in downstream UI kits like React Native wrappers).
- Apply the repo-wide documentation hygiene: no calendar dates in docs; clarify divergences from v1 explicitly and prefer reuse when unsure.
- Keep roadmap items implementable and scoped (tests, examples, and compatibility notes) before coding, but commit decisively once scoped.
