# TinyChain Node.js client

The Node.js client gives JavaScript and TypeScript publishers the same ergonomic surface that the Python client provides while keeping the runtime single-threaded and light. It wraps the TinyChain standard library of data structures (e.g., `Table`, `BTree`, `Tensor`, `Op`, `Txn`) with idiomatic async APIs that speak the same IR envelopes and capability manifests used by the Python and Rust hosts.

## How it parallels the Python client

- **Standard library access.** Exposes the same collection builders, ops, and transaction helpers so examples translate directly between Python and JavaScript.
- **Shard-aware helpers.** Provides client-side routing for shards across blocks and hosts using the same manifest contracts as Python; hosts remain shard-local.
- **IR parity.** Uses the TinyChain IR serialization contracts so manifests, claims, and transactions can be issued interchangeably from Python or Node.

## How it differs

- **Runtime model.** Runs in Node’s single-threaded event loop with no PyO3/native bindings. All I/O and tensor ops use async streams/promises to avoid blocking the loop.
- **Packaging.** Distributed as an npm package for Node.js. No global installer or system service is required.
- **Execution context.** Ships pure JavaScript/TypeScript APIs first; native acceleration is opt-in via user-provided handlers rather than bundled by default.

## Browser and edge integration

- **WASM runtime option.** Packages a WASM runtime so publishers can target browsers or edge workers while reusing the same IR manifests. Proprietary modules stay encapsulated in WASM with fetch-based transports (HTTP/WebSocket/WebTransport).
- **Browser TinyChain client.** The npm package exposes an ES module build that runs in modern browsers. It reuses the WASM bindings for low-level IR ops and falls back to `fetch`/`WebSocket` transports for requests. Configuration mirrors Node.js (`TinyChainClient.fromEnv(...)`) but loads endpoints/auth from `.env` (via Vite/Next) or manifest JSON at runtime.
- **Node proxy mode.** For apps that need SSR or server-side proxies, the Node client can host a lightweight API (`tinychain-express`) that forwards requests to TinyChain hosts while caching capability tokens and enforcing the same namespace rules.
- **WebSocket helper (planned).** Once `tc-server` exposes the WebSocket adapter (`--features ws`), the JS client will provide `tinychain.ws` helpers that automatically upgrade connections (Node and browser builds) for streaming dashboards or queue monitoring, falling back to HTTP when WS is unavailable.
- **No bundled UX.** UI components (React/React Native) live in downstream kits; the Node client focuses solely on protocol, storage, and compute ergonomics.

## Out of scope

- UX widgets or design systems (handled by React Native or other UI libraries).
- Alternative runtimes that add worker pools or native bindings by default; keep the baseline minimal.

## Roadmap fit

- **Parity with Python**: keep schema/manifest handling identical so libraries and examples stay portable.
- **Shard-aware collections**: finalize host- and cross-host routing helpers that mirror the Python client’s plans without moving routing into the host.
- **Edge-ready WASM**: ship a browser-friendly bundle for proprietary or low-latency deployments.
- **Multimedia/RTC enablement**: once backend multimedia handlers land, wire the Node client to drive real-time collaboration and media flows over the TinyChain protocol.
