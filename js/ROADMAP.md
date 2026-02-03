# Node.js client roadmap

## Active deliverables

1. **Standard library parity and IR alignment**
   - Match Python client APIs for `Table`, `BTree`, `Tensor`, and transactional helpers while emitting the same IR/state envelopes and capability masks.
   - Add fixtures/examples that demonstrate translating the same manifest between Python and Node to prove parity.

2. **Shard-aware routing (client-owned)**
   - Implement manifest-driven routing helpers for shards across blocks and hosts with deterministic fallback/health handling.
   - Keep hosts shard-local; all cross-host routing logic lives in the client library.

3. **Browser/edge bundle**
   - Produce a WASM-enabled build that uses fetch/WebSockets/WebTransport for transport, keeping the surface identical to the Node package where possible.
   - Document security/identity expectations for proprietary WASM modules in browser contexts.
   - Ship an ES module build plus a generic bundler integration guide (e.g., using Next.js, Webpack, or similar) so the TinyChain JS client can run in the browser with minimal config (env files, manifest-driven endpoints).
   - Multimedia support must reuse TinyChain transports (HTTP/WebSocket/WebTransport). WebRTC/STUN/TURN orchestration is intentionally out of scope; document how publishers can integrate their own media paths if needed without dragging TinyChain into that space.

4. **Runtime ergonomics**
   - Ensure streaming/tensor operations stay non-blocking in the single-threaded event loop and expose backpressure-friendly APIs.
   - Add observability hooks (logging/metrics) that mirror Python defaults without adding extra dependencies by default.

5. **Streaming/media readiness**
   - Once backend media handlers ship, add helpers for time-dependent access (`media.stream`, queue-backed uploads) that reuse TinyChain transports (HTTP/WebSocket/WebTransport).
   - Provide compatibility notes and sample flows that highlight any divergences from v1 behavior.
