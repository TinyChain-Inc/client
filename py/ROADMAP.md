# Python client roadmap

## Active deliverables

1. **Eager session ergonomics with v1 compatibility.**
   - Keep graph-style reuse and batching defaults visible (session.commit), while exposing PyO3-native methods for common math and collection operations.
   - Provide migration snippets that show one-to-one v1 deferred vs v2 eager flows for collection access, transactions, and auth headers.

2. **Shard-aware collection interface.**
   - Expose client-facing helpers for `BTree`, `Table`, and `Tensor` collections that are sharded across blocks by default and can be addressed as sharded across hosts. Cross-host routing lives in this client library; the Rust host binary continues to execute shard-local handlers.
   - Surface routing hints (hash ranges, replicas) and capability checks so callers can target specific shards or rely on automatic placement when working with `tc-collection` and `tc-chain` backends.
   - Keep error envelopes and batching semantics aligned with v1 expectations even when routing across hosts.

3. **Library install + runtime parity.**
   - Mirror the host `/lib` installer contract (streamed payloads, manifest schema) and ensure PyO3 bindings keep telemetry/billing headers intact.
   - Maintain test coverage for WASM installs and auth propagation as the runtime stabilizes.
   - Ensure publishers can declare an explicit dependency set in the manifest (library-wide), and that the installer persists those dependency edges so runtimes can enforce egress uniformly across local and remote execution.
   - Add a reference example which runs a library `A` in-process via PyO3 (no HTTP server) whose method calls a dependency `B` served by a remote HTTP host, then invokes `A` from Python and asserts the cross-host call succeeds.

4. **Python-defined library compilation (v1-style).**
   - Keep the client-side deferred execution model: Python methods build a typed graph spec using return type hints, without executing host-side effects.
   - Add a `tc.define` compiler which turns a `tc.define.Library` into a host-installable payload expressed purely in v2 primitives (`TCRef`, `OpRef`, `Scalar`, `LibrarySchema`), with no bespoke verbs.
   - Extend the `/lib` installer to accept this payload (alongside WASM) and serve the compiled routes through the same kernel dispatch path as WASM/HTTP/PyO3.
   - Enforce the same dependency whitelist + egress gates for Python-defined libraries as for WASM-defined libraries.
   - Keep compilation deterministic at install time (no adapter-specific branching/looping; long workflows remain `While` queues).

4. **Optional session ergonomics (no exposed txn IDs).**
   - Keep HTTP clients free of transaction handles; sessions are purely a batching convenience that can be disabled per request.
   - Document how session batching maps to server-managed transactions without surfacing `txn_id`, and ensure disabling sessions yields v1-equivalent semantics.
   - Maintain PyO3 bindings without transaction helpers so the kernel stays the sole owner of transaction state.

5. **LogChain helpers (planned).**
   - Expose a `tc.logchain` module that wraps `/logchain/topics`, `/logchain/subscribe`, `/logchain/export`, and `/logchain/publish`.
   - Provide both blocking (batch export) and async/streaming helpers (SSE/WebSocket) so TinyChain `Service`s implemented with the Python client can tail or emit logs as part of rollout guards, diagnostics, or governance workflows.
   - Reuse existing capability-token handling; LogChain access simply requires the appropriate tokens.

## Deferred explorations

- **Peer-assisted discovery.** Evaluate whether `tc://` or WireGuard overlay hints from the control plane should influence client routing when resolving shards or libraries in partially disconnected environments.
