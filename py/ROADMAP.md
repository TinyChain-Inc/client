# Python client roadmap

## Purpose

The Python client should make the ordinary TinyChain workflow obvious enough to
show in a short demo: define a `Library`, call a route locally, install the same
library on any authorized remote host, then call the remote route from normal
Python or a browser URL. The framework should hide transport, payload,
serialization, auth-context, and execution-mode mechanics unless the developer
explicitly asks for advanced control.

## Current invariants

- `tc.Library` plus `@tc.get`/`@tc.post` is the public route declaration path.
- Library identity is canonical: class metadata defines publisher and version;
  class and method names derive resource and route names.
- Route declarations do not accept path/name overrides.
- Dependencies are manifest metadata; callers do not pass ad hoc dependency or
  remote-routing tables to kernels.
- Reflection is deferred by default; imperative runtime calls execute eagerly by
  default.
- `tc.backend(..., mode="deferred")` is the single call-site switch for explicit
  planning in runtime code.
- Canonical route invocation style is keyword arguments (or one explicit
  `body=` payload). Positional route calls are compatibility-only and should not
  grow new conventions.
- `tc.install(...)` is the canonical install helper for Python libraries and WASM
  implementations.
- Auth context is framework-owned and derived from validated transport/auth state,
  never from application request bodies.
- Response encoding and decoding is framework-owned; application code should not
  implement payload/status wrappers or TinyChain state parsing.
- Transaction lifecycle ownership stays inside the kernel/host. Client helpers may
  compose plans but never mint or manage transaction handles.
- Advanced/internal helpers stay out of top-level `import tinychain as tc` unless
  they are part of the ordinary user path.
- `tc.execute(...)`, `tc.Host.execute(...)`, and `tc.Host.request(...)` remain
  advanced surfaces; route method calls are the ordinary application path.

## Current demo contract

- The Greeter demo path is generic: define a `Library`, call it locally, install
  it on any authorized `tc.Host`, then open a browser-callable route URL.
- Host targeting stays generic. Do not introduce `tc.testnet` or any network-name
  specific API.
- `tc.install(library, remote=tc.Host(...), token=...)` and
  `tc.install(library, wasm=..., token=...)` are the only install shapes shown to
  application developers.
- `tc.Host(..., token=...)` carries framework auth for install and route calls.
- `tc.Host.url(library, route, **query)` builds canonical browser URLs.
- Plain route method calls are the normal Python path. `tc.execute(...)` remains
  advanced plan execution, not the demo path.

Guardrails:
- Do not add `tc.testnet`.
- Do not add another install helper.
- Do not expose raw install payloads, schemas, artifacts, transaction IDs, or
  status wrappers in the demo path.
- Do not split ordinary Python `Library` installs into top-level schema plus
  sidecar route artifacts; the library definition itself is the install unit.
- Do not require per-method auth kwargs.
- Do not reintroduce `tc.Json`, `tc.define`, `tc.deferred`, `tc.wasm.install`,
  top-level `tc.testing`, or top-level IR compiler helpers.

## Maintenance backlog

- Keep `tinychain._autograph` internal until the transformer is validated as a
  public API.
- Keep `tc.Host.request` as a low-level primitive but avoid presenting it as the
  normal application path.
- Keep framework behavior out of `tinychain.testing`; that module should contain
  harness helpers only and remain an explicit import.
- Split large test modules by invariant when they become difficult to review,
  especially executor mode/routing behavior and kernel dependency-routing helpers.
- Remove generated artifacts such as `__pycache__` from commits and keep them
  ignored.
- Keep root and submodule docs aligned whenever public route, auth, execution, or
  serialization behavior changes.
