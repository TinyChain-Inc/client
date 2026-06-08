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

## Next milestone: remote install and browser-link ergonomics

Motivation:
- Enable a 60-second Greeter demo with no testnet-specific API and no boilerplate
  payload/auth plumbing:
  1. generate a simple `Hello, {name}!` `Library`,
  2. call `hello` locally,
  3. install the same library on a configured remote host with an auth token, and
  4. open a browser-callable remote `hello` URL.

Deliverables:
- Extend `tc.install(...)` to accept a remote TinyChain host:
  ```python
  tc.install(greeter, remote=tc.Host("https://host.example"), token=install_token)
  ```
- Let `tc.Host` carry default auth for install and route calls:
  ```python
  host = tc.Host("https://host.example", token=install_token)
  ```
- Keep host targeting generic. Do not introduce `tc.testnet` or any network-name
  specific API.
- Accept pre-generated short-lived tokens. If key material is provided, allow
  framework token minting with a seconds-scale TTL, but do not require keypair
  handling in the demo script.
- Add a canonical browser URL builder for route calls:
  ```python
  host.url(greeter, "hello", name="Ada")
  ```
- Ensure generated route URLs use canonical `/lib/{publisher}/{class-derived-name}/{version}/{route}`
  paths and standard query encoding.
- Keep plain route method calls as the normal Python path. `tc.execute(...)` remains
  advanced plan execution, not the demo path.

Validation:
- Unit test `tc.install(..., remote=host, token=...)` builds the same canonical
  install payload as local install and sends it to remote `/lib` with auth.
- Unit test `tc.Host(..., token=...)` applies auth to install and route requests
  without per-method auth kwargs.
- Unit test browser URL generation for route paths, query encoding, and authority.
- Mocked remote-host test proving Python-defined `Library` install and route call
  use framework request/response decoding, not custom payload/status parsing.
- Example script for the Greeter demo which contains only user-facing code and can
  be shown without revisions or setup detours.

Documentation:
- Add a concise README section: "60-second Greeter demo".
- Document that the remote may be any authorized TinyChain host, including a
  testnet host, without changing API shape.
- Document token expectations: pre-generated bearer token is acceptable; optional
  short-lived minting is framework-provided when key material is available.

Guardrails:
- Do not add `tc.testnet`.
- Do not add another install helper.
- Do not expose raw install payloads, schemas, transaction IDs, or status wrappers
  in the demo path.
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
