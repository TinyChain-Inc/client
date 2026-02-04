# TinyChain Python Client

This directory holds the refreshed Python client plus its PyO3-backed
integration tests.

From the client repo root, install the client dependencies:

```bash
python -m pip install -r py/requirements.txt
```

To run the PyO3-backed integration tests, you currently need the TinyChain runtime
workspace checked out (this repo is typically vendored there as the `client/`
submodule). From the runtime repo root, run:

```bash
scripts/install_tc_server_python.sh
```

The `requirements.txt` entry for `patchelf` keeps `maturin` builds quiet by
ensuring the utility is available inside the virtualenv.

## Canonical runtime endpoints

The Python client must treat the TinyChain runtime’s URI surface as immutable:

- `/state` (including `/state/chain` for chain-wrapped collections,
  `/state/collection` for shard-local data, `/state/scalar` plus `/state/scalar/value`,
  `/state/scalar/tuple`, `/state/scalar/map`, and the forthcoming `/state/media`
  abstraction) for built-in state structures.
- `/class` for class definitions that ship with a library/service.
- `/lib` for stateless standard libraries and WASM payloads.
- `/service` for publisher-owned stateful APIs (queues, trainers, etc.).
- `/host` for host telemetry, `/healthz` for liveness checks.

Client helpers should only compose URIs inside these namespaces; never invent
new top-level directories or alternate response envelopes when extending the
Python surface.

### Implicit URI construction

Following the reference v1 Python client, the refreshed client
derives URIs implicitly from structured inputs instead of asking users to concatenate
strings. Helpers such as `client.service(...)`, `client.library(...)`, the Autograph
decorators, and the `tc.uri.*` utilities accept `(publisher, namespace, name, version)` (plus
optional subpaths) and emit canonical URIs. Never hard-code strings like
`"/service/foo/bar/1.0"`—even in tests or documentation. The builder enforces:

- **Namespace + version required.** Missing publisher namespaces or semantic versions are a
  programmer error; the helper raises early so manifests stay deterministic.
- **Path normalization.** Mixed separators, repeated slashes, or `.`/`..` segments are
  rejected before requests are issued, mirroring the `service_uri` helpers in the legacy
  client.
- **Prefix safety.** The builder prepends `/service`, `/lib`, `/class`, etc., so callers
  cannot accidentally escape the canonical directories—even when composing URIs dynamically.

Common helpers:

- `tc.uri.service(publisher=..., namespace=..., name=..., version=...)`
- `tc.uri.library(...)`
- `tc.uri.state(namespace=..., path=(... ,))`
- `tc.uri.media(...)`, `tc.uri.healthz()`, etc.

This keeps the URI scheme consistent across adapters and prevents bespoke routing. Whenever
you add a helper that touches remote state, ensure it calls into the same URI constructors
so future publishers inherit the validation for free.

### Example: local WASM → remote OpRef via PyO3 (single file)

`py/examples/library_integration_example.py` is a single-file integration example which demonstrates:

1. Installing a local WASM library `A` (under `/lib/example-devco/a/0.1.0`) into a local `data_dir`.
2. Calling `A` in-process via the `tinychain-local` PyO3 backend.
3. Having `A` return an `OpRef` (as JSON) pointing at a remote `/lib/...` dependency `B`, which the
   kernel resolves via HTTP RPC under the standard dependency whitelist + route mapping.

The remote dependency `B` is expressed as a canonical path plus an optional `authority` (`tc.URI`). The canonical
path is what gets serialized into schemas and `OpRef`s; the authority is deployment configuration which the kernel
uses to route authorized egress.

The Python `Library` stubs use v1-style decorators (`@tc.get`, etc.) and execute under an explicit executor context:

```python
with tc.backend(kernel, bearer_token="..."):
    # deferred graph nodes (typed):
    a_ref = a.from_b("World")   # tc.String
    b_ref = b.hello("World")    # tc.String

    # eager execution:
    assert tc.execute(a_ref) == "Hello, World!"
    assert tc.execute(b_ref) == "Hello, World!"
```

Outside an executor context, calling a decorated stub returns a typed `tc.Ref[...]` value (e.g. `tc.String`) which can be passed around and executed later.

The example assumes the WASM artifact exists. From the TinyChain runtime repo root, build it with:

```bash
cargo build --manifest-path tc-wasm/Cargo.toml --example opref_to_remote --target wasm32-unknown-unknown --release
```

If you want the script to spawn the remote host automatically, from the runtime repo root build the Rust host example binary first:

```bash
cargo build --manifest-path tc-server/Cargo.toml --example http_rpc_native_host
```

Note: WASM installs now require authorization. The example expects `TC_BEARER_TOKEN` to be set
to a bearer token which includes the `/lib/...` install claim (see the token generator below).
When using signed tokens, also set `TC_TOKEN_HOST`, `TC_ACTOR_ID`, and `TC_PUBLIC_KEY_B64` so
the local PyO3 kernel can verify the signature.

You can generate a signed bearer token (and public key) with the Rust example:

```bash
cargo run --example rjwt_install_token -- \
  --host http://127.0.0.1:8702 \
  --actor example-admin \
  --lib /lib/example-devco/a/0.1.0 \
  --lib /lib/example-devco/example/0.1.0
```

Or run the helper script to generate a token, set the env vars, and execute the Python
installer + integration example end-to-end:

```bash
scripts/run_python_integration_with_token.sh
```

## WASM installer regression

The `py/tests/test_install_wasm_script.py` test exercises the
`py/bin/install_wasm.py` helper (which calls `tc.wasm.install`) against the
PyO3 kernel. It installs the bundled example schema
(from the runtime repo: `tc-server/examples/library_schema_example.json`) and a freshly built WASM module
(from the runtime repo: `tc-wasm/examples/hello_wasm.rs` ⇒ `.../release/examples/hello_wasm.wasm`) into a
temporary `data_dir`, then verifies:

1. `/lib` returns the newly installed schema.
2. `/lib/hello` dispatches into the WASM export.
3. The artifacts are persisted under `<data-dir>/lib/<id>/<version>/`.

Run it with:

```bash
# from the TinyChain runtime repo root:
cargo build --manifest-path tc-wasm/Cargo.toml --example hello_wasm --target wasm32-unknown-unknown --release

# from the client repo root:
python -m pytest py/tests/test_install_wasm_script.py -vv
```

If you call `py/bin/install_wasm.py` directly, pass `--bearer-token` to authorize the install:

```bash
python py/bin/install_wasm.py tc-server/examples/library_schema_example.json \
  tc-wasm/target/wasm32-unknown-unknown/release/examples/hello_wasm.wasm \
  --data-dir /tmp/tc-data \
  --bearer-token test-token
```

Run the build by hand before invoking pytest (or set `TC_AUTO_BUILD_WASM=1` to let
the test run the cargo build automatically when permitted). Use this test whenever
you touch the `/lib`
installer, filesystem layout, or PyO3 routing so we keep the closed-source WASM
workflow working end to end.

## How WASM libraries surface through PyO3

`tinychain-local` exposes the same kernel that powers the HTTP runtime. When you install the
optional backend and then pass `data_dir=...` to `tc.KernelHandle`/`tc.Backend`, the PyO3 layer hydrates `LibraryDir`
and registers every WASM library found under `<data-dir>/lib/<id>/<version>`.
That means:

1. Install the library once (either over HTTP `/lib` or via the
   `py/bin/install_wasm.py` helper).
2. Point both the HTTP server and PyO3 kernel at the same `data_dir`.
3. Invoke the routes from Python either through `tc.Backend` (direct
   in-process calls) or over HTTP using any client—the kernel is the same in
   both cases, so route discovery and manifest semantics stay aligned.

There is no PyO3-specific registration step; the shared txfs layout is the single
source of truth for both adapters. If a route resolves via HTTP, it will resolve
in PyO3 as soon as the kernel loads the same directory tree.

## Testing the temporary host-clock flow

While the `/host/time` service is under construction, libraries that validate
tokens can rely on the opt-in `clock::now_unverified` helper exposed by the host.
To exercise it during development:

1. Launch `tc-server` with `TC_ALLOW_UNVERIFIED_TIME=1` (or the equivalent CLI
   flag) so the kernel exposes `clock::now_unverified` to WASM guests.
2. Install your WASM library with `py/bin/install_wasm.py`.
3. From Python, mint a short-lived token (seconds-scale expiry) via your test
   harness, then invoke the library through either a minimal HTTP harness (e.g.
   `requests`) or the in-process `tc.Backend`.
4. Assert that the call succeeds before the token expires and fails after the
   expiry window elapses. This mirrors the production contract and keeps drift
   bounded while we finish `/host/time`.

When the signed time service is ready, remove the flag and run the same tests
against `clock::now_attested` to ensure your library handles both flows.

## Selecting the PyO3 backend

This workspace ships a pure-Python `tinychain` package and an optional in-process PyO3 backend (`tinychain-local`). For HTTP-based tooling/tests, use a minimal harness (e.g. `requests`) until a dedicated v2 Python HTTP client is published. Transaction IDs are minted and validated server-side; a client may need to propagate `txn_id` between requests, but it must not mint or manage transaction lifecycles directly. To exercise the in-process backend (for health checks or bespoke harnesses), install `tinychain-local` and drive requests directly against the shared kernel:

```python
import tinychain as tc

kernel = tc.KernelHandle.local(data_dir="path/to/data")
health = kernel.dispatch(tc.KernelRequest("GET", tc.uri.healthz(), None, None))
print(health.status())  # 200 when the kernel is wired correctly
```

`tc.Backend` wraps the same handle and adds the `healthz` helper. Transaction
helpers (`begin_txn`, `commit_txn`, etc.) are **not** exposed via PyO3; the
kernel remains the only owner of transaction state. Always point both HTTP and
PyO3 adapters at the **same `data_dir`** so they share the txfs state.

## Deferred client-side `Library` definitions (v1-style)

For ergonomics and static tooling (type checking, IDE completion), TinyChain keeps a client-side
deferred-execution model: calling a decorated method returns a typed reference node (not a runtime value).

There are two distinct use cases:

- `tc.Library` + `@tc.get` (stub): build an `OpRef` targeting an already-installed `/lib/...` route.
- `tc.define.Library` + `@tc.get` (definition): define and compile a library interface in Python using v1-style type hints.

Minimal example:

```python
import tinychain as tc
import pathlib

class Echo(tc.define.Library):
    @tc.define.get
    def hello(self) -> tc.String:
        ...

echo = Echo(publisher="example-devco", name="echo", version="0.1.0")
ref = echo.hello()   # tc.String (deferred)
```

To install a Python-defined library into a local `data_dir`, compile it to a tiny IR manifest and submit it through `/lib`:

```python
resp = tc.define.install(echo, kernel=kernel, data_dir=pathlib.Path("..."))
assert resp.status == 204
```

Execution is always explicit via an executor/backend:

```python
with tc.backend(kernel):
    assert tc.execute(ref) == "hello"
```

### Canonical identity vs authority

In v1, a `Library` or `Service` could be expressed either as:

- a canonical path identity (no host), suitable for compilation and local installs, or
- a full address including an authority, suitable for remote execution routing.

The v2 Python client follows the same idea using a single `tc.URI` type, which always includes a canonical `path`
and may also include an optional authority (`scheme`/`host`/`port`). Schemas and IR serialize only the canonical path.
Deployment configuration uses the authority to install dependency routes and enforce default-deny egress.

## Using `While` queues for long-running work

Because the transaction owner enforces a **3-second** cap, long-running workflows must
route through a queue service implemented as a TinyChain `While` loop. A future Python HTTP
client may expose a simple `service(...).enqueue(...)` helper; executors are just TinyChain services
whose public `While` loop pulls work and runs it outside the synchronous transaction.

```python
import requests
import tinychain as tc

base_url = "http://localhost:8702"
trainer = tc.uri.service(
    publisher="publisher",
    namespace="ml",
    name="trainer",
    version="1.0.0",
)
job = requests.post(
    f"{base_url}{trainer}/enqueue",
    json={
        "model": "resnet50",
        "dataset": tc.uri.state(
            namespace="media",
            path=("images",),
        ),
        "epochs": 50,
    },
).json()
print("queued training job", job["task_id"])
```

Workers do **not** call `claim`/`ack`; the kernel’s begin/commit cycle inside the `While`
loop handles leasing and failure recovery automatically. A queue is literally a single
`While` loop whose `state` spans many transactions, so every iteration must finish inside
three seconds before committing the next snapshot. Always push heavy work through these
services instead of trying to hold a synchronous request open, and keep the loop state
small (usually a reference into `/state/...`) so resuming on another host is trivial.

Queue entries persist under ordinary TinyChain state (e.g.,
`/state/publisher/ml/trainer/tasks/<task_id>`). When `enqueue` returns, it hands back the
path to that state so callers can poll status or fetch results later. Large artifacts live
under `/state/media/...`; the queue row only stores the reference.

## WebSocket helper (planned)

The Python client will gain an async WebSocket helper once the `tc-server` WebSocket adapter
(`--features ws`) lands. The helper will reuse the same capability tokens and route structure
as HTTP while providing streaming reads/writes for scenarios like live queue monitoring or
media chunk delivery. Until then, all requests run over HTTP; the helper will automatically
fall back to HTTP when the host does not advertise WebSocket support.

## Migration snippets: v1 deferred vs PyO3 eager

Use these side-by-side examples when updating docs or answering contributor questions about the new eager client. They keep v1’s request batching and graph semantics visible while demonstrating the PyO3 equivalents.

**Read a collection element**

```python
# v1 (deferred graph)
import tinychain as tc

client = tc.connect("http://localhost:8702")
with tc.Transaction() as txn:
    data = tc.Table(tc.uri.state(
        namespace="demo",
        path=("users",),
    ))
    txn.result = data["user:123"].name
response = client.submit(txn)

# v2 (HTTP eager; minimal harness)
import requests

base_url = "http://localhost:8702"
table = tc.uri.state(
    namespace="demo",
    path=("users",),
)
entry = requests.get(f"{base_url}{table}/user:123").json()
name = entry["name"]
```

**Batch arithmetic with reuse**

```python
# v1 (deferred graph; batching happens on submit)
import tinychain as tc

with tc.Transaction() as txn:
    x = tc.Number(2)
    y = tc.Number(3)
    txn.sum = x + y
    txn.double = txn.sum * 2
cluster = tc.connect("http://localhost:8702")
result = cluster.submit(txn)

# v2 (planned): a dedicated HTTP client may add batching ergonomics without exposing
# explicit transaction lifecycle APIs; until then, use single-request HTTP calls.
```

Batching helpers (when added) must not mint transaction IDs or expose transaction
handles as a public API. The server still interprets every request, assigns
`txn_id`, and decides when to commit/rollback based on the standard protocol
cues.

Carry forward the v1 ergonomics—graph-style reuse, batching by default, and predictable error envelopes—when expanding the eager client surface.

## Standard library surface (planned): generated vs Python-authored

The planned v2 Python client mirrors the v1 standard library but splits responsibility between the PyO3 bindings and the Python package:

- **Generated via PyO3 at build time**
  - Core standard library types and ops: `State`, `Value`, `Collection`, and primitive scalar/collection variants are exposed directly from the Rust host. The bindings are generated by the PyO3 build script so they stay in lockstep with the host kernel and require no handwritten Python stubs.
  - Host-side behaviors: request routing, validation, and storage-backed execution live in Rust and are surfaced through these bindings. New Rust standard-library additions appear automatically when rebuilt.

- **Authored in Python (mirrors v1 client behavior)**
  - Session and batching ergonomics, including the `Client` and `session` helpers that assemble batched requests, enforce auth headers, and normalize error envelopes.
  - Cluster-aware collection routing kept in the client: cross-host sharding for `BTree`, `Table`, and `Tensor` mirrors the v1 Python implementation instead of pushing routing into the host.
  - Convenience constructors and fluent helpers that map RESTful paths, encode bodies, and compose pipelines in a way familiar to v1 users.
  - Tooling and tests (e.g., WASM installer, integration harness) that exercise the bindings against a live host to keep the generated surface and Python ergonomics aligned.

When adding new surface area, prefer extending the Rust standard library (so PyO3 regenerates bindings) for canonical types/ops, and extend the Python layer only for client-owned concerns (routing, batching, ergonomics). Document any intentional divergence from the v1 Python client.

## Optional ORM & graph traversal API

The refreshed Python client ships with an **optional ORM layer** that wraps TinyChain `Table`
collections in higher-level models. The ORM automatically projects foreign-key relationships
between tables into a client-defined `Graph` data structure, giving publishers a Cypher-like
traversal API reminiscent of Neo4j while still talking to TinyChain collections under the
hood:

- Each model declares its table plus outbound edges (foreign keys). The ORM builds a graph
  catalog at import time and exposes traversal helpers such as `graph.match("User")
  .out("owns").out("trained")`.
- Traversals compile into scoped TinyChain requests with parameterized predicates, so query
  construction is immune to SQL-/JSON-/serialization-injection attacks. Callers pass typed
  parameters; the ORM encodes them via the standard TinyChain IR before hitting the host.
- Because the ORM is optional, advanced users can drop down to raw table operations or write
  bespoke IR calls. Mixing the two is supported: run a graph traversal to fetch IDs, then
  stream the underlying tables manually.

This layer stays purely client-side: no new kernel types or host adapters are required, and
the resulting requests still operate on `/state/...` collections registered by the publisher.
It complements the broader TinyChain architecture, which treats the entire network as a
queryable graph of capabilities, services, and data. The ORM simply gives Python users a
Neo4j-style surface for traversing their slice of that graph without sacrificing safety.

### Composing graph ops into composite IR

Both the ORM and lower-level Python helpers can compose multiple scalar and collection ops
into a **composite operation** before sending it to the host. The client builds a DAG of
TinyChain ops (a “graph”) and submits it as part of a library/service install (`PUT` to
`/lib/...` or `/service/...`). The kernel compiles the DAG at install time into the same
execution units used by the Rust host. At runtime, the scheduler from the reference host
(`tinychain/host`) orders the ops, enforces data dependencies, and executes them under the
standard transactional guard—exactly as if the graph were authored inside the host. This
keeps batched Python workloads aligned with native handlers without inventing new protocol
semantics.

### `TCRef` helpers

Python bindings expose lightweight helpers for constructing `TCRef` graphs: every `State`,
`Value`, or composite op you reference becomes a `TCRef` under the hood. Control-flow
helpers (`tc.if_`, `tc.for_each`, `tc.while_loop`) simply wrap the canonical `TCRef`
variants from `tinychain/host/scalar`, so when you install a library/service the kernel
sees the exact same IR as if it were authored in Rust. Execution happens on the host—HTTP
and PyO3 both transmit serialized `TCRef`s—so branch selection, loop iteration, and queue
behavior stay consistent and auditable.

When you need reusable behavior inside the same DAG, construct an `OpRef`: it packages a
TinyChain op (and any bound arguments) so multiple `TCRef`s can reference it without
re-encoding the body. `OpRef`s compile at install time along with `TCRef`s, ensuring the
host scheduler resolves them exactly once per invocation even when referenced repeatedly.
All of these helpers serialize to the canonical `Scalar` enum defined in `tc-ir`, keeping
HTTP, PyO3, and future adapters in lockstep.

## Autograph-style inline control flow

TinyChain’s Python client will support Autograph-like rewrites so publishers can use native
Python `if`/`while`/`for` constructs inside route definitions. Scope:

1. **Decorator API.** Decorators (`@tc.define.get`, `@tc.define.post`, etc.) wrap handler methods defined
   inside a `Library` or `Service`. The enclosing class declares the canonical URI
   (publisher, namespace, semantic version); decorators never take explicit paths, preventing
   divergence between handlers.
2. **AST transformer.** Based loosely on TF2/JAX Autograph, the transformer lowers Python
   control flow into `TCRef`/`OpRef` graphs:
   - `if` → `TCRef::If`
   - `while` → `TCRef::While`
   - `for`/`async for` → `TCRef::ForEach`
   - user-defined helper calls → `OpRef` (with captured args)
   Unsupported constructs (e.g., `try/except`, dynamic attribute injection) raise
   deterministic errors with remediation tips.
3. **Install-time compilation.** The decorator stages the generated DAG until the publisher
   invokes `tc.autograph.install(...)` (or `@tc.define.get(...).install()`); the client then issues
   an authorized `PUT` to `/service/...` or `/lib/...`, compiling the graph into the host
   kernel just like existing helpers.
4. **Queue enforcement.** Long-running loops automatically trigger a lint warning. Publishers
   can opt into wrapping them in a TinyChain `Queue` by adding `@tc.queue` or a decorator
   argument (`@tc.define.get(..., queue=True)`).

### Example

```python
import tinychain as tc

@tc.service(publisher="example-devco", namespace="ml", name="counter", version="1.0.0")
class Counter(tc.Service):

    @tc.define.get
    def handle(self, limit: int):
        step = tc.scalar.Value(0)
        while step < limit:
            if step % 2 == 0:
                tc.op.emit("even", step)
            else:
                tc.op.emit("odd", step)
            step = step + 1
        return tc.op.result("done")
```

The transformer produces a DAG with:

- `TCRef::While` holding the loop condition and body.
- `TCRef::If` inside the loop body for the parity check.
- An `OpRef` for `step = step + 1` reused on every iteration.

Installing the route compiles the DAG; runtime executions (HTTP or PyO3) run entirely in the
Rust host, so no Python control flow executes in production.

### Testing plan

1. **Transformer unit tests.** Feed representative functions into the transformer and assert
   the emitted `TCRef`/`OpRef` graph matches a manually authored graph.
2. **Integration test.** Add a `py/tests/test_autograph_counter.py` that installs the
   example above into a temp `data_dir`, hits it via a minimal HTTP harness, and verifies the emitted
   events/results.
3. **Queue lint test.** Ensure a loop annotated with `queue=True` gets lowered into a proper
   TinyChain queue and respects the 3-second limit.

This keeps user ergonomics close to JAX/TF2 while preserving TinyChain’s install-time
compilation and host scheduling guarantees.

## HTTP parity and optional transaction handles

- The public Python HTTP client should never expose transaction handles (`txn_id`)
  directly; the server mints them and handles inter-service signing internally.
- Session helpers (when added) should exist purely for batching ergonomics and can
  be implemented without surfacing raw transaction IDs. Keep them optional and
  document that disabling sessions yields single-request semantics identical to v1.
- PyO3 bindings do not expose transaction APIs; keep transaction logic
  encapsulated inside `tc-server`.
