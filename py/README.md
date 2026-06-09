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

The Python client derives URIs implicitly from structured inputs instead of asking users to concatenate
strings. Helpers such as `tc.Library`, route decorators, and the `tc.uri.*`
utilities accept structured publisher/version metadata and optional subpaths, then
optional subpaths) and emit canonical URIs. Never hard-code strings like
`"/service/foo/bar/1.0"`—even in tests or documentation. The builder enforces:

- **Publisher + version required.** Missing publisher IDs or semantic versions are a
  programmer error; the helper raises early so manifests stay deterministic.
- **Path normalization.** Mixed separators, repeated slashes, or `.`/`..` segments are
  rejected before requests are issued.
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

## Mixed backend execution context

Use one `with tc.backend(...):` block to run mixed local + remote calls:

- `kernel=...` handles local PyO3 execution.
- authority-qualified library links (`library.link()`) drive dependency routing.
- `tc.kernel.with_library(local_library, data_dir=...)` derives egress routes from declared dependency authorities.
- route method calls auto-execute inside the active backend.
- `tc.execute(op)` is available for explicit plan execution; ordinary application
  code should prefer route method calls.
- route stubs emit authority-qualified paths automatically when a `Library`
  instance is configured with `authority=tc.URI.parse("...")`.

This keeps method definitions transport-agnostic while giving explicit per-context
execution control for local + remote calls in the same flow.

## 60-second Greeter demo shape

The ordinary demo path should stay host-generic: define a `Library`, call it
locally, install it on any authorized remote host, then open a browser-callable
route URL. A testnet host is just a host URL; no `tc.testnet` API is required.

```python
import tinychain as tc

class Greeter(tc.Library):
    publisher = "demo"
    version = "0.1.0"

    @tc.get
    def hello(self, name: str) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)

greeter = Greeter()
host = tc.Host("https://host.example", token=install_token)

tc.install(greeter, remote=host)
print(host.url(greeter, "hello", name="Ada"))
```

The token can be pre-generated for a short video, or minted from key material
with `tc.auth.mint_rjwt_token(..., ttl_secs=3)`. `tc.install(...)` derives the
canonical `/lib` definition from the `Library` class; application code should not
construct schemas, artifacts, status wrappers, or request bodies by hand. For
WASM-backed libraries, pass `wasm=...` to the same install helper and keep the
`Library` manifest as the source of truth.

### Route Type Hints

Route signatures may use ordinary Python primitives for readability, but the
framework normalizes bound route stubs to TinyChain runtime value types. This
keeps deferred plans useful in IDEs and generated documentation.

```python
class Greeter(tc.Library):
    publisher = "demo"
    version = "0.1.0"

    @tc.get
    def hello(self, name: str) -> str:
        return tc.String("Hello, {{name}}!").render(name=name)

with tc.backend(mode="deferred"):
    plan = Greeter().hello("Ada")  # plan is a tc.String
```

The runtime type rules are:

- `str` normalizes to `tc.String`.
- numeric and boolean primitives normalize to `tc.state.Value`.
- mixed unions normalize to their greatest common TinyChain value ancestor.
- explicit `tc.Ref` remains `tc.Ref` for op/reflection routes.

`tc.String` is the value-module `String(Value)` type. It is the only value type
with `render(...)`; use it for string templating instead of custom payload logic
or placeholder `Ref[str]` wrappers.

Python route implementations are compiled by `tinychain._autograph`, which lowers
method source code into TinyChain IR. Route decorators capture source at definition
time when Python exposes it, so normal files and notebook-style environments with
source-backed cells work. Truly source-less generated functions still cannot be
installed as Python route implementations; use a normal source-backed method, a
stub backed by remote/WASM execution, or an explicit TinyChain op definition.

### Executor auth and routing contract

`tc.backend(...)` uses one remote auth rule:

1. Remote calls do not implicitly forward the backend `token`.
2. If a remote call needs auth, provide an explicit `Authorization` header on that call.

Remote target selection follows one routing contract:

1. Authority-qualified `OpRef` paths (`https://...`) dispatch to that authority.
2. Path-only canonical routes (`/lib/...`) use the active/default local host.
3. Cross-host dependency routing is declared on the library manifest and enforced
   by the local kernel; callers do not pass ad hoc remote routing tables.

See `py/examples/mixed_backend_modes.py`
for a complete framework-native demonstration (no custom request/response wrappers) of:

1. defining routes (`@tc.get`/`@tc.post`),
2. configuring authority-driven dependency routing, and
3. executing local+remote calls in one backend context.

The mixed-backend example also demonstrates minimal-scope signed auth in the
same script:

1. mint an install token with claim scope limited to local `/lib/.../a/...`,
2. install local WASM with `tc.install(library, wasm=...)`, then
3. execute local+remote calls with
   `tc.kernel.with_library(...)`.

Framework-native helpers used by this flow:

- `tc.auth.mint_rjwt_token(...)` for scoped token minting.
- `tc.auth.context()` to read authenticated route context (principal/claims/timing)
  from the framework instead of request payload fields.
- `tc.origin(...)` to normalize `scheme://host[:port]`.
- `library.link()` + `tc.kernel.with_library(...)` for authority-driven routing/auth selection.

### Framework-native authenticated route context

Library handlers should not accept auth payload fields such as `auth_token`,
`auth_public_key_hex`, or `auth_host`. Use framework context instead.

In Python route definitions:

```python
@tc.get
def evaluate(self, request: tc.Ref) -> tc.Ref:
    auth = tc.auth.context()
    # authorize against auth["principal"], auth["claims"], auth timing envelope, etc.
    return request
```

### Response typing contract

Route calls, `tc.Host.execute(...)`, and `tc.execute(...)` decode canonical
TinyChain state envelopes into typed Python values:

- self-describing scalar values (`"hello"`, `7`, `true`, `null`) -> Python primitives
- self-describing state maps/objects (`{"k": ...}`) -> `dict`
- self-describing state tuples/arrays (`[...]`) -> `list`/`tuple` as appropriate
- `/state/collection/tensor` -> `tc.Tensor` (when local backend types are available)
- `/state/scalar/op/*` -> `tc.state.OpDef` (decoded transparently)

This means callers should expect typed responses by default, not ad-hoc JSON/status parsing
or custom payload/status wrappers.

For Rust/native handlers, use `txn.auth_context()` from `TxnHandle`.
For WASM handlers, return an OpRef to `/host/auth/context` (see the
`opref_to_remote` example route `auth_context`).

Replay/time-window semantics are framework-owned:

1. bearer token verification is performed by the configured token verifier,
2. route context includes verifier time (`token_verified_at_nanos`) and transaction
   time (`txn_timestamp_nanos`), and
3. cross-host calls stay in transaction flow (no anonymous fallback).

Migration for ILC-style routes:

1. remove `auth_*` fields from request payload schema,
2. replace body-based auth parsing with `tc.auth.context()` (or `txn.auth_context()` in Rust),
3. keep route signatures focused on domain inputs only.

Run it from the runtime repo root after building the Rust examples:

```bash
cargo build --manifest-path tc-server/Cargo.toml --example http_rpc_native_host --example rjwt_install_token
cargo build --manifest-path tc-wasm/Cargo.toml --example opref_to_remote --target wasm32-unknown-unknown --release

./.venv/bin/python client/py/examples/mixed_backend_modes.py \
  --authority 127.0.0.1:8702 \
  --actor-id example-admin
```

Optional: pass `--secret-key-b64 <base64-ed25519-secret>` to pin key material
instead of using an ephemeral keypair.

### Example: local WASM -> remote dependency via PyO3

`py/examples/mixed_backend_modes.py` is the canonical end-to-end example. It demonstrates:

1. defining route stubs with `tc.Library` and `@tc.get`,
2. binding a remote dependency by authority,
3. minting minimal-scope install/runtime tokens,
4. installing a local WASM implementation with `tc.install(a, wasm=...)`,
5. executing ordinary route method calls through one `tc.backend(...)` context, and
6. switching to `mode="deferred"` only when an explicit plan is needed.

The key runtime shape is:

```python
kernel = tc.kernel.with_library(a, data_dir=data_dir, token=runtime_token)
tc.install(a, wasm=wasm_path, kernel=kernel, token=install_token)

with tc.backend(kernel, token=runtime_token):
    assert b.hello("World") == "Hello, World!"
    assert a.from_b("World") == "Hello, World!"
```

Build the Rust host and WASM example first:

```bash
cargo build --manifest-path tc-server/Cargo.toml --example http_rpc_native_host --example rjwt_install_token
cargo build --manifest-path tc-wasm/Cargo.toml --example opref_to_remote --target wasm32-unknown-unknown --release
```

Then run:

```bash
./.venv/bin/python client/py/examples/mixed_backend_modes.py
```

## WASM Install Validation

Use `tc.install(library, wasm=wasm_path, kernel=kernel, token=install_token)` for
local WASM implementations. The Python client does not expose a separate WASM
installer or schema-file CLI; the `Library` manifest remains the source of truth.

Run the WASM integration coverage with:

```bash
cargo build --manifest-path tc-wasm/Cargo.toml --example hello_wasm --target wasm32-unknown-unknown --release
PYTHONPATH=client/py .venv/bin/python -m pytest client/py/tests/test_wasm_helper.py client/py/tests/test_auth_context_integration.py -q
```

## How WASM libraries surface through PyO3

`tinychain-local` exposes the same kernel that powers the HTTP runtime. When you
install the optional backend and pass `data_dir=...` to `tc.kernel.with_library`,
the PyO3 layer hydrates per-library storage and registers every WASM library
found under `<data-dir>/lib/<id>/<version>`.
That means:

1. Install the library once with `tc.install(...)`.
2. Point both the HTTP server and PyO3 kernel at the same `data_dir`.
3. Invoke routes from Python through `tc.backend(...)` and library route
   methods. Low-level HTTP clients are useful for adapter diagnostics, but
   application packages should not hand-build request payloads.

There is no PyO3-specific registration step; the shared txfs layout is the single
source of truth for both adapters. If a route resolves via HTTP, it will resolve
in PyO3 as soon as the kernel loads the same directory tree.

## Selecting the PyO3 backend

This workspace ships a pure-Python `tinychain` package and an optional in-process PyO3 backend (`tinychain-local`). Transaction IDs are minted and validated server-side; client code must not mint or manage transaction lifecycles directly. To exercise the in-process backend, install `tinychain-local` and drive requests directly against the shared kernel:

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

## Deferred client-side `Library` definitions

For ergonomics and static tooling (type checking, IDE completion), TinyChain
infers execution mode from context. Reflection/definition code compiles
decorated calls into typed reference nodes; imperative runtime code executes
decorated calls eagerly unless a backend selects `mode="deferred"`.

There is one public route surface:

- `tc.Library` + `@tc.get`/`@tc.post` define a canonical `/lib/...` route.
- Reflection/install code compiles the route to IR.
- Imperative runtime code calls the route eagerly.
- `mode="deferred"` returns a typed plan when explicit planning is needed.

Minimal example:

```python
import tinychain as tc
import pathlib

class Echo(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.get
    def hello(self) -> tc.String:
        ...

echo = Echo()
with tc.backend(mode="deferred"):
    ref = echo.hello()   # tc.String plan
```

To install a Python-defined library into a local `data_dir`, submit it through
the canonical install helper with an authorized token:

```python
resp = tc.install(echo, kernel=kernel, data_dir=pathlib.Path("..."), token=install_token)
assert resp.status == 204
```

The same helper installs a WASM implementation of a `Library` manifest:

```python
resp = tc.install(echo, wasm=wasm_path, kernel=kernel, token=install_token)
assert resp.status == 204
```

Handlers that accept parameters (or return `tc.state.Scalar`/`tc.state.OpDef`) automatically compile to `OpDef` routes.
No explicit flag is required on the decorators.

Execution defaults to eager in imperative code:

```python
with tc.backend(kernel):
    assert echo.hello() == "hello"
```

For explicit deferred control at call-site scope (without per-method kwargs or
custom `*_op` helpers):

```python
with tc.backend(kernel, mode="deferred"):
    ref = echo.hello()
    assert isinstance(ref, tc.String)
```

Use `tc.execute(ref)` only when you intentionally want to execute a previously
constructed plan outside the normal route-call flow.

### Canonical identity vs authority

`Library` declarations use class-level manifest metadata plus class-derived
names. Declare publisher and version on the class; do not pass a decorator or
constructor `name` override. Route names come from method names.

```python
class Echo(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.get
    def hello(self, name: str) -> tc.String:
        ...
```

`Echo` maps to `/lib/example-devco/echo/0.1.0`; `hello` maps to
`/lib/example-devco/echo/0.1.0/hello`. A path-only URI targets the local/default
host. Passing `authority=tc.URI.parse("https://api.example.test")` keeps the same
canonical path but routes eager runtime calls over HTTP(S).

The Python client uses a single `tc.URI` type, which always includes a canonical
`path` and may also include an optional authority (`scheme`/`host`/`port`). Schemas and IR serialize only the canonical path.
Deployment configuration uses the authority to install dependency routes and enforce default-deny egress.

## Using `While` queues for long-running work

Because the transaction owner enforces a **3-second** cap, long-running workflows must
route through a queue service implemented as a TinyChain `While` loop. Expose the
queue as an ordinary `/service/...` route and call that route through the
framework; do not hold a synchronous request open while doing long-running work.

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

## Execution snippets: deferred planning vs eager execution

Use these side-by-side examples when updating docs or answering contributor
questions about the canonical eager/deferred client model.

**Read a collection element with a low-level host request**

Prefer library route calls in application packages. Use `Host.request` for
adapter diagnostics or direct collection probes.

```python
import tinychain as tc

table = tc.uri.state(
    namespace="demo",
    path=("users",),
)
host = tc.Host("http://localhost:8702")
entry = host.request("GET", tc.uri(table, "user:123"))
name = entry["name"]
```

**Switch route calls between eager and deferred mode**

```python
import tinychain as tc

class Math(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.get
    def add(self, left: tc.Number, right: tc.Number) -> tc.Number:
        ...

math = Math(authority=tc.URI.parse("http://localhost:8702"))

with tc.backend(mode="eager"):
    value = math.add(2, 3)

with tc.backend(mode="deferred"):
    plan = math.add(2, 3)

# Use framework route calls for ordinary eager execution; long-running or
# multi-step workflows should be modeled as queue services.
```

Deferred mode returns the same TinyChain plan objects produced during reflection;
eager mode executes route calls through the active backend or the authority on the
library instance.

## Migration note: route-level `deferred` kwargs

Framework execution mode is now controlled at call-site scope, not per route method.

- Do not add `deferred=False` (or similar) kwargs to package route method signatures.
- Do not add `*_op` helpers just to expose the deferred form.
- Use `with tc.backend(..., mode="deferred"):` for deferred planning.
- Use ordinary route calls for normal eager execution.

Before:

```python
def add(self, x, y, deferred=False):
    op = self._add_route(x, y)
    return op if deferred else tc.execute(op)
```

After:

```python
def add(self, x, y):
    return self._add_route(x, y)

with tc.backend(kernel):
    value = client.add(1, 2)

with tc.backend(kernel, mode="deferred"):
    plan = client.add(1, 2)
```

If a package declared route or library names explicitly, rename the Python
class/method so the derived URI is the intended URI. The framework does not
support a parallel `name=` override.

## Compute graph payloads

For numerically heavy workloads (e.g. linear algebra and fixed-count training loops), the client
also exposes an **Op-graph** builder in `tinychain.compute`. This produces a deterministic, typed
payload intended for:

- static analysis (e.g., FLOP accounting), and
- conservative bound propagation/certification prior to execution.

The builder is intentionally separate from `TCRef` graphs: it is a self-describing DAG with
explicit type parameters (e.g., tensor shapes/dtypes) and explicit operator invocations, so a
host-side analyzer can reject graphs it cannot analyze/certify with actionable errors.

## `TCRef` helpers

Python bindings expose lightweight helpers for constructing `TCRef` graphs: every
`State`, `Value`, or composite op you reference becomes a canonical TinyChain IR
node under the hood. Execution happens on the host; HTTP and PyO3 both transmit
serialized `TCRef`s, so dependency ordering and transaction behavior stay
consistent and auditable.

When you need reusable behavior inside the same DAG, construct an `OpRef`: it packages a
TinyChain op (and any bound arguments) so multiple `TCRef`s can reference it without
re-encoding the body. `OpRef`s compile at install time along with `TCRef`s, ensuring the
host scheduler resolves them exactly once per invocation even when referenced repeatedly.
All of these helpers serialize to the canonical `Scalar` enum defined in `tc-ir`,
keeping HTTP and PyO3 in lockstep.

For tiny, expression-only helpers, `tc.post` accepts a Python `lambda` and compiles
it into a POST `OpDef`. Complex control flow should be expressed as named routes
and canonical TinyChain IR, not as transport-specific Python callbacks.

## Transaction handles

- The public Python HTTP client should never expose transaction handles (`txn_id`)
  directly; the server mints them and handles inter-service signing internally.
- PyO3 bindings do not expose transaction APIs; keep transaction logic
  encapsulated inside `tc-server`.
