# TinyChain Python Client

This directory holds the refreshed Python client plus its PyO3-backed
integration tests.

See [the Class parity and migration guide](CLASS_PARITY.md) for canonical
user-defined Class inheritance, prototypes, construction, and bound methods.

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

- **Publisher, resource_name, and version required.** `publisher`, `resource_name`,
  and `version` are canonical static class metadata; `resource_name` is the library
  name path component in `/lib/{publisher}/{resource_name}/{version}`. Python class
  names never define public resource identity, so every concrete library must declare
  `resource_name` explicitly. Missing publisher IDs, resource names, or semantic
  versions are a programmer error; the helper raises early so manifests stay
  deterministic.
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

## State model and hierarchy

`tc.state.State` is the universal symbolic state-machine node in the Python
client. It is not a scalar-only type and must not impose scalar semantics.

- `Scalar` is one `State` subclass for scalar IR forms.
- `Collection` (including `Tensor`, `BTree`, and upcoming `Table`) is another
  `State` branch for collection IR forms.
- Control-flow and op references (`TCRef`, `OpRef`, `Cond`, `While`,
  `ForEach`, etc.) are symbolic references that compile into the same TinyChain
  IR/state machine, not eager Python values.

Equality semantics:
- `State` does not define Python boolean equality semantics.
- Symbolic equality is expressed through TinyChain ops (`eq`/`ne`) and resolves
  at runtime.
- Tests/tools that need structural checks should inspect canonical forms with
  `form_of(...)`, not Python `==` and not a `to_json()` round trip.

Serialization semantics:
- `State.to_json()` serializes the canonical underlying form directly.
- It does not coerce through `Scalar(...)`; collections and other non-scalar
  state types remain in their own branch.
- Call `to_json()` only for an actual transport/export boundary or a test of the
  wire contract. Do not use serialization for comparison, hashing, cloning,
  validation, reference construction, or local delegation.

Network semantics:
- A given application describes one distributed transactional state machine.
- Client symbolic graphs compile to canonical TinyChain IR and execute under
  host/kernel transaction control across the TinyChain network.

## Canonical application call path

Use this order as the default application path:

1. Define routes with `tc.Library` and `@tc.get` / `@tc.post`.
2. Call bound route methods directly.
3. Use `with tc.backend(...)` to select local/remote execution context.

Treat these as advanced APIs (for framework internals, adapters, or explicit
plan execution tooling), not the ordinary app path:

- `tc.execute(...)`
- `tc.Host.execute(...)`
- `tc.Host.request(...)`

### Canonical route stub call shape

Prefer keyword arguments for route parameters:

```python
result = library.route(name="Ada", count=3)
```

When you need to pass one explicit payload object, use `body=`:

```python
result = library.route(body={"name": "Ada"})
```

Route parameters are keyword-only. Positional arguments are rejected so every
route has one unambiguous call shape.

## 60-second Greeter demo shape

The ordinary demo path should stay host-generic: define a `Library`, call it
locally, install it on any authorized remote host, then open a browser-callable
route URL. A testnet host is just a host URL; no `tc.testnet` API is required.

```python
import tinychain as tc

class Greeter(tc.Library):
    publisher = "demo"
    resource_name = "greeter"
    version = "0.1.0"

    @tc.get
    def hello(self, name: str) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)

greeter = Greeter()
host = tc.Host("https://host.example", token=install_token)

tc.install(greeter, remote=host)
print(host.url(greeter, "hello", name="Ada"))
```

The token can be pre-generated for a short video, or minted with
`tc.auth.mint_rjwt_token(..., ttl_secs=3)`. TinyChain v0.17 mints Falcon-512
RJWT tokens by default via the `rjwt` PyO3 package; Ed25519 is retained only for
legacy credential rotation. `tc.install(...)` derives the
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
    resource_name = "greeter"
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
- explicit `tc.Tensor` remains a symbolic `/state/collection/tensor` value for
  deferred planning and route reflection.
- explicit `tc.Ref` remains `tc.Ref` for op/reflection routes.

`tc.String` is the value-module `String(Value)` type. It is the only value type
with `render(...)`; use it for string templating instead of custom payload logic
or placeholder `Ref[str]` wrappers.

### Tensor Method Surface

Use `tc.Tensor` in route signatures and method bodies for symbolic tensor
authoring. Its methods mirror the v1 Tensor ergonomics while compiling to
canonical TinyChain op references:

`Tensor` now lives under `tinychain.collection` (`tc.collection.Tensor`), with
`tc.Tensor` as the canonical shorthand.

```python
class Math(tc.Library):
    publisher = "demo"
    resource_name = "math"
    version = "0.1.0"

    @tc.post
    def mm(self, left: tc.Tensor, right: tc.Tensor) -> tc.Tensor:
        return (left @ right).reshape([2, 2])
```

The initial v2 port covers the method-definition surface only: `shape`, `dtype`,
`ndim`, `size`, `all`, `any`, `broadcast`, `cast`, `copy`, `expand_dims`,
`cond`, `max`, `min`, `mean`, `norm`, `product`, `reshape`, `slice`, `std`,
`sum`, `transpose`, `write`, arithmetic operators, logical operators,
`matmul`, `tile`, `split`, `concatenate`, and `einsum`. Do not add per-package
Tensor wrappers or deferred flags; execution mode still comes from
`tc.backend(..., mode=...)`.

### Autodiff Transform

```python
class Math(tc.Library):
    publisher = "demo"
    resource_name = "math"
    version = "0.1.0"

    @tc.post
    def matmul(self, left: tc.Tensor, right: tc.Tensor) -> tc.Tensor:
        return left @ right

with tc.backend(mode="deferred"):
    op = Math().matmul(left=tc.state.id("left"), right=tc.state.id("right"))
    grad_op = tc.grad(op, wrt=("left", "right"))
```

Autodiff follows a JAX-like call-site transform model: routes define ordinary
TinyChain computation, and the autodiff compiler decides `wrt`, traversal,
fanout, and accumulation when `tc.grad(...)` is called. Do not create
autodiff-specific `get`/`post` decorators or route-level `rule`/`wrt` metadata.
TensorGraph targets return experimental Python-owned derivative programs. Bound
route targets use the experimental route derivative discovery path below.
All other target forms still fail with `AutodiffError("autodiff_not_implemented", ...)`.

Python route implementations are compiled by `tinychain._autograph`, which lowers
method source code into TinyChain IR. Route decorators capture source at definition
time when Python exposes it, so normal files and notebook-style environments with
source-backed cells work. Truly source-less generated functions still cannot be
installed as Python route implementations; use a normal source-backed method, a
stub backed by remote/WASM execution, or an explicit TinyChain op definition.

Route source capture is not automatic route-body tracing for autodiff. Discovery
does not inspect Python route bodies to infer derivative rules, does not compile
or execute user route code during discovery, and does not install derivative
routes during discovery.

Phase 5 real execution is still experimental, but supported tensor graph targets
now follow a single-installed-route model: `generate(...)` produces a
Python-owned `DerivativeProgram`, `compile_derivative_program(...)` lowers that
program to normal TinyChain route IR, `build_derivative_execution_library(...)`
wraps the compiled route in an installable `Library`, and
`DerivativeExecutionDispatcher` installs and calls that one route through the
active local backend. Production execution must not drive derivative nodes one by
one from Python; `ExecutionScheduler` and injected NumPy dispatch remain unit-test
seams.

A minimal real-execution shape looks like this:

```python
import tinychain as tc
from tinychain.autodiff import (
    AddOperator,
    DerivativeExecutionDispatcher,
    DerivativeMetadata,
    DerivativeProgram,
    TensorNodeRecord,
    build_derivative_execution_library,
)

metadata = DerivativeMetadata(
    source_graph_id="graph",
    transform_version="0.1.0",
    tensor_op_contract_version="0.1.0",
    wrt_signature=("x",),
    seed_contract="seed matches output",
)
program = DerivativeProgram(
    nodes=[
        TensorNodeRecord(
            node_id="n0",
            output_value_id="gradient",
            operator=AddOperator(),
            op_params={},
            input_value_ids=["seed", "other"],
        )
    ],
    gradients={"x": "gradient"},
    output_gradients=["gradient"],
    metadata=metadata,
)
execution_library = build_derivative_execution_library(
    publisher="demo",
    class_name="AddDerivativeExecution",
    version="0.1.0",
    program=program,
)
dispatcher = DerivativeExecutionDispatcher(
    library_cls=execution_library,
    kernel=local_kernel,
    token=install_token,
)
result = dispatcher.execute(
    program,
    values={"seed": seed_tensor, "other": other_tensor},
)
(gradient,) = result.gradients
print(gradient.values)
```

For this program, the decoded gradient is the normal TinyChain tensor result of
`seed.add(other)`, returned from a single installed `execute` route call.

Phase 5 supports Add/Sub/Mul/Div, Matmul, Transpose, Sum, Mean, Reshape,
Broadcast, broadcast reduction, symbolic shape metadata with runtime or explicit
shape bindings, explicit multi-output graphs, ordered multi-`wrt` gradients, and
remote route metadata discovery where class-level or bound-instance metadata is
available. Max, Min, and Product are recognized but fail clearly when a VJP is
not expressible with the current route primitives.

Multi-output and ordered multi-`wrt` selection are explicit:

```python
from tinychain.autodiff import generate

program = generate(
    graph,
    output_value_id=["loss", "auxiliary"],
    wrt=["right", "left"],
    seed=["loss_seed", "auxiliary_seed"],
)
# result.gradients follows wrt order: right gradient, then left gradient.
```

Current exclusions are part of the API contract: complex tensor dtypes,
slice/general-view gradients beyond reshape, stable backward-compatible
`tc.grad` guarantees, backend artifact registry behavior, server/Rust derivative
changes, and automatic tracing of arbitrary Python route bodies remain out of
scope.

Autograph enforces strict TinyChain-only symbol usage inside compiled route
expressions. Names must resolve to route parameters, prior local bindings,
`self`, or `tc`; non-TinyChain globals (for example `urllib`, `tensorflow`/`tf`,
`jax`) are rejected at compile time with an `AutographNameError`.

### Public typed Tensor tracing (experimental)

`tc.autodiff.TensorGraphBuilder` can capture an ordinary typed Tensor expression
directly, so application code no longer needs to hand-build graph records,
operator descriptors, or raw type-spec dictionaries to reach `generate(...)`.
This surface is **experimental** and lives under `tinychain.autodiff` only;
there is no top-level `tinychain` alias for it, and it does not carry a
post-0.x stability guarantee.

#### Typed inputs and the linear-MSE example

Declare each traced input with `input(name, *, dtype, shape)` while the builder
is the active trace context, then write the forward computation with ordinary
Tensor expressions:

```python
import tinychain as tc

# A trace that chains intermediate results (for example `predictions - labels`)
# must run inside an active binding context so each intermediate becomes a
# referenceable id — this is the ordinary symbolic-IR requirement, not specific
# to tracing.
with tc.state.scoped_context():
    with tc.autodiff.TensorGraphBuilder() as trace:
        images = trace.input("images", dtype="f64", shape=(4, 65))
        weights = trace.input("weights", dtype="f64", shape=(65, 10))
        labels = trace.input("labels", dtype="f64", shape=(4, 10))

        predictions = images @ weights
        residual = predictions - labels
        loss = (residual * residual).mean([0, 1])

# `vjp(...)` runs after the trace context has exited; it reads recorded metadata
# and does not need the binding context.
weight_vjp = trace.vjp(
    loss,
    wrt=(weights,),
    seed="loss_seed",
    graph_id="typed-linear-mse",
)
```

`input(...)` returns an ordinary symbolic `Tensor`; `name` must be a unique,
non-empty, non-keyword Python identifier (generated derivative route parameters
use it), and graph input order is declaration order. The public call accepts
`dtype` and `shape` directly and never accepts a raw type-spec dictionary.

#### Direct `generate(...)` interoperability

`build(outputs=...)` and `value_id(...)` produce a graph and value IDs that work
with the existing module-level `generate(...)` signature, so the high-level
`trace.vjp(...)` above is equivalent to:

```python
graph = trace.build(outputs=loss)
weight_vjp = tc.autodiff.generate(
    graph,
    graph.outputs[0],
    [trace.value_id(weights)],
    "loss_seed",
    graph_id="typed-linear-mse",
)
```

Both paths return a `DerivativeProgram`; neither creates graph records, operator
objects, or raw type-spec dictionaries in application code. The generated weight
gradient path carries the framework-owned transpose/matmul derivative structure.

#### Trace lifecycle

- **Enter/exit:** capture is active only inside the `with ... as trace:` block.
  The active builder is stored in a `ContextVar`; `tc.autodiff.get_active_builder()`
  is `None` before entry and after exit (including after an exception unwinds the
  block).
- **Single-trace reuse:** a `TensorGraphBuilder` records exactly one trace.
  Re-entering the same builder after its context has exited raises `RuntimeError`.
- **Nested rejection:** entering any builder while another builder is already
  active raises `RuntimeError`; only one trace is active at a time.
- **`vjp(...)` / `build(outputs=...)` after exit:** the high-level `trace.vjp(...)`
  surface is callable only after the trace context has exited successfully. It
  re-runs typed finalization on the selected path each call and never mutates the
  recorded forward graph, so repeated VJP generation with different `wrt` subsets
  is deterministic. `value_id(...)` on an object the active or most-recent
  completed builder never traced raises `ValueError`.

#### Supported operations and metadata

While a trace is active, the recorder captures exactly six operations —
**Add, Sub, Mul, Matmul, Mean, and Transpose** — as concrete operator nodes in
evaluation order. `Div`, `Sum`, and `Reshape` have VJP rules but are the
documented deferred subset and are **not** source-captured in this release; their
intentional absence is enforced by the capture-vs-VJP registry parity test.
`tc.autodiff.captured_route_operators()` returns the route-name → concrete
operator allowlist, and `tc.autodiff.captured_operator_types()` returns the
captured operator types used by the capture-vs-VJP parity check.

Forward dtype and shape are inferred for every captured node:

- **Add/Sub/Mul:** NumPy-style broadcasting; the output dtype matches the
  (equal) operand dtypes.
- **Matmul:** standard rank-≥2 matmul with a shared inner dimension and optional
  leading batch dimensions.
- **Mean:** reduction over **explicit** axes. `Tensor.mean(axes)` for a traced
  input must pass explicit integer axes (for example `loss.mean([0, 1])`);
  `axes=None` is rejected. A full reduction yields a rank-0 (`shape=()`) output.
- **Transpose:** axis permutation; the default reverses all axes.

#### Dtype and shape limits

- Only differentiable floating dtypes `f32` and `f64` are accepted. Non-floating
  dtypes fail with `AutodiffError("dtype_not_differentiable", ...)`.
- There is **no dtype promotion and no mixed-dtype arithmetic**: operands with
  different dtypes fail with `AutodiffError("dtype_mismatch", ...)`. Literal /
  scalar tensor constants are not supported in traced expressions.
- Shapes must be ranked. Each dimension is a non-negative `int` or a symbolic
  identifier string; rank zero is `shape=()`. Symbolic dimensions are supported
  and are only compatible when provably equal — unprovable broadcast, matmul, or
  reduction constraints fail with the corresponding existing categories (for
  example `matmul_shape_mismatch`, `unresolved_symbolic_shape`). Unknown rank is
  not supported: the recorder only performs static inference when every operand
  has complete ranked metadata.
- Transpose permutations must be concrete sequences of integer axes during an
  active typed trace. Runtime-valued permutations are not evaluated by the
  client and fail with `AutodiffError("invalid_permutation", ...)`.
- Typed finalization is **fail-closed**: any reachable input or captured output
  that lacks complete dtype/shape metadata raises before `generate(...)` runs,
  rather than silently returning a partial derivative.

#### Inactive behavior

When no builder is active, every one of these operations emits exactly the same
symbolic form, return type, and view metadata as before — tracing has zero effect
on payloads, dispatch, or eager/deferred execution outside a trace context. In
particular, `Tensor.mean(...)` still returns its usual value type when no trace
is active.

Downstream consumers can replace manual fixed-helper graph construction with
this typed tracing surface when they adopt it.

### Route derivative discovery

Route autodiff is a discovery and planning layer for library authors. A
`Library` subclass declares derivative metadata on the class-level
`derivative_routes` mapping. Keys are route names such as `"matmul"` or route
paths such as `"/matmul"`; values are `RouteDerivativeMetadata` instances or
JSON-compatible dictionaries with the same fields. Route decorators do not accept
autodiff-specific metadata.

```python
import tinychain as tc
from tinychain.autodiff import (
    ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
    RouteDerivativeMetadata,
)
from tinychain.graph_reflection import TypeSpec

tensor_type = TypeSpec(
    "/state/collection/tensor",
    {"dtype": "f32", "shape": [2, 2]},
)

class Math(tc.Library):
    publisher = "demo"
    resource_name = "math"
    version = "0.1.0"
    derivative_routes = {
        "matmul": RouteDerivativeMetadata(
            source_kind=ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
            is_pure=True,
            is_differentiable=True,
            input_signature=(tensor_type, tensor_type),
            output_signature=(tensor_type,),
            supported_wrt=("left", "right"),
            seed_contract="cotangent:output",
            transform_version="autodiff-v1",
            tensor_op_contract_version="tensor-op-v1",
            artifact_uri="/lib/demo/MathMatmulDerivative/0.1.0/artifact",
            artifact_digest="sha256:example",
            artifact_source_library="/lib/demo/Math/0.1.0",
            artifact_source_library_version="0.1.0",
            artifact_source_route="/matmul",
            artifact_visibility="public",
        )
    }

    @tc.post
    def matmul(self, left: tc.Tensor, right: tc.Tensor) -> tc.Tensor:
        return left @ right

plan = tc.grad(Math().matmul, wrt=("left", "right"))
```

For route targets, `tc.grad(route_target, wrt=...)` reads bound route metadata
from the local Python object and returns a `RouteDerivativePlan` when metadata is
present, pure, differentiable, tensor-shaped, floating-typed, seed-compatible,
and artifact-compatible with the source library and route. Discovery failures
raise route-specific `AutodiffError` categories: missing metadata is reported as
`non_differentiable_route`, side-effecting metadata as
`side_effecting_route_unsupported`, missing tensor dtype or shape as
`missing_dtype_metadata` or `missing_shape_metadata`, non-floating dtype as
`dtype_not_differentiable`, and unsupported derivative behavior as
`missing_derivative_behavior`.

Artifact-backed discovery validates declared artifact URI, digest, source
library, source library version, source route, seed contract, and tensor
contract metadata. Artifact visibility is surfaced on the plan as descriptive
metadata only; it does not enforce hidden/internal routes or install-time access
control. Broader route-body tracing, stable public `tc.grad` semantics, backend
execution, server-side derivative graphs, placeholder gradients, visibility
enforcement, and broad autodiff operation coverage remain deferred.

`tinychain.std.autodiff.Autodiff.grad`, `.vjp`, and `.trace` remain fail-only
placeholder methods. They raise `NotImplementedError("autodiff_not_implemented: ...")`
and do not produce identity gradients or fallback derivative plans.

Focused verification for this surface uses the route discovery, artifact, graph,
and placeholder suites from the client repo root:

```bash
python -m pytest py/tests/test_autodiff_route_discovery.py -v
python -m pytest py/tests/test_autodiff_artifact.py py/tests/test_autodiff_add.py py/tests/test_std_autodiff.py -v
```

### Derivative artifact lifecycle

Derivative artifacts are Python-owned packaging metadata for a
`tinychain.autodiff.DerivativeProgram`. Artifact metadata is separate from
`DerivativeMetadata`: `DerivativeProgram.to_dict()` stays artifact-free, and the
artifact payload wraps the derivative program rather than changing the program
schema.

Create a manifest with `artifact_manifest_from_program(...)` or by calling
`DerivativeArtifactManifest.from_program(...)`. Required manifest fields are:

- `artifact_name`, `artifact_version`, and `artifact_publisher`: the public
  library identity for the packaged derivative. The generated artifact `Library`
  declares an explicit `resource_name` derived from `artifact_name`, following the
  same explicit `resource_name` contract as ordinary TinyChain libraries; the
  generated Python class name is an implementation detail and never defines public
  identity.
- `source_graph_id`, `transform_version`, `tensor_op_contract_version`,
  `wrt_signature`, and `seed_contract`: copied from the source
  `DerivativeProgram.metadata` and validated against that program before
  digesting or payload construction.
- `visibility`: artifact metadata only. Supported values are `public`,
  `private`, and `internal`; the default is `public`.
- `digest_algorithm`: currently `sha256`.

Optional manifest fields are `artifact_digest`, `source_library`,
`source_library_version`, `source_route`, and `source_operator`.
`source_library` and `source_library_version` must be supplied together. When a
source library is known, it may be written as `publisher/name` or as a canonical
`/lib/{publisher}/{name}/{version}` path, and the artifact library exposes it
through normal `Library.dependencies`. Graph-only artifacts omit source library
dependencies.

Artifact digests are computed by `compute_artifact_digest(...)` with SHA-256 over
the canonical JSON returned by `canonical_artifact_json(...)`. The digest input
contains both `manifest` and `program`, uses sorted JSON keys with compact
separators, and forces `manifest["artifact_digest"]` to `None` before hashing
so an existing digest value cannot hash itself. The resulting `artifact_digest`
is a content/package digest, not a graph identity. `source_graph_id` still names
the derivative program's source graph and is validated separately from the
artifact digest.

`artifact_payload(manifest, program)` returns a JSON-compatible payload with two
top-level keys: `manifest` and `program`. The `program` value is exactly
`DerivativeProgram.to_dict()`. The `manifest` value is the manifest dict with a
computed `artifact_digest`. `attach_artifact_digest(...)` returns a new manifest
with that digest set without mutating the input manifest.

Use `build_derivative_artifact_library(...)` to package a derivative artifact as
a normal TinyChain `Library` subclass. The generated class has class-level
`publisher`, `resource_name`, `version`, and `dependencies`, does not define
`__init__`, and exposes one static GET route at `/artifact` whose
value is the artifact payload. The class works with the usual `compile_ir(...)`,
`library_definition(...)`, and `tc.install(...)` paths; application code should
not construct alternate artifact install envelopes.

For lifecycle comparisons, `public_artifact_identity(...)` derives the public
`/lib/{publisher}/{resource_name}/{version}` identity.
`compare_artifact_identity(existing, candidate)` treats the same public identity
plus the same `artifact_digest` as an idempotent repeat, allows different public
identities, and raises `ArtifactError("artifact_conflict", ...)` when the same
public identity has a different digest. Conflict messages identify the public
artifact path and the two digest values, and they do not include source program
payload internals.

Source dependency and metadata validation happen before digest and payload
construction. The manifest must match the derivative program's
`source_graph_id`, `transform_version`, `tensor_op_contract_version`,
`wrt_signature`, and `seed_contract`; mismatches raise
`ArtifactError("source_metadata_mismatch", ...)`. `artifact_source_dependencies(...)`
returns the normal library dependency tuple when a source library is declared,
and returns an empty tuple for graph-only artifacts.

Visibility is intentionally metadata-only. `public`, `private`, and
`internal` describe the artifact for callers and future tooling, but they do not
change TinyChain route compilation, route exposure, install authorization, or
server behavior. Derivative artifact packaging also does not add route-level
autodiff, a backend artifact registry or backend artifact schema,
hidden/internal route compiler behavior, Rust contracts, or production
`tc-server` derivative execution.

### Structured dependency analysis, extensible program lowering, and traced optimizer updates (experimental)

This surface extends public typed Tensor tracing with three additive pieces
under `tinychain.autodiff`: a structured analysis of what a selected forward
or derivative output depends on, a framework-owned traversal that hands
individual operations to consumer-supplied lowering handlers, and a small
composition helper for tracing an optimizer update as ordinary Tensor code.
All three are **experimental**, live only under `tinychain.autodiff`, and
carry no post-0.x stability guarantee, matching the rest of this section.

The framework owns graph meaning, reachability, topological order, and
selected-output handling. A consumer owns its own target representation,
its supported-operator mapping, its fusion policy, and its runtime. The
framework never imports, inspects, compares, hashes, or iterates a
consumer's target value — it only carries it from the handler that produced
it to the handlers (or the final output selection) that consume it.

#### Structured dependency analysis

`analyze_graph_dependencies(graph, *, outputs=None)` and
`analyze_derivative_dependencies(program, *, forward_graph, seed_value_ids,
outputs=None)` both return a `DependencyAnalysis`: every value the selected
outputs depend on, together with a provenance category and normalized
`dtype`/`shape` metadata. Each dependency is exactly one of:

- `DEPENDENCY_PROVENANCE_DECLARED_INPUT` — a value the forward trace declared
  as a named graph input (a parameter, training data, a label). The caller
  supplies it.
- `DEPENDENCY_PROVENANCE_SEED_INPUT` — an upstream cotangent the derivative
  transform introduced. The caller supplies it, typically as ones shaped like
  the differentiated output. Only present for a derivative-program analysis.
- `DEPENDENCY_PROVENANCE_FORWARD_CAPTURE` — an intermediate the forward graph
  produced and the derivative program consumes again. The caller must run the
  forward graph and retain this value before running the derivative program.
  Only present for a derivative-program analysis.
- `DEPENDENCY_PROVENANCE_LOCAL_VALUE` — a value produced inside the analyzed
  selection itself. Nothing to bind.

`DependencyAnalysis.declared_inputs`, `.seed_inputs`, `.forward_captures`,
`.local_values`, and `.required_inputs` (every non-local dependency) filter by
category. Ordering is deterministic and documented: dependencies are grouped
in `DEPENDENCY_PROVENANCE_ORDER`, declared inputs follow forward-graph
declaration order, seeds follow caller declaration order, forward captures
follow forward-graph topological order, and local values follow the
analyzed selection's schedule — topologically valid, with each operation
emitted as late as its consumers allow, so a producer stays next to the
operation that reads it *whenever nothing forces them apart*, and a
consumer's bounded fusion window can usually see the two together — so
repeated analyses of equivalent traces compare equal.

Size a fusion look-ahead from the qualifier, not from the adjacency. "As late
as its consumers allow" leaves a producer adjacent to its consumer only when
that consumer has one operand left to wait for. A wider fan-in necessarily
separates them: the consumer must wait for all of its operands, and only the
last operand emitted can end up beside it. An operation reading four produced
values emits `['t3', 't2', 't1', 't0', 'out']` — only `t0` is adjacent to
`out`, and `t3` is four positions away from the operation that reads it. Put
intermediate consumers in between and the same effect appears one level down:
`['t3', 't2', 'c1', 't1', 't0', 'c0', 'out']`, where `t3` is separated from
its consumer `c1` by `t2`.

A hook with `lookahead = 2` offered `t3` sees `['t3', 't2']` in either case,
and never the consumer it wants to fuse into. So a consumer whose pattern
spans a producer and a multi-operand consumer must size its window for the
fan-in it expects to meet — and must still tolerate never being offered the
pair, since `FusionHook.fuse` returning `None` is always a legal outcome and
the operation then lowers through its own handler.

A consumer never needs a private node map,
an ID-prefix convention, or a producer scan to work out what to bind; the
analysis is structural, never derived from how a value id happens to be
spelled.

Malformed selections fail closed with a categorized `AutodiffError` before any
target program is produced: `missing_dependency` (a reachable value has no
producer and no provenance), `ambiguous_producer` (one value would carry two
provenances, or two nodes produce the same value), `invalid_selected_output`
(the selection is empty or names an unknown value), `malformed_derivative_ir`
(a cycle), and `missing_dtype_metadata` / `missing_shape_metadata` (incomplete
type metadata on a value that requires it). Analysis for a forward graph
requires complete metadata everywhere, matching typed-graph finalization;
analysis for a derivative program is best-effort where the program itself is
best-effort, resolving metadata from the program's recorded value typespecs
first, then its nodes, then the forward graph, and reporting
`dtype=None`/`shape=None` only where no metadata exists anywhere for that
value — metadata that is present anywhere must still be complete.

##### When a forward-graph cycle is and is not reported

Cycle detection is reachability-scoped for a forward graph analyzed on its
own, and for the derivative program itself within a derivative-program
analysis: `malformed_derivative_ir` fires exactly when the reachable region
contains one. For the *forward graph* half of a derivative-program analysis,
it is not reachability-scoped — it is gated on whether the selection captures
anything from the forward graph at all. Capturing no forward value skips
walking the forward graph entirely, so a cycle anywhere in it, related to the
selection or not, is not reported. Capturing any forward value walks the
whole forward graph, so a cycle anywhere in it is reported, even one with no
relationship to what was captured. This is a deliberate, coarser boundary,
not a precise "was this particular cycle on the path to a captured value"
check.

#### Extensible program lowering

`lower_graph(graph, *, handlers, outputs=None, fusion=None, bind_input=None)`
and `lower_derivative_program(program, *, forward_graph, seed_value_ids,
handlers, outputs=None, fusion=None, bind_input=None)` walk the same analyzed
region and return a `LoweredProgram`: `operations` in traversal order,
`values` mapping every bound value id to the consumer's opaque target value
for it, and `dependencies` (the `DependencyAnalysis` the traversal was
derived from). `LoweredProgram.output_values` returns the target values of
the selected outputs in selection order.

A consumer supplies:

- An `OperationHandlerRegistry` with one `OperationHandler` per concrete
  `TensorOperator` type it supports. `handler.operator_type` declares the
  type; `handler.lower(context)` receives an `OperationContext` — the
  operator instance, its normalized `op_params`, its already-lowered `inputs`
  in operand order, each input's dependency provenance, and the output value
  id/typespec — and returns the consumer's opaque target value for that
  operation. Dispatch is by exact concrete operator type, never by route name
  or any other string; two operators that happen to share a route name are
  still two distinct dispatch targets.
- Optionally, a `FusionHook` (`lookahead: int`, `fuse(context) ->
  FusionResult | None`) that may recognize a local pattern over a bounded
  look-ahead window (`FusionContext.candidates`, capped at `lookahead`, always
  starting with the operation being offered) and replace it with one
  instruction. A `FusionResult` names the consumed operation ids (a subset of
  the offered candidates, including the offered operation, with no repeats)
  and the one opaque target value the fused region still owes the rest of the
  program. `FusionContext.value_of(value_id)` resolves an already-bound
  operand for the hook.
- Optionally, `bind_input`, converting each analyzed free dependency
  (`ValueDependency`) into the consumer's target value for it. The default
  binds the `ValueDependency` itself, which already carries the value id,
  provenance, and type metadata a consumer typically needs to construct its
  own runtime binding.

Every reachable operation is lowered by exactly one handler, replaced by
exactly one fusion, or the whole call fails before any target program is
returned — an injected handler may lower an operation or report it
unsupported, but it may never silently fall back to executing ordinary Python
arithmetic in its place. A fusion may only claim operands that are already
bound or produced by another operation in the same claimed set, and it may
leave at most one value still needed outside the fused region; a region that
would silently discard a second such value is rejected instead. Failures
raise a categorized `AutodiffError`: `unsupported_operator` (no handler
claims an operation, or the reachable region has an operation nothing
claimed), and `handler_contract_violation` for any other way a handler,
registration, or fusion hook breaks the seam's contract — including a
consumer's own exception type, which the framework cannot enumerate ahead of
time and therefore normalizes rather than lets escape uncategorized.
`KeyboardInterrupt` and `SystemExit` are interpreter control flow, not a
handler reporting failure, and are deliberately left to propagate rather than
being folded into `handler_contract_violation`.

### Every reachable operation needs a registered handler, fusion or not

**Precondition.** Support is checked for every reachable operation *before*
any lowering or fusion runs. An operation with no registered handler raises
`unsupported_operator` even when a fusion hook would have consumed it and the
handler would never have been called. The hook is not offered anything: the
call has already failed.

This bites precisely where a consumer least expects it. A target with no
standalone transpose instruction, which only ever wants a transpose folded
into the matmul that reads it, is exactly the shape fusion exists for — and
registering a hook for `(transpose, matmul)` while registering no transpose
handler fails with `unsupported_operator` naming the transpose.

**Why it is this way.** Fusion collapses operations that are *already*
supported into a cheaper form; it does not confer support. Making support
conditional on a hook would mean the same program lowers or fails depending
on which pattern a hook happened to recognize at run time, and an operation
the hook declined would surface as a failure deep inside traversal instead of
before it. The precondition keeps "can this program be lowered at all?" a
question answered once, up front, from the registry alone.

**What to register.** Declare the operator supported with a stub handler
whose `lower` raises:

```python
from dataclasses import dataclass

from tinychain.autodiff import AutodiffError, TransposeOperator


@dataclass(frozen=True)
class FusedOnlyTranspose:
    """Declares transpose supported; it is only ever reachable via fusion."""

    operator_type: type = TransposeOperator

    def lower(self, context):
        raise AutodiffError(
            "unsupported_operator",
            f"transpose {context.output_value_id!r} reached the standalone "
            "lowering path; this target only supports a transpose fused into "
            "the matmul that reads it",
        )


registry.register(FusedOnlyTranspose())
```

**The stub must raise — that is the point of it.** A stub that returns some
plausible target value instead would let an unfused transpose lower silently
and produce a wrong program, and nothing downstream would flag it: lowering
would report success. Raising makes the same situation a categorized failure
that names the operation, so a fusion window that turned out too narrow, or a
pattern the hook declined, is reported rather than absorbed. Raise
`AutodiffError` with a category of your own choosing — `unsupported_operator`
reads best here — and it propagates with that category intact; any other
exception type is normalized to `handler_contract_violation`.

Lowering runs dependency analysis first, so a lowering call can also fail
with any category that analysis raises: `missing_dependency`,
`ambiguous_producer`, `invalid_selected_output`, `malformed_derivative_ir`,
and the metadata categories `missing_dtype_metadata` and
`missing_shape_metadata`. For a derivative program, `malformed_derivative_ir`
is not a guarantee that every cycle is caught before lowering runs, and this
holds on both halves of the analysis: a cycle inside the derivative program
itself is reachability-scoped, so one the selected output cannot reach is not
reported either, exactly like a forward graph analyzed on its own; a cycle in
the forward graph is reported only when the selection captures some forward
value, and then the whole forward graph is checked, not just the captured
path — see "When a forward-graph cycle is and is not reported" above.
Lowering a program built from a graph with an unreported cycle can still fail
later, from whatever the fused or per-operation handlers do with the
resulting values.

#### Traced optimizer updates

`trace_parameter_update(update, *, parameter, gradient,
optimizer_inputs=None)` traces an ordinary Tensor callable — an optimizer
step — the same way an application loss is traced. `parameter`, `gradient`,
and each `optimizer_inputs` value are typed input specs
(`{"dtype": ..., "shape": ...}`) forwarded to the same typed-tracing builder
described above; `update` is called once, by keyword, with a `Tensor` for
each declared input, and must return the single updated-parameter `Tensor`
expressed with ordinary Tensor operations. The result is a `TracedUpdate`
carrying the finalized `graph`, the `updated_parameter_id`, and
`input_value_ids` (declared input name to its stable value id in `graph`),
so a caller binds runtime values by name rather than scanning `graph.inputs`.

#### The optimizer contract

`update` may be a plain callable, as above, or an `Optimizer`. `Optimizer` is
an abstract class that binds two facts a caller would otherwise keep in
agreement by hand: the update expression, and the names of the optimizer
inputs that expression reads. An implementation provides exactly two members:

- `required_optimizer_inputs` — the names of the optimizer inputs `update`
  reads. `parameter` and `gradient` are declared by every traced update and
  are never named here. An implementation may answer per instance, so
  configuration is free to decide which inputs the expression reads.
- `update(*, parameter, gradient, **optimizer_inputs)` — the expression,
  authored in ordinary Tensor operations, returning the single updated
  parameter Tensor.

An implementation that has configuration validates it in its own constructor.
There is deliberately no member for that: the contract *admits* configuration
without *mandating* it, so an implementation with none carries no empty hook.

**What an optimizer does not own.** Graph construction, dependency analysis,
lowering, provider execution, encrypted state lifecycle, and the training loop
are all outside it. It holds no state and no persistence, and it knows nothing
about dtype or shape — a caller still declares those, because they are
properties of the values being trained rather than of the algorithm. This is a
contract for one update expression, not an optimizer catalog and not a
training framework.

`SGD()` is the first and, today, the only implementation:
`parameter - learning_rate * gradient`, declaring the single required
optimizer input `learning_rate`. A consumer writes its own the same way:

```python
from tinychain.autodiff import Optimizer, trace_parameter_update

class ScaledStep(Optimizer):
    required_optimizer_inputs = ("step_size",)

    def update(self, *, parameter, gradient, step_size):
        return parameter - step_size * gradient

traced = trace_parameter_update(
    ScaledStep(),
    parameter={"dtype": "f32", "shape": (2, 3)},
    gradient={"dtype": "f32", "shape": (2, 3)},
    optimizer_inputs={"step_size": {"dtype": "f32", "shape": ()}},
)
```

**Given an optimizer, the declared inputs are checked twice — by signature
and by name.** Two different mistakes are possible, and each check catches
one:

- The **signature** check binds the `update` *method* against the declared
  inputs, catching an implementation whose parameters do not match what it
  declares. It is applied to `update` rather than to the instance because an
  optimizer is invoked through a call path that accepts arbitrary keywords:
  `inspect.signature(optimizer).bind(...)` succeeds for *any* declared input
  set, while `inspect.signature(optimizer.update).bind(...)` does not.
- The **name** check compares the declared `optimizer_inputs` keys against
  `required_optimizer_inputs`, catching a declaration that names inputs the
  expression never reads — which binding cannot catch when `update` absorbs
  keywords.

Both raise `AutodiffError("invalid_update_signature", ...)`, and a
malformed `required_optimizer_inputs` — one that is not a collection of
names, or a bare string, which would otherwise declare one input per
character — raises the same category rather than escaping as a raw
`TypeError`. All of it runs before the builder is entered and before any typed
input declaration is read, so a rejection never reaches the expression. The
result is that an optimizer is checked at least as strictly as the equivalent plain
callable: the same mistake yields the same category on both paths. A plain
callable is traced exactly as before, including a `**kwargs` callable, which
still binds any declaration and lets its own body decide.

`sgd_update(*, parameter, gradient, learning_rate)` is the compatibility path
for callers that already import the reference update as a function. It
authors nothing of its own — the expression lives in `SGD`, and the function
delegates to a shared instance of it, so the two cannot drift apart. New
callers should pass `SGD()` instead, which additionally gets the name check
above. Neither it, nor `SGD`, nor
`trace_parameter_update` constructs a graph-record or operator type directly;
doing so is forbidden, not just a style preference, and is checked by
a dedicated regression test that additionally scans the module's own
namespace so an import alias cannot quietly reintroduce direct construction
either. See "What this surface does and does not guard against ILC-style
coupling" below for the exact boundary of that check.

Update-callable well-formedness is checked once, before any input is
declared or the builder is entered, so an invalid callable's body never runs;
a signature that does not accept exactly the declared inputs by keyword
raises `AutodiffError("invalid_update_signature", ...)`; for an `Optimizer`
the same category reports a declared-input mismatch and a malformed
declaration of the required names, so the contract adds no new error
category. A callable that does
not return a `Tensor` raises `AutodiffError("invalid_update_output", ...)`
after it runs. Typed-input completeness and traced-expression shape/dtype
compatibility are not re-validated here — they are the same checks typed
tracing already performs on every other traced expression. This module
defines no optimizer catalog and no optimizer state lifecycle; it is a
composition helper, not a training loop.

#### Compatibility

These three pieces are additive to every existing serialized payload, but not
to the public error-category surface: `AUTODIFF_ERROR_CATEGORIES` grew from 20
to 26 entries to name the dependency-analysis, lowering, and traced-update
failure modes above, which is an observable change for a consumer matching
exhaustively on that frozenset. They add no new field to
`DerivativeProgram.to_dict()` or any other existing serialized payload;
`DependencyAnalysis`, `ValueDependency`, and `LoweredProgram` are in-memory
analysis results, not part of any wire format today. Existing
`TensorGraphBuilder`, VJP generation, derivative program compilation, and
`ExecutionScheduler` behavior are unchanged — dependency analysis and lowering
read a graph or program after it exists; they do not participate in tracing,
VJP generation, or execution scheduling. All new names -- `Optimizer` and `SGD`
included -- are reached the same lazily-loaded way as the artifact and
route-derivative surfaces already documented above: `import tinychain.autodiff
as autodiff` and access the name directly, or import it from the specific
submodule.

#### Extension example

A consumer registers a handler per concrete operator type it supports and
lowers a selection in one call. `my_target_ir` below stands in for whatever
representation the consumer owns — the framework never sees it:

```python
import tinychain as tc
from tinychain.autodiff import OperationHandlerRegistry, lower_graph

class AddToTargetIR:
    operator_type = tc.autodiff.AddOperator

    def lower(self, context):
        left, right = context.inputs
        return my_target_ir.add(left, right)  # opaque to the framework

registry = OperationHandlerRegistry()
registry.register(AddToTargetIR())
# ... register one handler per supported concrete operator type ...

lowered = lower_graph(graph, handlers=registry, outputs=[output_value_id])
program = lowered.output_values  # the consumer's own opaque target values
```

A consumer that wants a local fusion supplies a `fusion=` hook alongside
`handlers=`; declining a fusion (`fuse(...)` returning `None`) always falls
back to the per-operator handler path for that operation.

#### What this surface does and does not guard against ILC-style coupling

Two separate artifacts each prove a narrow, specific claim. Neither is a
general regression guard against the framework becoming coupled to a
specific consumer over time, and the boundary of each is worth stating
explicitly rather than leaving a reader to discover it later.

A dedicated regression test walks every module under `tinychain/autodiff/`
and fails if any of them **imports** a module whose dotted path names an ILC
consumer. This is a static AST scan over `import`/`from ... import`
statements — it catches `import ilc_api` and `from ilc_api.target import
Foo` written directly in an autodiff module's source. It does **not** catch
a dynamic import that only names the module at runtime, for example
`importlib.import_module("ilc_api.target")` — the module name never appears
as a literal import statement, so the AST walk has nothing to match. It also
does not attempt to detect every way framework code could informally assume
something about one consumer's target representation or physical layout
without ever importing that consumer's package by name; that remains a
review concern, not an automated one.

A separate test proves the extension seam is *usable* generically: a
throwaway, non-ILC consumer that defines its own target expression type,
registers handlers for two concrete operators, supplies one supported fusion
hook, and lowers a real traced graph through this seam using only the public
`tinychain.autodiff` names above, with no framework-private access and no
target-specific concept. That is a capability demonstration, not a
regression guard — it proves the seam *can* be used generically today, not
that it *stays* generic. A framework change that added consumer-specific
behavior to the lowering path while leaving the public names and their
signatures alone would leave this consumer passing unchanged, because its
own inputs and expectations never changed.

The related regression test guarding traced optimizer updates against
manual graph-record construction (see above) has a matching limit: it
catches a literal unaliased construction call in `training.py`'s own source,
and it additionally catches an import alias or any other name in
`training.py`'s own namespace that resolves to a graph-record type. It does
**not** catch construction that lives in another module and is only reached
at call time — a helper module built for this purpose and imported by
`training.py` binds a module object or a function in `training.py`'s
namespace, not the record type itself, so the namespace scan cannot see
through it. Nor does it catch a function-local aliased import nested inside
one of `training.py`'s own functions, since that alias is bound in the
function's local scope rather than the module's namespace. These checks are
aimed at the realistic accident, not at deliberate circumvention by someone
with commit access to this package.

#### Mean expansion (opt-in, experimental)

`tinychain.autodiff` also exports two opt-in passes that rewrite an all-axis,
rank-2 `.mean(...)` into matmuls and constants — the point being that a
backend with no reduction, broadcast, or division handler can still lower a
graph or a derivative program that reduces this way, at the cost of
registering a couple of handlers it might not otherwise need:

```python
from tinychain.autodiff import expand_mean_graph, expand_mean_derivative_program

expanded_graph = expand_mean_graph(graph)                    # forward artifact
expanded_program = expand_mean_derivative_program(program)   # gradient path
```

Each pass is called explicitly — no flag, no registry, no default behavior
changes. `expand_mean_graph_detailed` and `expand_mean_derivative_program_detailed`
return the same rewritten artifact alongside ordered `MeanExpansionRegion`
provenance records; the composable forms above are exactly those detailed
forms with the provenance dropped, so the two cannot disagree.

Neither expanded artifact needs a reduction, broadcast, or division handler,
but it is not free to lower:

| Tier | Mean form | Handlers the expanded region requires |
|---|---|---|
| Rank-preserving | `.mean(axes=[0, 1], keepdims=True)` | `FillOperator`, `MatmulOperator`, `MulOperator` |
| Rank-reducing | `.mean(axes=[0, 1], keepdims=False)` | the above, **plus** `ReshapeOperator`, restricted to trivial reshapes |

`MatmulOperator` and `MulOperator` (`right_literal` form) are already
supported by any backend that lowers ordinary derivative programs, so the
genuinely new handlers are `FillOperator` (both tiers) and a trivial
`ReshapeOperator` (rank-reducing tier only) — the latter is required for
**both** the forward artifact and the derivative program: the gradient path's
own leading seed reshape (`() -> [1, 1]`) survives expansion untouched, so a
backend that registers the reshape handler only for the forward side will
fail closed on the gradient path. Registering `ReshapeOperator` unconditionally
whenever `FillOperator` is registered avoids this trap. A consumer that wants
the smaller rank-preserving handler set writes `.mean(axes=[0, 1],
keepdims=True)` — a one-token application change with no framework cost — and
that is the recommended lower-cost form when both tiers are not otherwise
needed.

See `tinychain/autodiff/expansion.py`'s module docstring for the full
contract: the supported mean domain, both emitted regions with their true
shapes, the `FillOperator`/`FillDescriptor` schema, the broadcast-and-scale
predicate and the identity it rests on, the numerical tolerance the
reciprocal-multiply substitution implies, and why a pass promises nothing
about the differentiability of what it returns — a consumer needing both a
derivative and an expanded artifact differentiates the unmodified source
graph first, then expands only the artifacts destined for analysis and
lowering.

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
- `/state/collection/tensor` -> `tc.Tensor` with materialized backing data
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

Run it from the runtime repo root after installing the local Python extensions and
building the WASM example:

```bash
maturin develop --manifest-path deps/rjwt/rjwt-py/Cargo.toml
cargo build --manifest-path tc-server/Cargo.toml --example http_rpc_native_host
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
cargo build --manifest-path tc-server/Cargo.toml --example http_rpc_native_host
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

`tinychain-local` exposes the same native kernel implementation that powers the HTTP runtime. When you
install the optional backend and pass `data_dir=...` to `tc.kernel.with_library`,
the PyO3 layer hydrates per-library storage and registers every WASM library
found under `<data-dir>/lib/<id>/<version>`.
That means:

1. Install the library once with `tc.install(...)`.
2. Point both the HTTP server and PyO3 kernel at the same `data_dir`.
   Pass the separate `workspace=...` only when they should also share persistent
   BTree/Table state; transaction-local collection storage belongs there, never
   under `data_dir`.
3. Invoke routes from Python through `tc.backend(...)` and library route
   methods. Low-level HTTP clients are useful for adapter diagnostics, but
   application packages should not hand-build request payloads.

There is no PyO3-specific registration step; the host-owned data cache is the
single Library/artifact source of truth. If a route resolves via HTTP, it will
resolve in PyO3 as soon as its local kernel loads the same `data_dir`.

## Tensor response decoding without `tinychain-local`

`Tensor(native=...)` decoding — converting a server-returned tensor payload into a
Python object with materialized `.values`, `.dtype`, and `.shape` — requires the
`tinychain-local` PyO3 backend. When `tinychain-local` is not installed:

- Route responses that contain tensor payloads are returned as symbolic `tc.Tensor`
  references backed by the server-side IR ref; `.values` will raise `AttributeError`.
- HTTP-only users receive symbolic refs for tensor results and can pass those refs
  into further route calls or deferred plans, but cannot access raw numeric data
  client-side without installing the optional backend.

Install `tinychain-local` (built from the runtime workspace) to enable eager tensor
materialization via PyO3.

## Selecting the PyO3 backend

This workspace ships a pure-Python `tinychain` package and an optional
in-process PyO3 backend (`tinychain-local`). Application code should not import
or reference `tinychain-local` classes directly. Use the public TinyChain API and
let `tc.backend(...)` select local eager execution:

```python
import tinychain as tc

class Greeter(tc.Library):
    publisher = "demo"
    resource_name = "greeter"
    version = "0.1.0"

    @tc.get
    def hello(self, name: str) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)

kernel = tc.kernel.with_library(Greeter(), data_dir=data_dir, token=install_token)

with tc.backend(kernel, token=runtime_token):
    print(Greeter().hello("Ada"))
```

The private `tinychain._local` bridge exists only for framework internals and
low-level tests. Transaction helpers (`begin_txn`, `commit_txn`, etc.) are **not**
exposed via PyO3; the kernel remains the only owner of transaction state. Always
point both HTTP and PyO3 adapters at the **same `data_dir`** so they share the
txfs state.

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
    resource_name = "echo"
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

`Library` declarations use class-level manifest metadata as the only source of
library identity. Declare `publisher`, `resource_name`, and `version` on the
class — all three are mandatory. `resource_name` is the canonical library name
path component and is never derived from the Python class name. Do not declare a
raw class-level `name`, and do not pass a decorator or constructor metadata
override. Route names still come from method names.

```python
class Echo(tc.Library):
    publisher = "example-devco"
    resource_name = "echo"
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

**Switch route calls between eager and deferred mode**

```python
import tinychain as tc

class Math(tc.Library):
    publisher = "example-devco"
    resource_name = "math"
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

For explicit side-effect sequencing when no data dependency exists, use
`tc.after(dependency, then)` (or `tc.state.after` in scalar-only code). It records
an explicit ordering dependency and returns `then` so the original wrapper type
remains available for method chaining.

When you need reusable behavior inside the same DAG, construct an `OpRef`: it packages a
TinyChain op (and any bound arguments) so multiple `TCRef`s can reference it without
re-encoding the body. `OpRef`s compile at install time along with `TCRef`s, ensuring the
host scheduler resolves them exactly once per invocation even when referenced repeatedly.
All of these helpers serialize to the canonical `Scalar` enum defined in `tc-ir`,
keeping HTTP and PyO3 in lockstep.

For tiny, expression-only helpers, `tc.post` accepts a Python `lambda` and compiles
it into a POST `OpDef`. Complex control flow should be expressed as named routes
and canonical TinyChain IR, not as transport-specific Python callbacks.

`tc.state.Scalar` is a generic IR container and should not be treated as a
numeric type. Use `tc.Number` (and numeric tensor wrappers) for arithmetic in
client code and transforms.

## LogChain taxonomy planning note

LogChain diagnostics remain planning-level in the Python client. The intended
client contract is taxonomy-aware logging metadata that aligns with control-plane
PII policy:

- Structured user fields are classified before logging.
- Unclassified user payload fields default to PII and are not emitted as raw log
  values.
- Taxonomy labels are extensible so applications can add namespaced categories in
  addition to platform-defined labels.

The authoritative admit/reject decision for log payload safety remains on the
host/control-plane boundary; client helpers are ergonomic hints, not security
enforcement.

## ORM + graph query planning note

The Python client roadmap includes a Django-style ORM layer with Cypher-like
graph traversal semantics, built as typed fluent APIs rather than raw query
strings.

- SQL-like and Cypher-like ergonomics should come from `Model`/`ForeignKey`/
  `QuerySet` builders, not string construction/parsing.
- Foreign keys are treated as graph edges for typed forward/reverse traversal.
- Query builders compile to canonical TinyChain request/IR forms using existing
  method-typed ops and executor behavior.
- The primary client API must not expose raw graph query-string execution.

Planned ORM declarations prioritize intuitive defaults:

- Implicit FK by type annotation: a field typed as another `Model` is treated
  as a foreign key edge to that model primary key.
- Explicit FK declarations are available for non-default targets and policies.

Illustrative authoring shape (planning-level):

```python
class User(tc.Model):
    id = tc.String(primary_key=True)

class Article(tc.Model):
    id = tc.String(primary_key=True)
    author: User  # implicit FK edge -> User.id
    reviewer = tc.ForeignKey(User, to_field="id", related_name="reviews")

    class Taxonomy:
        labels = ["platform.pii.contact", "app.example_devco.content"]
        classification = "pii"
        regulatory = ["gdpr", "ccpa"]
        retention_policy = "retention.default_90d"
        redaction_policy = "redaction.hash_email"
```

The exact class/field names may evolve, but the defaults are stable: type-driven
FK inference, explicit override hooks, and versioned taxonomy metadata.

For schema/model upgrades, the client submits policy/config inputs, but
authoritative canary/soak/rollback orchestration remains in
`/service/std/rollout` on the control-plane.

Planned Tensor datastore integration follows the same contract boundaries:

- Tensor-backed persistent fields participate in typed ORM/query planning, but
  Tensor payload bindings are not implicit model foreign-key edges.
- Sparse Tensor components (coordinates/indices and values) follow the same
  taxonomy safety model: unclassified user-supplied components default to PII
  classification until explicitly classified.
- Tensor query predicates remain typed and parameterized; raw expression-string
  execution is not part of the primary client API.
- Implementation sequencing remains planner/taxonomy contract first, then
  client ergonomic expansion.

## Transaction handles

- The public Python HTTP client should never expose transaction handles (`txn_id`)
  directly; the server mints them and handles inter-service signing internally.
- `tc.Host` rejects caller-supplied transaction query parameters; use ordinary
  route calls, `tc.install`, and bearer auth instead of transaction handles.
- PyO3 bindings do not expose transaction APIs; keep transaction logic
  encapsulated inside `tc-server`.
