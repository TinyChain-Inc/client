# TinyChain Python Client

The Python client provides symbolic TinyChain values, application authoring,
deferred graph construction, HTTP access, and an optional in-process PyO3
backend. All backends use the same application definitions and native request
semantics; Python does not own transaction lifecycle or server routing.

## Install and test

From the client repository root:

```bash
python -m pip install -r py/requirements.txt
python -m pytest py/tests
```

The `tinychain_local` extension is owned by `client/rust`. From the runtime
workspace, build and install it with:

```bash
scripts/install_tc_server_python.sh
```

Tests which require local execution must fail if that extension is unavailable;
they must not silently switch to HTTP or a mock backend.

## Mental model

`tc.state.State` is the universal symbolic node. Scalars, values, collections,
references, and control flow all compile to the shared TinyChain IR. Typed
wrappers such as `tc.Number`, `tc.String`, `tc.Tuple`, `tc.Map`, and `tc.Tensor`
own their native operations.

Symbolic values do not use Python equality or truthiness as runtime semantics.
Use TinyChain operations for deferred comparisons, and use `form_of(...)` only
when tooling or tests need to inspect canonical in-memory structure.
Serialization is reserved for HTTP, persistence, WASM, or explicit Python
materialization; local PyO3 calls pass native State handles.

The implemented top-level namespaces are:

- `state` for native values, references, and collection types;
- `class` for immutable Class definitions;
- `lib` for JSON and WASM Libraries;
- `service` for persisted Service definitions and discovery;
- `host` for host capabilities and health.

Service execution, standalone named collection hosting, Chain replay/repair,
and persistent Tensor storage are not implemented. Client code must not emulate
them with alternate endpoints or registries.

## URI construction

Use `tc.uri` and `tc.URI` rather than string concatenation. Application
identities contain a kind, publisher, one or more resource segments, and a
semantic version. Publishers, resource paths, and versions are explicit class
metadata; Python class names are not public identity.

Application classes expose their validated identity through `class_id()`, and
`tc.uri(subject)` retrieves the canonical URI of a type or instance. `tc.URI`
is the explicit constructor for boundary code. These builders reject invalid or
reserved segments before a request is sent. Authority-qualified URIs select a
remote host; path-only application URIs select the active local backend.

## Author a Library

```python
import tinychain as tc

class Greeter(tc.Library):
    publisher = "example-devco"
    resource_name = "greeter"
    version = "1.0.0"

    @tc.get
    def hello(self, name: str) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)
```

Route names come from decorated member names. Decorators never accept public
paths. Route calls use keyword arguments, or `body=` for one explicit payload:

```python
greeter = Greeter()
result = greeter.hello(name="Ada")
```

`compile_ir(greeter)` produces one literal application definition whose
canonical Library URI is the sole top-level key. It does not produce a package,
artifact list, route envelope, or adapter-specific payload.

Install that same definition locally or remotely through `tc.install`:

```python
host = tc.Host("https://host.example", token=token)
tc.install(greeter, remote=host)
```

For a WASM Library, pass the raw module through the same helper. The module
contains its canonical Library definition and export bindings; it is sent as an
`application/wasm` body without base64 or multipart encoding.

## Execution backends

Bound route methods are the ordinary application API. `tc.backend(...)` selects
eager local or deferred behavior for a lexical block:

```python
with tc.backend(mode="deferred"):
    plan = greeter.hello(name="Ada")
```

Supply `kernel=` for in-process execution. Authority-qualified references use
HTTP when the active executor resolves them remotely. `tc.execute`,
`tc.Host.execute`, and `tc.Host.request` are lower-level tools for explicit
plans and transport integrations; they are not parallel application runtimes.

Python code never mints a transaction ID, commits, rolls back, or rewrites a
protocol claim. Those operations belong to the host and the most-specific
resource cluster.

## Classes

`tc.Class` declares an immutable, versioned Class value. A Class has one native
or user-defined parent and a prototype. Decorated instance methods compile to
ordinary prototype `OpDef`s whose `$self` is the concrete instance.

```python
class Message(tc.Class, tc.Map):
    publisher = "example-devco"
    resource_name = "message"
    version = "1.0.0"

    @tc.get
    def render(self, name: tc.String) -> tc.String:
        return tc.String("hello, {{name}}").render(name=name)

message = Message(prefix="hello")
```

`tc.install` submits each declared Class through the ordinary `/class` PUT path
before submitting the Library. The Library contains ordinary links to those
Classes. Each immutable install is independently idempotent, so retry is safe;
the sequence is intentionally not a second batch protocol. See [the Class
contract](CLASS_PARITY.md) for the stable authoring rules.

## Control flow and Autograph

`tc.cond`, `tc.after`, `tc.state.while_loop`, and `tc.state.for_each` construct
ordinary shared IR references. Graph order follows data dependencies; use
`tc.after` when side effects require an ordering edge that data flow does not
provide.

Decorated methods without an explicit `cxt`, `ctx`, or `txn` parameter use the
bounded Autograph transform. Its supported Python subset and rejection behavior
are defined in [DESIGN_AUTOGRAPH.md](DESIGN_AUTOGRAPH.md). An explicit context
parameter retains direct v1-style authoring.

Top-level operation contexts are lexical and single-assignment. `State` values
construct typed symbolic operations, but they are not namespaces or executors:

```python
with tc.scoped_context() as cxt:
    cxt.total = tc.Number(1)
    temporary = cxt.bind_auto(cxt.total + 1)
    op = cxt.result(temporary)
```

Assigning `cxt.total` again, binding the same value under another explicit name,
or shadowing `total` in a nested context raises `ValueError`. `bind_auto` is the
only API which generates a unique temporary name, and calling it again for the
same object returns its existing binding. Nested contexts may capture enclosing
bindings; leaving a context—normally, by exception, or by cancellation—restores
the previous context. `ContextResult.form` is an immutable snapshot.

The host validates emitted IR again. The normative binder, capture, `ForEach`,
`While`, and POST-input rules live in `tc-ir/IR_INTERFACE_GUIDELINES.md`; every
client must enforce that contract rather than inventing a language-specific
namespace.

Runtime `Ref` and request `OpRef` values are lowered to canonical IR only while
compiling a decorated route. Generic `tc.state.autobox` accepts semantic values
and canonical IR; it does not persist request descriptors or transport headers.

## Tensor and autodiff surface

`tc.Tensor` is the backend-neutral symbolic Tensor wrapper. Its operators and
methods compile to native collection routes; client APIs do not expose storage
engine names.

`tc.grad(...)` is an experimental call-site transform over canonical route IR.
Autodiff route discovery, dependency analysis, lowering, tracing, and derivative
metadata live under `tinychain.autodiff`. They do not change application install
format or add server-side artifact registries. A “derivative artifact” in this
package means Python-owned, immutable derivative-program metadata packaged as an
ordinary Library—not the removed server package/artifact architecture.

`tinychain.autodiff.compile_training_step` composes loss tracing,
differentiation, capture analysis, expansion, and lowering into one immutable
`CompiledTrainingStep`. The result contains lowered forward, derivative, and
per-parameter update programs plus their provenance. Backends supply an explicit
operation-handler registry and input bindings; the client does not install or
execute graph nodes one by one as an alternate runtime. The module contract in
`tinychain/autodiff/training_step.py` and
`py/tests/test_autodiff_training_step_*.py` are authoritative for its precise
compile sequence, supported operators, and failure categories.

The supported operation set and validation behavior are executable contracts in
`py/tests/test_autodiff_*.py`. Experimental APIs should remain labeled as such;
planned compiler or optimizer work belongs in the roadmap, not this README.

## Response and collection behavior

Remote responses decode once at the transport boundary. Native local responses
remain backed by Rust State handles until Python requests materialization.
Collection streams stay lazy; convenience APIs must not turn an unknown-length
result into an eager list or hidden prefetch queue.

The host owns the request deadline and capacity permits. Cancellation, decoding
failure, or materialization failure releases local guards but never causes the
client to select a transaction outcome.

## Related documents

- [Autograph contract](DESIGN_AUTOGRAPH.md)
- [Class authoring contract](CLASS_PARITY.md)
- [Python contributor rules](AGENTS.md)

The following superproject documents are non-normative integration context:

- [Workspace architecture](https://github.com/TinyChain-Inc/tcv2/blob/main/ARCHITECTURE.md)
- [Transaction protocol](https://github.com/TinyChain-Inc/tcv2/blob/main/docs/protocol/transactions.md)
- [Backpressure contract](https://github.com/TinyChain-Inc/tcv2/blob/main/BACKPRESSURE.md)
