# Python client Agent Notes

The Python client owns cross-host collection routing, session ergonomics, and parity
with the v1 HTTP interface. Keep it the canonical reference for client behavior while
staying thin and well-documented for new users.

## Design expectations

- Keep deferred (v1-style) and eager execution paths aligned; prefer refactors that
  share a single request envelope over adapter-specific fallbacks.
- Never add implicit authorization or install-claim shortcuts in the PyO3 path; installs
  must require an explicit bearer token so behavior matches the HTTP transport.
- Implement cluster-aware sharding for `BTree`, `Table`, and `Tensor` collections here,
  not in the host. Hosts remain shard-local.
- Preserve v1 request/response semantics (batching, auth headers, error envelopes) so
  publishers can migrate gradually. Document intentional breaks with migration notes.
- Treat the PyO3 bridge as a window into the same kernel: once a WASM library is
  installed under `<data-dir>/lib/...`, it is immediately callable via both HTTP
  and `tinychain` bindings. Avoid adding library-registration logic to the client.
- Keep v1 ergonomics in client-facing examples/tests: prefer injected context parameters
  (`cxt`/`ctx`/`txn`) over inline `cxt =` assignments, and use Python literals plus
  `tc.state.id(...)` instead of `Scalar.tuple_of`/`Scalar.id`.
- PyO3 transaction helpers (`begin_txn`, `commit_txn`, etc.) do **not** exist; do
  not reintroduce them. Transaction orchestration lives entirely inside the
  kernel.
- Enforce the 3-second temporal locality rule. Client ergonomics (sessions,
  batching, convenience helpers) must never hide long-running work; expose
  TaskQueue helpers (`enqueue`/`claim`/`ack`) so publishers push heavy workloads
  through queues instead of synchronous routes.
- Preserve server and transport backpressure. Collection responses stay lazy,
  request concurrency and prefetch are finite and explicit, and retries never
  form an unbounded queue or conceal a structured resource-exhaustion error.
- Distinguish two classes of gaps:
  framework gaps (missing native capability we want `tinychain` to provide,
  independent of v1/v2) and parity gaps (behavior differences versus v1).
- Treat v1 parity claims as contract-level requirements:
  transport parity means framework-native `Host`/executor request paths with
  bearer-header auth and shared response/error envelopes.
- Do not classify application-specific payload fields (for example route-specific
  auth fields inside a request body) as Python client framework gaps unless the
  same requirement exists across unrelated libraries.
- Keep key/token parsing helpers package-local by default; only promote them into
  `tinychain` when they are broadly reusable across multiple publishers and
  transport modes.
- Do not add custom request/response wrapper classes or hand-written payload/status
  parsing in examples or client APIs when framework surfaces already exist
  (`tc.backend`, `tc.execute`, `tc.Host`, `tc.testing.decode_json_body`).
- Treat execution mode as contextual framework behavior: reflection/definition
  contexts are deferred, imperative runtime calls are eager by default, and
  `with tc.backend(..., mode="deferred")` is the explicit planning override.
  Do not add package-level `deferred` kwargs, `*_op` helpers, or extra execution
  wrappers.
- Keep route type hints aligned with runtime TinyChain value types. Authoring
  conveniences such as `str` are allowed in route signatures, but bound route
  stubs must normalize annotations to the greatest common runtime TinyChain type
  (`str` -> `tc.String`, numeric/bool primitives -> `tc.state.Value`,
  mixed unions -> their common ancestor) so deferred plans preserve useful IDE
  and documentation types. Do not reintroduce placeholder wrappers like
  `Ref[str]`.
- Preserve `tc.String` as the value-module `String(Value)` type. String
  templating belongs on `String.render(...)` only; do not add render methods to
  generic `Value` or unrelated scalar types.
- Declare library resource identity explicitly: every concrete `Library` subclass
  sets canonical class-level `publisher`, `resource_name`, and `version`, where
  `resource_name` is the library name path component in
  `/lib/{publisher}/{resource_name}/{version}`. Do not derive library identity from
  the Python class name, and do not reintroduce a raw `name` field or decorator/
  constructor `name` overrides as an identity source. Route names still come from
  method names.
- Choose `resource_name` as a well-formed TinyChain `Id` path component: use
  lowercase ASCII letters and digits separated by single `-` or `_` characters.
  Keep it descriptive but concise, and prefer the domain concept or capability
  being named. Normally omit redundant resource-kind suffixes such as `library`,
  `service`, or `class`; include one only when needed to disambiguate the
  resource. Treat `resource_name` as stable external identity because changing
  it changes every published URI.
- Path-only library/service/class URIs target the active/default local PyO3 host;
  authority-qualified URIs target HTTP(S). Preserve `with tc.backend(...)` as an
  override, not as a requirement for ordinary package calls.
- Treat `tc.state.Scalar` as a generic IR union, not a numeric type. Do not add
  arithmetic dunder semantics to `Scalar`; numeric arithmetic belongs to
  `tc.Number` (and future numeric tensor wrappers) with explicit coercion rules.
- Keep control-flow and container constructors as module helpers
  (`tc.state.cond`, `tc.state.while_loop`, `tc.state.after`, `tc.state.map_of`,
  `tc.state.tuple_of`, `tc.state.id`), not `Scalar` static methods.
- Do not add compile-time tree-walk APIs on `Scalar` (for example `walk` or
  `walk_tcref`). Traversal logic must operate on IR/form payloads at the point of
  use and must not assume values are known during route compilation.
- Keep type-specific operations on typed wrappers (`tc.Number`, `tc.Bool`,
  `tc.Tuple`, `tc.Map`, `tc.String`) and avoid growing generic operation helpers
  on `tc.state.Scalar`.
- Keep symbolic IR wrappers (`tc.state.Scalar`, `tc.state.TCRef`, `tc.state.OpRef`,
  and related control-flow/reference forms) on a single canonical representation
  per instance. Do not model them as dataclass-like containers with parallel
  optional fields for each variant.
- Avoid literal-membership or eager-value checks to choose symbolic behavior
  (for example checking whether a symbolic node "contains" a concrete value kind)
  in deferred planning paths. Route dispatch should follow symbolic form/type
  semantics and operation subjects, not runtime literals.
- Prefer a v1-style form accessor pattern (`form_of(...)`-style helpers) for
  internal traversal and compilation logic. Do not rely on `.value/.ref/.op/.map/.tuple`
  field probing on symbolic wrappers.
- Keep `tc.state.Value` as a minimal base with explicit concrete subclasses
  (`Null`, `Link`, `Bool`, `Number`, `String`, `Map`, `Tuple`). Do not add
  type-specific constructors/accessors on `Value`, and do not expose a `.value`
  property for client code; access underlying representation via `form_of(...)`
  helpers.
- Keep symbolic wrappers focused on their canonical IR shape. Serialization is
  an explicit transport/export operation, never an internal traversal or
  normalization mechanism. Runtime arithmetic/comparison/container behaviors
  belong on typed wrappers (`tc.Number`, `tc.Bool`, `tc.Tuple`, `tc.Map`,
  `tc.String`) and protocols.
- Avoid shared-helper type ladders (`if/elif isinstance(...)`) for runtime
  behavior dispatch. Prefer type-specific implementation on the owning wrapper
  class/module. If a type ladder is unavoidable at a decode/normalization
  boundary, isolate it in one explicit dispatch function and keep it small.
- Keep Python response materialization as one recursive boundary projection:
  recursively project ordinary state/map/tuple structure and preserve collection
  sequences as lazy terminal iterators. Do not duplicate that traversal across
  response wrappers, type-specific collection helpers, or JSON re-encoding
  paths; collection leaves remain owned by their collection modules.
- Preserve concrete method type information for symbolic operation forms.
  Do not erase `Get/Put/Post/Delete` operation refs/defs behind parent-class
  method strings or generic `args` shape checks when constructing, validating,
  or serializing IR. Prefer concrete subclasses and `isinstance` dispatch.
- Autodiff is a call-site transform over canonical route IR, not route
  metadata. Do not add autodiff-specific route decorators such as `diff_get`
  or `diff_post`, and do not put `rule`/`wrt` metadata on `@tc.get`/`@tc.post`.
  Reserve `tc.grad(target, wrt=...)` for the JAX-like autodiff transform API;
  route definitions remain ordinary TinyChain routes.
- High-level symbolic wrappers must construct refs through the canonical typed
  builder methods on `Scalar` (for example `_get`, `_post`, `_put`,
  `_post_ref`). Do not hand-write `TCRef(GetOpRef(...))`,
  `TCRef(PostOpRef(...))`, etc. in wrapper modules such as
  `collection/tensor/core.py`.
- Do not use or reintroduce `TCRef.id(...)`. Construct id refs directly via
  `IdRef(name)`, and keep `tc.state.id(name)` as the user-facing helper that
  returns a typed symbolic scalar ref.
- Keep URI values structured until the serialization/transport boundary; avoid
  extracting `.path` in symbolic wrappers.
- Treat `.path` as a boundary-only escape hatch. In runtime/domain code, pass
  `URI` values directly and prefer `str(uri_value)` at encode/transport boundaries
  (JSON keys, HTTP/kernel request paths) instead of `URI(...).path` extraction.
  Do not introduce new `URI(...).path` constructions outside explicit boundary
  adapters.
- For runtime URI composition, use `uri(TypeOrInstance, ...)` and `URI(...)`
  directly. `uri(...)` is the generic type/instance accessor and path builder;
  `URI(...)` is the explicit constructor. Do not define local URI-constructor
  helpers (for example `*_uri(...)`) or module URI constant tables (`*_URI`)
  in runtime modules.
- In runtime/client modules, construct canonical TinyChain resource paths only
  through URI helpers (`tc.uri`, `tinychain.uri.path`, etc.). Do not hardcode
  literal `/state/...`, `/service/...`, `/lib/...`, `/class/...`, `/host/...`,
  or `/healthz...` strings outside tests/doc prose.
- For native `/state/...` resources, define exactly one root URI on the owning
  class/module (for example `State.__uri__`) and derive all descendants through
  typed subjects (`uri(Type, ...)`, `path(Type, ...)`, `uri(root_uri, ...)`).
  Do not scatter repeated `path("state", ...)`/`uri("state", ...)` constants
  across runtime modules.
- Ban `try: import ...` / `try: from ... import ...` in client/runtime code.
  The only allowed exception is explicit conversion glue for optional large
  external tensor ecosystems (`tensorflow`, `torch`, `jax`) where import
  availability directly gates that conversion path.
- For symbolic refs/opdefs, do not add `_cmp_key` helper APIs. Any structural
  inspection must use the canonical in-memory form directly; never call
  `to_json`, `json.dumps`, or `json.loads` to implement equality, hashing,
  cloning, reference construction, validation, or type inspection.
- PyO3 local execution must pass Rust-backed State and collection handles
  directly. JSON/HTTP body conversion is permitted only for a real remote HTTP
  call or explicit Python materialization, never as a local execution bridge.
- Resolve active state context through the shared context helper
  (`state.context.resolve_context`) rather than ad hoc `current_context()`
  call sites.
- Do not use `coerce` naming in symbolic/state runtime helpers. Where a helper
  validates or wraps inputs, name it explicitly for its role (for example
  `normalize_*`, `*_operand`, `autobox`) so no implicit-conversion semantics
  are implied.
- Naming convention for path metadata:
  use `*_URI` for `URI` objects and `*_PATH` for serialized path strings.
  Avoid `*_tag` naming for path-like values, and avoid `*_path()` helper
  functions when a canonical module constant can represent the same value.
- Keep one canonical route-stub call shape in application code.
  Require keyword arguments for route parameters and use `body=` only when
  passing one explicit payload. Positional route arguments are prohibited.
- Preserve a single obvious execution path for applications:
  route method calls inside `tc.backend(...)` contexts are the default,
  while `tc.execute`, `tc.Host.execute`, and `tc.Host.request` remain
  advanced/low-level APIs.
- Keep public tensor/client APIs backend-neutral: never expose storage-engine-
  specific names (e.g., `fensor`, `ha-ndarray`, filesystem labels) in exported
  types, method names, or payload keys. Backend wire details may exist internally
  but must stay behind neutral interfaces.

## Gap triage guardrails

- Before adding a "framework gap" item, cite the missing primitive in
  `client/py` and explain why it should be framework-native across packages.
- For "parity gap" items, also cite the matching v1 behavior in
  `~/Documents/tinychain/client/py`.
- If the behavior can already be expressed with `tinychain.Host`,
  `tinychain.Executor`, or route stubs, treat it as package integration work,
  not a framework gap.
- Prefer deleting package-local transport shims once equivalent framework calls
  exist; avoid parallel request stacks.

## Testing and docs

- Activate `.venv` and run `python -m pytest py/tests` (or the specific module
  you touched) after changes. Add focused tests instead of defensive branches.
- PyO3 gateway tests must hard-fail when the local backend is missing; do not
  add skip paths for `tinychain-local` availability in these tests.
- Update `README.md` or `ROADMAP.md` in this directory when altering user-facing APIs
  or shard-routing behaviors. Keep examples concise for fast unboxing.

## Python style guidance

- Prefer idiomatic Python syntax in client-facing examples/tests: use `len(x)` instead of
  `.len()`, comparison operators (`>`, `<`, `==`) instead of method calls, `+` for
  addition/concatenation, and indexing/slicing (`x[i]`, `x[1:]`) instead of `head/tail`
  helpers where supported.
