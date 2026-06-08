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
- Derive library/service/class names from Python class names and route names from
  method names. Do not reintroduce decorator or subclass `name` overrides; rename
  the class/method when a different URI is needed.
- Path-only library/service/class URIs target the active/default local PyO3 host;
  authority-qualified URIs target HTTP(S). Preserve `with tc.backend(...)` as an
  override, not as a requirement for ordinary package calls.

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
