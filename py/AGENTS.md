# Python Client Agent Notes

The Python client owns symbolic authoring, URI construction, boundary
projection, and user-facing execution ergonomics. It does not own host routing,
transaction lifecycle, storage, replication, or application compatibility.

- Keep one canonical in-memory form per symbolic `State`, `Scalar`, `TCRef`, and
  operation wrapper. Use `form_of(...)` for structural inspection; never use a
  JSON round trip for equality, cloning, validation, or dispatch.
- Build references through shared typed builders and preserve concrete
  GET/PUT/POST/DELETE forms. Do not add adapter-local method enums or handwritten
  URI concatenation.
- Application identity is explicit publisher, resource path, and version.
  Route names come from decorated members. Emit one literal definition and do
  not add schema, package, artifact, or payload envelopes.
- Bound route calls are the ordinary API. Execution mode is contextual through
  `tc.backend(...)`; do not add per-route eager/deferred flags or parallel
  request stacks.
- A runtime `Ref` or `OpRef` returned while authoring a route is lowered to the
  canonical IR reference inside that route's `OpDef`. Return annotations affect
  call-site typing only. Never persist request descriptors or transport headers
  in a Library definition. A returned Python mapping is one map value; only
  explicit lexical Context bindings introduce operation providers.
- PyO3 local calls pass native State handles without serialization. HTTP encodes
  once. Response materialization is one recursive projection and leaves
  collection streams lazy.
- The client never exposes transaction IDs or selects commit/rollback. It
  preserves bearer authority and delegates the request to the kernel or host.
- Keep typed behavior on the owning wrapper (`Number`, `String`, `Tuple`, `Map`,
  Tensor, and collection variants). Avoid central `isinstance` ladders except at
  one explicit decode/normalization boundary.
- `State` wrappers describe typed values and construct symbolic operations. They
  do not own lexical bindings, compiler validation, runtime request lowering, or
  ambient name lookup. Top-level `Context` owns authoring bindings; the Library
  compiler alone lowers runtime requests into canonical IR.
- Autograph and autodiff compile to shared IR. Unsupported syntax or operations
  fail explicitly; experimental work does not alter route metadata or host
  protocols.
- Persistent collection hosting is not a client concern. Service and Chain will
  own naming and durable history; do not add client-side registries or sharding
  policy as a substitute.
- Preserve request deadlines and downstream backpressure. Retries, prefetch,
  buffering, and concurrency are finite and explicit.

Run `python -m pytest py/tests` for Python changes. Changes to the local backend
also require building/importing `tinychain_local` from `client/rust`; tests must
not skip by falling back to another transport.

Keep [README.md](README.md) user-focused,
[DESIGN_AUTOGRAPH.md](DESIGN_AUTOGRAPH.md) limited to implemented lowering, and
[CLASS_PARITY.md](CLASS_PARITY.md) limited to the stable Class authoring
contract. Future work belongs in a roadmap.
