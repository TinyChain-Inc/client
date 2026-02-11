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

## Testing and docs

- Activate `.venv` and run `python -m pytest py/tests` (or the specific module
  you touched) after changes. Add focused tests instead of defensive branches.
- PyO3 gateway tests must hard-fail when the local backend is missing; do not
  add skip paths for `tinychain-local` availability in these tests.
- Update `README.md` or `ROADMAP.md` in this directory when altering user-facing APIs
  or shard-routing behaviors. Keep examples concise for fast unboxing.
