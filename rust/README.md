## Native Python core and optional local host

This crate builds `tinychain_local` (distributed as `tinychain-local`), required
by the Python client. The default feature set contains the Rust value and IR
codecs and a format-neutral Python-object `destream` adapter. It does not include
the server, HTTP stack, or WASM runtime. Python authoring wrappers lower to native
scalar handles; wire interpretation is owned by `tc-value` and `tc-ir`.

Build with `--features local-host` to include `KernelHandle`, `KernelRequest`,
and native State execution. Local composition and application definitions pass
native values directly. Collection leaves decode with the active request's
allocation context and remaining depth allowance. The synchronous PyO3 boundary
uses one shared wait bridge, releases the GIL, and rejects active Tokio re-entry.

`Codec(max_bytes=...)` owns a current-thread runtime and a per-operation finite
materialization budget (64 MiB by default). Its decoder and result projection
share that budget. Scalar and Value builders enforce the owning Rust depth
limits before publishing handles. There is no Python wire-grammar fallback.

Today, `tinychain-local` depends on the TinyChain runtime crate via a workspace-relative
path, so building the extension assumes this repo is vendored into a runtime checkout.

See [`ROADMAP.md`](ROADMAP.md) for planned
Rust client milestones.

If you are developing without that parent checkout, you will need to either:

- check out the TinyChain runtime repository so the relative dependency paths in
  `Cargo.toml` resolve, or
- temporarily adjust the dependency paths in `Cargo.toml` to match your local
  layout.

## Embedded lifecycle

Use `KernelHandle` as a context manager, or call its idempotent `shutdown()` before
releasing the runtime or reopening storage. Consume or drop unfinished result
handles first: shutdown drains requests and joins active finalization. Dropping a
response does not reverse execution's commit. Cloned handles share shutdown state.

HTTP and Python share the bound request and canonical application definition codec.
Adapters own conversion; the bound request owns execution, deadlines, and completion.
Native view acquisition uses the same `TxnHandle::view` path as HTTP, including
fixed-expiry checks around awaited storage acquisition. Both adapters use
`Deadline::check` around synchronous conversion and ready output; pending Python
results derive their deadline from the retained bound request.
Owning PUT/POST/DELETE requests commit successful execution and roll back execution
errors, timeout, and cancellation through their claimed coordinator. The host retains
cleanup under a fresh bounded deadline; failed rollback terminates the host. Python
conversion and projection failures select no decision and cannot reverse a commit.
Python native handles pass State directly without serialization. Every application
mutation of a persistent mutable collection must pass through its owning Chain.
Returned collection views are read-only, including full reads and identity
transforms; the adapter grants no separate mutation authority. All native results,
including scalars and completed projections, retain their transaction capability.
Transfers retain source capabilities through derived results. Fixed expiry rejects
reuse; it does not revoke ownership. Retained handles can indefinitely delay
oldest-first materialization and workspace cleanup until their last clone drops.
Shutdown interrupts that pre-finalization wait but joins active materialization. Serialized inputs
obey request byte limits, and asynchronous decoding and projection retain the
original deadline. Synchronous Python conversion is checked around its execution;
Tokio cannot preempt it.

Synchronous adapter calls release the GIL and drive the existing Tokio runtime
directly, without a separate adapter wait task or nested runtime. Calls from an
active Tokio runtime are rejected. The kernel retains its own execution task to
finish cancellation rollback; this is shared host policy, not an adapter waiting
mechanism. Worker threads use ordinary stack sizes.

Resources return native State, and `execute_bound` completes execution and its
transaction decision before returning. Ordinary results retain a request owner
for Python projection's deadline, capacity, and shutdown tracking. Projection
failure leaves an already committed transaction committed. HTTP encodes the same
native values through the State codec; neither adapter selects serialization based
on resource provenance.
Shared policy errors become structured Python exceptions. `StateHandle.value()`
materializes Python wire objects through the format-neutral encoder under a
finite byte budget, without an intermediate JSON buffer. Native State handoff
remains unencoded.

The Python encoder produces awaitable materialization results: completed scalars
remain inline, while asynchronous containers retain boxed futures. Its private
builders share the existing byte budget, including retained child-result slots.


### Application panic policy

Ordinary I/O and execution errors reject an operation; their context determines
whether recovery is required. Poisoned cache bookkeeping is an invariant failure,
not permission to continue using the cache. Supported server and Python application
builds use `panic = "abort"` in development and release. A Rust panic, including one
in a detached task, aborts the process without graceful shutdown, rollback, or
running destructors. Chain recovery handles restart. Structured Python exceptions
and WASM guest errors remain ordinary errors.

Rust embedders must set this policy at their application/workspace root: dependency
profiles do not control the final application. The server executable and Python
module reject unwinding builds before creating host resources. Cargo's ordinary
test harness uses unwinding, so unit tests cannot certify the application policy;
see [Cargo's panic profile documentation](https://doc.rust-lang.org/cargo/reference/profiles.html#panic).
