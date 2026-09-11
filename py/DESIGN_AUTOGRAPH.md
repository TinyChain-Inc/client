# Autograph Lowering Contract

Autograph is a client-side source transform for decorated Library methods. It
turns a deliberately small Python subset into the same `Scalar`, `TCRef`,
`OpRef`, and `OpDef` forms used by explicit-context authoring. It is not a new
IR or a host-side Python runtime.

## Selection

A route whose parameters include `cxt`, `ctx`, or `txn` uses the explicit
v1-style compiler path. A route without one of those parameters is transformed
by Autograph. The two paths emit the same canonical route definition.

A runtime `Ref` or request `OpRef` returned by either path is lowered to the
canonical IR reference inside the route's `OpDef`; request metadata is never an
installed Library member. Return annotations type the bound Python call only.
A returned mapping is one map value, while explicit `Context` bindings define
the operation's lexical providers.

Autograph obtains the function source, creates a fresh local `Context`, and
rewrites supported local bindings into that context. It does not inspect or
mutate process-global state.

## Supported statements

The implemented subset includes:

- simple name assignment and augmented assignment;
- `return`;
- expression-oriented `if`/`elif` forms;
- bounded `while` lowering to `tc.state.while_loop`;
- `for` lowering to `tc.state.for_each`; and
- `pass` where it does not introduce an ambiguous value.

Control flow is lowered to ordinary `TCRef` variants. Literal boolean branches
are folded before IR emission. Branch assignments must have deterministic merge
semantics; unsupported nested or mixed forms fail rather than inventing Python
execution order.

Assignments are local to the transformed route. `global`, `nonlocal`, async
control flow, exception handling, arbitrary expression statements, unsupported
mutation targets, and access to undeclared globals are rejected with a typed
Autograph error.

## Names and expressions

Parameters, reserved compiler names, and generated bindings occupy one checked
namespace. Collisions and reads before binding fail deterministically. Calls and
attribute access are accepted only when they resolve through the supported
TinyChain symbolic vocabulary; arbitrary Python modules or runtime side effects
are not captured.

Python local reassignment is lowered to deterministic, unique SSA-style
providers before IR emission. It never rebinds or repoints a `Context` name.
Nested graph callbacks may capture enclosing providers but may not shadow them;
the emitted `OpDef` is validated against the canonical lexical-scope contract.

The transform preserves annotations, defaults, documentation, and source
location where practical. Error messages identify the unsupported construct but
must not depend on runtime-global state.

## Ordering

TinyChain graphs execute by dependency, not Python statement order. Independent
nodes may run concurrently. Autograph does not synthesize side-effect ordering.
Use `tc.after(dependency, value)` when a required order is not already expressed
by data flow.

Loops become explicit graph closures over loop-carried state. They remain
subject to the host deadline and the project temporal-locality rule; Autograph
does not turn a long-running Python loop into detached host work.

## Extension rule

Extend Autograph only when a Python construct has one deterministic lowering to
existing shared IR. Add the lowering and its accept/reject tests together. A
construct requiring SSA variants, a new graph language, host-side Python, or
implicit side-effect policy remains unsupported until the shared IR contract
defines it.

The implementation in `tinychain/_autograph.py` and focused
`py/tests/test_autograph_*.py` tests are authoritative for the precise accepted
syntax. Proposed compiler architectures and unimplemented nested-control-flow
schemes belong in the roadmap, not this contract.
