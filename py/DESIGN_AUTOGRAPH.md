# Autograph-style OpDef compilation (Python client)

## Goal

Enable a decorator-driven, autograph-style authoring experience for `@tc.get/@tc.post/@tc.put/@tc.delete`
methods such that local assignments (e.g. `n = 2`) are compiled into op-graph bindings without explicit
`cxt` plumbing. The resulting IR must remain v2-native (`Scalar`/`TCRef`/`OpRef`) so the host executes a
single, transport-agnostic code path.

## Non-goals

- No runtime execution of user code on the host (the transform runs in the Python client).
- No implicit authorization or transport-specific shortcuts.
- No expansion of the supported language surface beyond a minimal, deterministic subset.
- No bespoke IR; emit the same `Scalar`/`OpDef` shapes as the existing compiler path.

## High-level flow

1. Decorator receives a Python method.
2. If the method signature includes an explicit `cxt/ctx/txn` parameter (first arg after `self`),
   keep the current v1-style compile path (no AST transform).
3. Otherwise, run the autograph transform:
   - Parse the function body to an AST.
   - Rewrite local assignments/reads into context bindings.
   - Emit a new function which, when executed, populates a fresh `Context` and returns the result.
4. Compile the rewritten function using the unified `tc.Library` opdef compiler.

## Design notes (lessons from v1 reflection)

TinyChain v1 compiled handler graphs by executing decorated callables under a fresh
transaction context, using `inspect.signature` reflection and name assignment rules
implemented by the v1 `Context`. Even though v2 Autograph uses AST rewriting rather than
`inspect`-driven compilation, several v1 behaviors are intentional precedents for v2:

- **Fail-fast namespace collisions.** v1 validates that a handler's parameter namespace does
  not collide with ids bound in the context form, and errors deterministically when it does.
  v2 Autograph inherits this policy: any collision (parameters, reserved ids, or previously
  bound locals) is an error.

- **Constant folding of literal-bool conditionals.** v1 `If` short-circuits during encoding
  when its condition is a literal Python `bool`, returning only the selected branch rather
  than emitting a conditional reference. v2 Autograph adopts the same rule during AST
  rewriting: only fold branches when the condition is a transform-time boolean constant.

- **`elif` as a nested conditional branch.** v1 forbids nested conditionals as the
  *condition* of an `If`. This naturally drives the `elif` lowering strategy used in v2:
  nest conditionals in the `or_else` branch (`tc.cond(c0, t0, tc.cond(c1, t1, e))`).

- **Loops are explicit closures, not inferred side effects.** v1 does not attempt to infer
  loops from Python syntax; it expresses looping with an explicit `While(cond, step, state)`
  whose `cond` and `step` are callable op closures. v2 Autograph follows the same model:
  lower Python `while` into `tc.state.while_loop(cond_opdef, step_opdef, init_state)` by
  synthesizing deterministic condition/body opdefs (rather than trying to represent arbitrary
  Python loop bodies).

- **Error style.** v1 primarily uses standard Python exceptions with short, stable messages.
  v2 Autograph errors should follow the same pattern (optionally including source location),
  and must not depend on global runtime state.

## Transform rules (minimal)

### Context construction

- Autograph mode introduces a synthetic context, `cxt`, not visible to the user code.
- Each assignment `name = expr` becomes a binding:
  - `cxt.name = tc.state.autobox(expr)`
- Each subsequent read of `name` becomes `cxt.name`.
- All namespaces are local: there is no supported mechanism for reading or mutating global
  state. Any attempt to use `global` or `nonlocal` is rejected.

### Compile-time constant folding (v0)

When an `if`/`while` condition can be evaluated at transform-time (i.e., during the AST
rewrite), Autograph must eliminate the unreachable branch and must not emit a `TCRef::Cond`
or `TCRef::While`:

- `if True: ... else: ...` lowers to just the `then` branch.
- `if False: ... else: ...` lowers to just the `else` branch.

When constant folding eliminates an `if`, any `return`/assignment lowering is applied
to the selected branch as if it were written inline.

These cases are expected to be rare because Autograph does not allow global namespace
introspection; most meaningful conditions will be TinyChain refs resolved on the host.

### `if` statement lowering (v0)

Autograph `if` is **expression-oriented**: each `if` must produce values which are then
joined explicitly using TinyChain conditional references.

#### Condition

The `if` condition must compile to a TinyChain reference (a `Scalar` whose `ref` is not
`None`), because `TCRef::Cond` requires a reference-valued condition. Autograph must reject
conditions which are plain Python literals (e.g. `True`) or any expression which compiles
to a non-ref `Scalar`.

Constant-valued conditions are a special case handled by compile-time constant folding
(see above).

#### Mapping to TinyChain IR

An `if` statement is lowered using `tc.cond(condition, then, or_else)` (a `Scalar` whose
`ref` is a `TCRef::Cond`).

Autograph supports the following `if` forms in v0:

1) **Return-if**

```python
if cond:
    return then_expr
else:
    return else_expr
```

Lowers to:

```python
return tc.cond(cond, then_expr, else_expr)
```

2) **Assignment-if**

```python
if cond:
    a = then_a
    b = then_b
else:
    a = else_a
    b = else_b
```

For each assigned name `x`, lower to a single joined binding:

```python
cxt.x = tc.cond(cond, then_x, else_x)
```

The branch bodies do **not** construct independent contexts. Instead, the lowered form
computes each branch RHS as a normal expression and joins them with `tc.cond`.

3) **Assignment-if without else**

```python
value = initial
if cond:
  value = next_value
```

Lowers to identity-preserving else semantics:

```python
cxt.value = tc.cond(cond, next_value, tc.state.id("value"))
```

This form is only valid when every assigned name in the `then` branch is already
bound in the surrounding scope.

4) **Return-if with immediate fallback return**

```python
if cond:
  return then_expr
return else_expr
```

Lowers to:

```python
return tc.cond(cond, then_expr, else_expr)
```

#### v0 structural constraints (fail fast)

To keep the transform deterministic and avoid SSA/phi semantics in v0, Autograph must
reject any `if` which violates any of the following:

- The `then` and `else` bodies must be composed only of supported statements:
  - either a single `return <expr>` (Return-if), or
  - a sequence of simple assignments `name = <expr>` (Assignment-if).
- For `if` without `else`, every assigned name in the `then` branch must already
  be bound in the surrounding scope.
- `if cond: return ...` is allowed without `else` only when followed immediately
  by a fallback `return ...`.
- Assignment-if: the `then` and `else` branches must assign the **same set of names**,
  and each name must be assigned **exactly once** per branch.
- No nested control flow inside the branches (no nested `if`, no `for`, no `while`).
- No `return` mixed with assignments within the same branch.
- No name collisions: a name assigned by Assignment-if must not already be bound in the
  surrounding scope (including parameters).

## Ordering semantics and side effects

Autograph lowers Python syntax into TinyChain dataflow. Execution order is derived from
dependency edges, not source order. Independent nodes may execute concurrently.

Autograph does not infer side-effect ordering and must not synthesize sequencing edges.
If route logic depends on side-effect order, the developer must add an explicit `After`
reference in the graph.

The Python client exposes this as `tc.after(dependency, then)` (and `tc.state.after`
at the scalar layer). The helper records an explicit ordering dependency and returns
`then` unchanged so typed wrappers remain chainable.

Use explicit `After` when:

- Two side-effecting operations must happen in a specific order.
- An externally visible effect (notification, write, audit emission) must occur only
  after another effect commits.
- There is no natural data dependency between two operations, but ordering is required.

Example:

```python
@tc.post
def update_then_read(self, key: tc.String, value: tc.Number) -> tc.Number:
    write_op = self.store.put(key, value)
    read_ref = self.store.get(key)
    ordered_read = tc.after(write_op, read_ref)
    return ordered_read
```

In this shape, `ordered_read` keeps the same wrapper type as `read_ref`, but the
graph records that the read must execute after the write.

### `elif` lowering (v0)

`elif` lowers to nested `tc.cond` references by desugaring:

```python
if c0:
    t0
elif c1:
    t1
else:
    e
```

is treated as:

```python
if c0:
    t0
else:
    if c1:
        t1
    else:
        e
```

This yields a `tc.cond(c0, t0, tc.cond(c1, t1, e))` shape in the `or_else` branch.

Numeric expressions in lowered routes target TinyChain numeric value semantics
(`tc.Number` and numeric tensor wrappers), not generic `tc.state.Scalar`
arithmetic.

### `while` statement lowering (v0)

Autograph `while` is lowered to the canonical TinyChain `TCRef::While` reference via
`tc.state.while_loop(cond, closure, state)`.

Because TinyChain `While` requires *callable* condition and body closures (expressed as
`OpDef`s), Autograph constructs them deterministically as internal, synthetic opdefs.

#### Supported `while` shape

Autograph supports the following `while` form in v0:

```python
state = init_state
while cond_expr(state):
    tmp = expr(state)
    state = step_expr(state, tmp)
return state
```

where:

- `state` is a single local name bound exactly once before the loop, and then updated
  exactly once per loop iteration (the final statement in the loop body).
- The loop condition and body are pure expressions over `state` and locals bound within
  the loop body (no side-effecting statements). `if` statements are permitted only in the
  Assignment-if form and lower to `tc.cond`.

#### Lowering

Autograph lowers the loop by synthesizing two opdefs:

- `cond(state) -> Bool`: returns the rewritten `cond_expr(state)`
- `step(state) -> state`: returns the rewritten `step_expr(state)`

and then emits:

```python
state = tc.state.while_loop(cond_opdef, step_opdef, init_state)
```

#### v0 `while` constraints (fail fast)

- `break`/`continue` are rejected.
- The loop must have exactly one loop-carried variable (`state`) which is updated by a
  single assignment in the body.
- The body must contain only assignments and Assignment-if statements, and the `state = ...`
  update must be last.
- `for` loops are not yet implemented (backend support is pending).

### Return

- `return expr` remains a plain `return` after binding rewrite.
- The unified `tc.Library` compiler wraps a non-`ContextResult` return value using the
  current scoped context form (equivalent to `tc.state.context().result(expr)`), so
  the OpDef captures both the bound form and a final `result`.

### Supported statements

- Assignments: `name = expr` (single target, non-annotated).
- Augmented assignments: `name += expr` (desugared to `name = name + expr`).
- `return expr`
- `if` statements with expression-only bodies (no `break/continue`).
- `while` statements in the supported v0 shape (see above).
- `for` loops in the supported v0 shape (see below).

### Supported expressions

- Literals (`None`, bool, number, string).
- Attribute access on TinyChain refs and context-bound names.
- Function/method calls to TinyChain stubs or operators.
- Binary operators supported by Scalar/Value types.
- `lambda` expressions are allowed when passed to `tc.post(...)` to construct a POST
  `OpDef` from a single expression (no statements or local bindings).

## Rejection criteria (fail fast)

Reject the transform if the AST contains any of:

- `global`, `nonlocal`, `del`, `with`, `yield`, `await`, `async`.
- Comprehensions, generator expressions, or lambdas.
- Multiple assignment targets, tuple unpacking, annotated assignment.
- `try/except/finally`, `raise`, or dynamic `exec/eval`.
- Name binding that collides with reserved identifiers (`self`, `cxt`, `ctx`, `txn`).
- Any assignment which collides with an existing name in the same namespace:
  - route parameters,
  - previously bound local names (no re-assignment in v0),
  - compiler-reserved ids like `result`.

## Error taxonomy

All errors should be deterministic and descriptive:

- `AutographSyntaxError`: unsupported syntax node with location.
- `AutographNameError`: invalid or reserved variable name.
- `AutographControlFlowError`: unsupported control flow (e.g. `for`, `while`, `try`).
- `AutographAssignmentError`: invalid assignment target or destructuring.
- `AutographMixedContextError`: explicit `cxt/ctx/txn` plus autograph mode.
- Error formatting should follow the v1 Python client style: short, stable messages with
  any structured details rendered inline (no reliance on global state).

## Integration points

- `client/py/tinychain/library.py::_compile_opdef_route`:
  - Detect explicit `cxt/ctx/txn`.
  - If absent, call `autograph.transform(form)` to produce a compiled callable.
  - Execute the transformed callable inside `scoped_context()` and compile as today.

## Compatibility & migration

- Explicit `cxt/ctx/txn` remains fully supported and unchanged.
- Autograph mode is opt-in via absence of explicit context param.
- IR output is identical in shape to the current compiler (`OpDef` with context form).

## Testing strategy

- Golden IR tests: compare opdef JSON output for a method written with explicit `cxt`
  vs the same method written in autograph style.
- Rejection tests: verify each unsupported syntax emits the correct error type/message.
- Integration test: use the op reflection analysis test as the primary smoke test for
  autograph behavior (no `cxt =`, no explicit `autobox`).
### `for` statement lowering (v0)

Autograph `for` lowers to `TCRef::ForEach`, which applies a POST `OpDef` to each element
of a scalar tuple (or the keys of a scalar map) and returns the last result.

#### Supported `for` shape

```python
for item in items:
    tmp = expr(item)
    out = other_expr(tmp)
```

where:

- `item` is a single target name.
- `items` is a scalar tuple or scalar map (or ref to one).
- The body contains only assignments and Assignment-if statements.
- The loop item is not reassigned within the body.

#### Lowering

Autograph constructs a POST opdef from the loop body, using the loop item name as the
parameter id, and then emits:

```python
tc.state.for_each(items, item_name="item", op=opdef)
```

The resulting `ForEach` ref is bound to an internal temp to preserve ordering.

When `items` is a map, the loop item is the key as a string, iterated in deterministic
`Id` order (the underlying `BTreeMap` key order).

#### v0 `for` constraints (fail fast)

- `for-else` is rejected.
- The loop target must be a single name.
- The body must contain only assignments or Assignment-if statements.
- The loop item must not be reassigned.

## Proposed: Fully-nested control flow (draft)

This section outlines a forward-compatible lowering strategy that allows nested `if`,
`for`, and `while` so user code reads like regular Python while still producing a
deterministic TinyChain graph.

### Goals

- Preserve a single, deterministic IR path (no adapter-specific semantics).
- Keep variable binding rules explicit and predictable.
- Avoid implicit global state or Python-side side effects.
- Keep generated graphs minimal and composable via `TCRef` control primitives.

### Core model

Autograph treats each control-flow block as a pure graph fragment which:

- Takes an explicit input environment (a map of named scalars).
- Emits an explicit output environment (a map of named scalars).
- Is encoded as an `OpDef` and invoked via `TCRef::Cond`, `TCRef::While`, or `TCRef::ForEach`.

This makes nested control flow just nesting of graph fragments.

### Binding and merge rules (SSA-lite)

1. A name is **defined** when assigned in the current scope.
2. A name is **captured** when referenced from an outer scope.
3. Any name assigned in a branch **must** be assigned in all sibling branches.
4. On merge, each assigned name is bound to a `tc.cond` (or equivalent) which selects
   the branch value. This is the only implicit merge operation.
5. A name may be reassigned; the latest dominating assignment is the visible value.
6. Shadowing a captured name is forbidden in v1 of this feature to avoid ambiguity.
   The error message should follow v1 style: `namespace collision: <name>`.

### Nested `if`

- Lower each branch to its own `OpDef`.
- The enclosing scope invokes `TCRef::Cond` with the branch `OpDef`s.
- The result is an environment map; it is merged into the outer scope via rule (4).

### Nested `while`

`while` is modeled as a loop-carried state map:

- The loop state is an explicit map of all variables which are mutated in the body.
- The condition `OpDef` reads the state and returns a Bool.
- The body `OpDef` reads the state and returns the next state.
- Variables not in the loop state are read-only captures.
- The enclosing scope binds the final state to the mutated names.

### Nested `for`

`for` lowers to `TCRef::ForEach` with an explicit loop state:

- The loop body `OpDef` takes `{item_name, state}` and returns the next `state`.
- The loop state is a map of mutated names; captures are read-only.
- `ForEach` iterates over items and reduces state, returning the final state.

This requires the `ForEach` executor to accept a loop state map (not just last result),
or a dedicated `Fold` control ref which returns the final state.

When iterating over a map, the loop item behaves like Python `dict` iteration: it yields
keys. In v2 this means the key as a string `Id`, in deterministic `BTreeMap` order.

### Returns

- A `return` inside a nested block becomes an assignment to a reserved result name,
  followed by short-circuiting the outer function to return that value.
- Early `return` is allowed, but requires explicit short-circuit semantics:
  - Introduce a reserved `__tc_returned` flag and `__tc_return_value`.
  - Each enclosing control block propagates the flag and skips further work when set.
  - The top-level transform returns `__tc_return_value` when the flag is set.

### Break / continue

- `break` and `continue` are not supported in the initial nested-control release.
- They require additional loop state flags (`break`, `continue`) and scheduler support.

### Error handling

- Error classes remain the same; messages should note nested location when available.
- Unsupported constructs must fail fast at transform time, not runtime.

### Migration / rollout

1. Implement nested `if` inside `while`/`for` bodies (no nested loops yet).
2. Add loop-state maps for `while` with mutated-name analysis.
3. Extend `ForEach` or add `Fold` to carry loop state.
4. Enable fully nested `if`/`for`/`while`.

## Proposed: Autograph compiler architecture (draft)

This section documents the intended implementation model for Autograph so we avoid
implicit bindings and hidden `_tmp` ids that can escape the OpDef form.

### Design goals

- All refs must be explicit and appear in the OpDef form (no hidden context binds).
- The compiler must be deterministic and transport-agnostic.
- Nested control flow must be expressed as `TCRef` primitives with explicit env maps.

### Two-phase lowering

**Phase 1: Build a binding IR**

- Parse the function body into a small IR:
  - `Bind(name, expr)`
  - `If(cond, then_block, else_block)`
  - `While(cond, body_block)`
  - `ForEach(items, item_name, body_block)`
  - `Return(expr)`
- During this phase, collect:
  - All bound names (including temporaries)
  - All referenced names (free variables)
  - The exact set of names assigned in each branch
- Reject unsupported constructs here, with v1-style errors.

**Phase 2: Emit OpDef + TCRefs**

- Topologically order `Bind` nodes.
- Emit every bound name into the OpDef form **once**, in order.
- Lower each control node to a `TCRef` that takes an explicit environment map.
- Merge branch outputs with explicit `tc.cond` expressions.
- The final `return` becomes `("result", <expr>)`.

### No implicit context binds

- Autograph should never call `Scalar._subject` in a way that triggers
  `current_context().bind_auto`.
- If a subject needs an id, it must already be bound in Phase 1 and
  referenced explicitly in Phase 2.
- This implies a small helper for “safe refs” which refuses to auto-bind.

### Environment maps

- Every nested block is modeled as a pure function:
  - **Input**: explicit map of ids → scalars
  - **Output**: explicit map of ids → scalars
- The enclosing scope always merges the output map back into locals.

### Early return (short-circuit)

- Model `return` as writing to `__tc_return_value` and setting
  `__tc_returned = true`.
- All enclosing blocks must propagate the flags and skip further work
  once `__tc_returned` is true.

### Benefits

- Prevents hidden `_tmp` ids (all ids are in the OpDef form).
- Makes nested control composable and testable.
- Removes dependency on `current_context` side effects.
