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
4. Compile the rewritten function using the existing `tc.define` opdef compiler.

## Transform rules (minimal)

### Context construction

- Autograph mode introduces a synthetic context, `cxt`, not visible to the user code.
- Each assignment `name = expr` becomes a binding:
  - `cxt.name = tc.state.autobox(expr)`
- Each subsequent read of `name` becomes `cxt.name`.

### Return

- `return expr` becomes `return tc.state.context().result(expr)` after binding rewrite,
  so the compiler captures the bound form and a final `result`.

### Supported statements

- Assignments: `name = expr` (single target, non-annotated).
- Augmented assignments: `name += expr` (desugared to `name = name + expr`).
- `return expr`
- `if` statements with expression-only bodies (no `break/continue`).
- `for`/`while` are not supported in autograph mode (use `While` opdefs instead).

### Supported expressions

- Literals (`None`, bool, number, string).
- Attribute access on TinyChain refs and context-bound names.
- Function/method calls to TinyChain stubs or operators.
- Binary operators supported by Scalar/Value types.

## Rejection criteria (fail fast)

Reject the transform if the AST contains any of:

- `global`, `nonlocal`, `del`, `with`, `yield`, `await`, `async`.
- Comprehensions, generator expressions, or lambdas.
- Multiple assignment targets, tuple unpacking, annotated assignment.
- `try/except/finally`, `raise`, or dynamic `exec/eval`.
- Name binding that collides with reserved identifiers (`self`, `cxt`, `ctx`, `txn`).

## Error taxonomy

All errors should be deterministic and descriptive:

- `AutographSyntaxError`: unsupported syntax node with location.
- `AutographNameError`: invalid or reserved variable name.
- `AutographControlFlowError`: unsupported control flow (e.g. `for`, `while`, `try`).
- `AutographAssignmentError`: invalid assignment target or destructuring.
- `AutographMixedContextError`: explicit `cxt/ctx/txn` plus autograph mode.

## Integration points

- `client/py/tinychain/define.py::_compile_opdef_route`:
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
