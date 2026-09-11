# Extending Python Graph Transforms

TinyChain autodiff is a Python-owned transform over canonical route IR. Other
graph transforms should reuse the same boundary rather than extending server
protocols or route definitions.

## Extension shape

A transform normally consists of:

1. immutable request and result values with explicit canonical projection;
2. domain operator types describing atomic graph operations;
3. a registry mapping operator types to transform rules;
4. a builder which records an ordinary graph; and
5. one pure entry point which returns the transformed program.

The implementations under `py/tinychain/autodiff` provide the concrete
examples: `TensorGraph`, `TensorGraphBuilder`, `TensorOperator`, `VjpRegistry`,
and `generate`. Reuse those contracts when they fit; a new transform may define
parallel Python types only when its domain semantics genuinely differ.

## Boundary rules

- Transforms consume and produce canonical Python IR forms. They do not add Rust
  IR variants merely to represent Python compiler internals.
- Route decorators remain ordinary `tc.get`, `tc.put`, `tc.post`, or
  `tc.delete`. Do not add transform-specific decorators or `rule`/`wrt` route
  metadata.
- Registries are explicit instances owned by the transform. Do not introduce a
  mutable process-global rule registry.
- Graph traversal, dependency ordering, and validation each have one owner.
  Extensions delegate to it rather than serializing and rescanning graphs.
- Unsupported operators fail with a structured transform error. They never
  silently fall back to eager Python execution or a remote service.
- Derivative artifacts are Python-owned immutable metadata expressed as
  ordinary Libraries. They do not change the host installation format or imply
  a server artifact registry.

Add focused tests for immutable round trips, deterministic traversal, rule
selection, unsupported operators, and absence of server/route metadata changes.
Experimental extensions remain clearly labeled until their public contract is
stable.
