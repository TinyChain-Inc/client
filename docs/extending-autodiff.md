# Extending Autodiff with Custom Graph-Transform Protocols

Other graph-transform systems can define Python-owned protocols following the
same pattern as the autodiff module. The extension pattern has five steps:

1. **Frozen request/result dataclasses** — Define immutable dataclasses for
   request and result types, following `AutodiffRequest` / `DerivativeProgram`
   in `autodiff/protocol.py`. Use `@dataclass(frozen=True)` and include
   `to_dict()` / `from_dict()` for JSON-serializable round-tripping.

2. **Domain operator subclasses/types** — Define operator types that represent
   the atomic operations in your domain, following `TensorOperator` subclasses
   (`AddOperator`, `MatmulOperator`, `TransposeOperator`) in
   `autodiff/graph.py`. Each operator carries a `route_name` and optional
   metadata.

3. **Rule registration via `@registry.rule()` or `registry.register()`** —
   Define transform rules that map each operator type to its derivative (or
   equivalent transform) nodes. Register them using the decorator
   (`@registry.rule(OperatorType)`) or imperative (`registry.register(rule)`)
   API, following `VjpRegistry` in `autodiff/vjp.py`.

4. **Builder/context manager to record operations** — Use `TensorGraphBuilder`
   or an equivalent context manager to record operations during the forward
   pass, following `autodiff/graph.py`. The builder produces a graph
   (`TensorGraph`) that the transform consumes.

5. **`generate()`-style transform entrypoint** — Expose a top-level function
   that takes the recorded graph and produces the transformed program,
   following `generate()` in `autodiff/__init__.py`.

## Constraint

New protocols **must not** add autodiff-specific route decorators (e.g.
`diff_get`, `diff_post`) or `rule`/`wrt` route metadata on `@tc.get`/`@tc.post`
definitions. All protocol types stay in the Python client — no `tc-ir` Rust
types, no `tc-server` or `tc-state` changes. Route definitions remain ordinary
TinyChain routes.

## Minimal Toy Example

The following example demonstrates the five-step pattern with a custom
"negation" operator and its transform rule, without modifying `autodiff/`
internals:

```python
from __future__ import annotations

from dataclasses import dataclass
from tinychain.autodiff import (
    TensorGraph,
    TensorGraphBuilder,
    TensorOperator,
    VjpRegistry,
    generate,
)


# Step 1: frozen request/result dataclasses

@dataclass(frozen=True)
class NegationRequest:
    graph: TensorGraph
    output_value_id: str
    wrt: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "graph": self.graph.to_dict(),
            "output_value_id": self.output_value_id,
            "wrt": list(self.wrt),
        }


# Step 2: domain operator subclass

class NegationOperator(TensorOperator):
    """Negates a tensor element-wise."""

    def __init__(self, route_name: str = "negate"):
        super().__init__(route_name=route_name)


# Step 3: rule registration

registry = VjpRegistry()

@registry.rule(NegationOperator)
class NegationVjpRule:
    """Derivative of negate is negate: d(negate(x))/dx = -1, so dL/dx = -dL/dy."""

    def __call__(self, node, graph, output_value_id):
        # Produce a derivative node that negates the upstream gradient
        derivative_node = graph.add_node(
            operator=NegationOperator(),
            inputs={output_value_id: node.input_value_ids[0]},
        )
        return {node.input_value_ids[0]: derivative_node.output_value_id}


# Step 4: builder/context manager — reuse TensorGraphBuilder as-is

# Step 5: generate()-style entrypoint — reuse tinychain.autodiff.generate()
# with the custom registry by passing it to ReverseTraversal if needed,
# or write a thin wrapper:

def generate_negation(
    graph: TensorGraph,
    output_value_id: str,
    wrt: list[str],
    seed: str,
) -> object:
    """Transform entrypoint for the negation protocol."""
    return generate(
        graph=graph,
        output_value_id=output_value_id,
        wrt=wrt,
        seed=seed,
    )
```

This example defines a custom operator and rule, reuses the existing
`TensorGraphBuilder` and `generate()` infrastructure, and does not modify any
`autodiff/` internals or add route decorators.
