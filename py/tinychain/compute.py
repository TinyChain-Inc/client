from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional

from .opref import OpRef, post as opref_post
from .state.base import State
from .state.value import Number
from .state.value import Value
from .uri import path, uri


DType = Literal["i64", "u64", "f32", "f64"]

TENSOR_CLASS_URI = uri(State, "collection", "tensor")
NUMBER_CLASS_URI = uri(Number)

NUMERIC_OPS_CLASS_ROOT_URI = uri("class", "tinychain", "numeric", "0.1.0")


@dataclass(frozen=True, slots=True)
class Encoding:
    kind: Literal["plain", "fixed_point", "encrypted"]
    signed: Optional[bool] = None
    bits: Optional[int] = None
    scale_pow2: Optional[int] = None
    scheme: Optional[str] = None
    params: Optional[Mapping[str, str]] = None

    @staticmethod
    def plain() -> "Encoding":
        return Encoding(kind="plain")

    @staticmethod
    def fixed_point(*, signed: bool, bits: int, scale_pow2: int) -> "Encoding":
        return Encoding(kind="fixed_point", signed=signed, bits=bits, scale_pow2=scale_pow2)

    @staticmethod
    def encrypted(*, scheme: str, params: Optional[Mapping[str, str]] = None) -> "Encoding":
        return Encoding(kind="encrypted", scheme=scheme, params=params)

    def to_json(self) -> object:
        if self.kind == "plain":
            return "plain"

        if self.kind == "fixed_point":
            if self.signed is None or self.bits is None or self.scale_pow2 is None:
                raise ValueError("fixed_point encoding requires signed, bits, and scale_pow2")
            return {
                "fixed_point": {
                    "signed": self.signed,
                    "bits": self.bits,
                    "scale_pow2": self.scale_pow2,
                }
            }

        if self.kind == "encrypted":
            if not self.scheme:
                raise ValueError("encrypted encoding requires scheme")
            payload: dict[str, object] = {"scheme": self.scheme}
            if self.params:
                payload["params"] = dict(self.params)
            return {"encrypted": payload}

        raise AssertionError(f"unexpected Encoding.kind {self.kind}")


@dataclass(frozen=True, slots=True)
class ScalarType:
    dtype: DType
    encoding: Encoding = field(default_factory=Encoding.plain)

    def to_json(self) -> dict:
        return {
            "class": path(NUMBER_CLASS_URI),
            "params": {"dtype": self.dtype, "encoding": self.encoding.to_json()},
        }


@dataclass(frozen=True, slots=True)
class TensorType:
    dtype: DType
    shape: tuple[int | str, ...]
    encoding: Encoding = field(default_factory=Encoding.plain)

    def to_json(self) -> dict:
        return {
            "class": path(TENSOR_CLASS_URI),
            "params": {
                "dtype": self.dtype,
                "shape": list(self.shape),
                "encoding": self.encoding.to_json(),
            },
        }


ValueType = ScalarType | TensorType


@dataclass(frozen=True, slots=True)
class AbsMax:
    abs_max: float

    def to_json(self) -> dict:
        return {"abs_max": self.abs_max}


@dataclass(frozen=True, slots=True)
class Target:
    decode_margin: int = 0
    require_quantize_each_repeat: bool = True
    opaque_policy: Literal["reject", "ignore", "metric_only"] = "reject"

    def to_json(self) -> dict:
        return {
            "decode_margin": self.decode_margin,
            "require_quantize_each_repeat": self.require_quantize_each_repeat,
            "opaque_policy": self.opaque_policy,
        }


@dataclass(frozen=True, slots=True)
class _Input:
    name: str
    vtype: ValueType
    value: int

    def to_json(self) -> dict:
        return {"name": self.name, "type": self.vtype.to_json(), "value": self.value}


@dataclass(frozen=True, slots=True)
class _Output:
    name: str
    value: int

    def to_json(self) -> dict:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class _Node:
    id: int
    op: dict
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    output_types: tuple[ValueType, ...]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "output_types": [t.to_json() for t in self.output_types],
        }


@dataclass(slots=True)
class OpGraph:
    """
    Minimal Op-graph IR builder for certification and execution requests.

    This module intentionally keeps the surface small: it is a builder + JSON encoder for a
    deterministic, analyzable op-graph payload. Operators are referenced by canonical URI
    (typically `/class/...`), so domains can define their own operator sets and certification
    contracts.

    The helpers in this file default to the standard numeric operator URIs under
    `NUMERIC_OPS_CLASS_ROOT_URI`, but callers can override `operator=...` per node.
    """

    version: str = "0.1.0"
    _inputs: list[_Input] = field(default_factory=list)
    _nodes: list[_Node] = field(default_factory=list)
    _outputs: list[_Output] = field(default_factory=list)
    _values: dict[str, int] = field(default_factory=dict)
    _next_value: int = 0
    _next_node: int = 0

    TYPE_TAG_URI = uri(Value, "op_graph")

    def _alloc_value(self, name: str) -> int:
        if name in self._values:
            raise ValueError(f"duplicate value name: {name}")
        value_id = self._next_value
        self._next_value += 1
        self._values[name] = value_id
        return value_id

    def _value(self, name: str) -> int:
        try:
            return self._values[name]
        except KeyError as exc:
            raise KeyError(f"unknown value name: {name}") from exc

    def input(self, name: str, vtype: ValueType) -> "OpGraph":
        value_id = self._alloc_value(name)
        self._inputs.append(_Input(name=name, vtype=vtype, value=value_id))
        return self

    def output(self, name: str) -> "OpGraph":
        self._outputs.append(_Output(name=name, value=self._value(name)))
        return self

    def matmul(
        self,
        out: str,
        a: str,
        b: str,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        out_type: Optional[TensorType] = None,
        operator: str = f"{path(NUMERIC_OPS_CLASS_ROOT_URI)}/matmul",
    ) -> "OpGraph":
        out_id = self._alloc_value(out)
        op = {operator: {"transpose_a": transpose_a, "transpose_b": transpose_b}}
        node = _Node(
            id=self._next_node,
            op=op,
            inputs=(self._value(a), self._value(b)),
            outputs=(out_id,),
            output_types=(out_type or TensorType(dtype="f32", shape=(), encoding=Encoding.plain()),),
        )
        self._next_node += 1
        self._nodes.append(node)
        return self

    def quantize(
        self,
        out: str,
        x: str,
        *,
        signed: bool,
        bits: int,
        scale_pow2: int,
        out_type: Optional[ValueType] = None,
        operator: str = f"{path(NUMERIC_OPS_CLASS_ROOT_URI)}/quantize",
    ) -> "OpGraph":
        out_id = self._alloc_value(out)
        op = {operator: {"signed": signed, "bits": bits, "scale_pow2": scale_pow2}}
        node = _Node(
            id=self._next_node,
            op=op,
            inputs=(self._value(x),),
            outputs=(out_id,),
            output_types=(out_type or ScalarType(dtype="i64", encoding=Encoding.fixed_point(signed=signed, bits=bits, scale_pow2=scale_pow2)),),
        )
        self._next_node += 1
        self._nodes.append(node)
        return self

    def to_json(self) -> dict:
        payload = {
            "version": self.version,
            "inputs": [i.to_json() for i in self._inputs],
            "nodes": [n.to_json() for n in self._nodes],
            "outputs": [o.to_json() for o in self._outputs],
        }
        return {path(self.TYPE_TAG_URI): payload}


def analyze_opref(
    graph: OpGraph,
    *,
    envelope: Mapping[str, object],
    dims: Optional[Mapping[str, int]] = None,
    target: Optional[Target] = None,
    compute_version: str = "0.1.0",
    publisher: str = "tinychain",
) -> OpRef[Any]:
    body: dict[str, object] = {"graph": graph.to_json(), "envelope": dict(envelope)}
    if dims:
        body["dims"] = dict(dims)
    if target:
        body["target"] = target.to_json()

    route_path = path("lib", publisher, "compute", compute_version, "analyze")

    return opref_post(route_path, body=body)


certify_opref = analyze_opref


def run_opref(
    graph: OpGraph,
    *,
    inputs: Mapping[str, object],
    compute_version: str = "0.1.0",
    publisher: str = "tinychain",
    headers: Optional[Iterable[tuple[str, str]]] = None,
) -> OpRef[Any]:
    route_path = path("lib", publisher, "compute", compute_version, "run")

    op = opref_post(route_path, body={"graph": graph.to_json(), "inputs": dict(inputs)})
    return op.with_headers(headers)
