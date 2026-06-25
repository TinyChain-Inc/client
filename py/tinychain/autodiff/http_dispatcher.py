from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import requests

from .graph import AddOperator, BroadcastReduceOperator, MatmulOperator, TransposeOperator, TensorNodeRecord, TensorOperator
from .protocol import AutodiffError

_COLLECTION_TENSOR = "/state/collection/tensor"
_DEFAULT_ROUTE_ROOT = "/lib/std/autodiff/0.1.0"

_DTYPE_WIRE = {
    "f32": "/state/scalar/value/number/float/32",
    "f64": "/state/scalar/value/number/float/64",
}


@dataclass(frozen=True)
class TensorLiteral:
    """HTTP execution tensor value that knows its TinyChain JSON literal form."""

    dtype: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    @classmethod
    def from_numpy(cls, tensor: np.ndarray) -> TensorLiteral:
        if tensor.dtype == np.float32:
            dtype = "f32"
        elif tensor.dtype == np.float64:
            dtype = "f64"
        else:
            raise TypeError("TensorLiteral supports only float32 and float64 numpy arrays")
        return cls(
            dtype=dtype,
            shape=tuple(int(dim) for dim in tensor.shape),
            values=tuple(float(value) for value in tensor.flatten().tolist()),
        )

    @classmethod
    def from_backend_tensor(cls, tensor: object) -> TensorLiteral:
        if not hasattr(tensor, "dtype_tag") or not hasattr(tensor, "shape"):
            raise TypeError("expected tensor object with dtype_tag() and shape()")
        dtype = str(tensor.dtype_tag())
        is_f32 = dtype == "f32" or ("float" in dtype and "32" in dtype)
        is_f64 = dtype == "f64" or ("float" in dtype and "64" in dtype)
        if is_f32:
            values = tensor.flattened_f32()
        elif is_f64:
            values = tensor.flattened_f64()
        else:
            raise TypeError(f"TensorLiteral supports only floating tensors, got {dtype}")
        return cls(
            dtype=dtype,
            shape=tuple(int(dim) for dim in tensor.shape()),
            values=tuple(float(value) for value in values),
        )

    def to_json_literal(self) -> dict[str, object]:
        dtype_path = _DTYPE_WIRE.get(self.dtype, self.dtype)
        return {_COLLECTION_TENSOR: [[dtype_path, list(self.shape)], list(self.values)]}

    def to_numpy(self) -> np.ndarray:
        dtype = np.float32 if "32" in self.dtype else np.float64
        return np.array(self.values, dtype=dtype).reshape(self.shape)

    def __array__(self, dtype=None) -> np.ndarray:
        array = self.to_numpy()
        return array.astype(dtype) if dtype is not None else array


def _tensor_literal(value: object) -> dict[str, object]:
    to_json_literal = getattr(value, "to_json_literal", None)
    if not callable(to_json_literal):
        raise TypeError(
            f"TcServerDispatcher expected TensorLiteral-compatible value, got {type(value).__name__}"
        )
    return to_json_literal()


def _decode_tensor_response(payload: dict) -> TensorLiteral:
    """Decode a server tensor JSON response into a TensorLiteral."""
    if not isinstance(payload, dict) or _COLLECTION_TENSOR not in payload:
        raise ValueError(f"TcServerDispatcher: unexpected response format: {payload!r}")
    meta, values = payload[_COLLECTION_TENSOR]
    dtype_str = str(meta[0])
    shape = tuple(int(d) for d in meta[1])
    dtype = "f32" if "32" in dtype_str else "f64"
    return TensorLiteral(dtype=dtype, shape=shape, values=tuple(float(v) for v in values))


class HttpOperatorHandler(Protocol):
    route_name: str

    def build_body(self, node: TensorNodeRecord, args: list[object]) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class AddHttpHandler:
    route_name: str = "add"

    def build_body(self, node: TensorNodeRecord, args: list[object]) -> dict[str, object]:
        return {
            "x": _tensor_literal(args[0]),
            "y": _tensor_literal(args[1]),
        }


@dataclass(frozen=True)
class BroadcastReduceHttpHandler:
    route_name: str = "broadcast_reduce"

    def build_body(self, node: TensorNodeRecord, args: list[object]) -> dict[str, object]:
        return {
            "x": _tensor_literal(args[0]),
            "target_shape": list(node.op_params["target_shape"]),
        }


@dataclass(frozen=True)
class MatmulHttpHandler:
    route_name: str = "matmul"

    def build_body(self, node: TensorNodeRecord, args: list[object]) -> dict[str, object]:
        return {
            "x": _tensor_literal(args[0]),
            "y": _tensor_literal(args[1]),
        }


@dataclass(frozen=True)
class TransposeHttpHandler:
    route_name: str = "transpose"

    def build_body(self, node: TensorNodeRecord, args: list[object]) -> dict[str, object]:
        return {
            "x": _tensor_literal(args[0]),
            "perm": list(node.op_params["perm"]),
        }


_DEFAULT_HANDLERS: dict[type[TensorOperator], HttpOperatorHandler] = {
    AddOperator: AddHttpHandler(),
    BroadcastReduceOperator: BroadcastReduceHttpHandler(),
    MatmulOperator: MatmulHttpHandler(),
    TransposeOperator: TransposeHttpHandler(),
}


class TcServerDispatcher:
    """RouteDispatcher that calls installed OpDef-backed tc-server routes."""

    def __init__(
        self,
        host: str,
        *,
        route_root: str = _DEFAULT_ROUTE_ROOT,
        handlers: dict[type[TensorOperator], HttpOperatorHandler] | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._route_root = "/" + route_root.strip("/")
        self._handlers = dict(_DEFAULT_HANDLERS if handlers is None else handlers)

    def __call__(self, node: TensorNodeRecord, args: list[object]) -> np.ndarray:
        try:
            handler = self._handlers[type(node.operator)]
        except KeyError as exc:
            raise AutodiffError(
                "unsupported_operator",
                f"TcServerDispatcher: no handler for operator '{node.operator.route_name}'",
            ) from exc
        return self._post(handler.route_name, handler.build_body(node, args))

    def _post(self, route: str, body: dict[str, object]) -> np.ndarray:
        url = f"{self._host}{self._route_root}/{route}"
        response = requests.post(
            url,
            data=json.dumps(body, separators=(",", ":")),
            headers={"content-type": "application/json", "accept": "application/json"},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"TcServerDispatcher: server error {response.status_code}: {response.text}"
            )
        return _decode_tensor_response(response.json())
