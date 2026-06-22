from __future__ import annotations

import json
from typing import Any

import numpy as np
import requests

from .protocol import AutodiffError

_COLLECTION_TENSOR = "/state/collection/tensor"
_OP_POST_ROUTE = "/state/scalar/op/post"


def _encode_tensor(tensor: Any) -> dict:
    """Encode a tensor value as a TinyChain JSON literal for OpDef POST bodies.

    Accepts:
    - numpy.ndarray (float32 or float64)
    - Objects with dtype_tag() -> str, shape() -> list[int], and
      flattened_f32() / flattened_f64() -> iterable[float] methods
      (e.g. tc_state::Tensor PyO3-bound objects returned by the server).
    """
    if isinstance(tensor, np.ndarray):
        dtype_str = "f32" if tensor.dtype == np.float32 else "f64"
        return {_COLLECTION_TENSOR: [[dtype_str, list(tensor.shape)], tensor.flatten().tolist()]}
    if hasattr(tensor, "dtype_tag") and hasattr(tensor, "shape"):
        dtype_str = tensor.dtype_tag()
        shape = list(tensor.shape())
        values = list(tensor.flattened_f32() if "32" in str(dtype_str) else tensor.flattened_f64())
        return {_COLLECTION_TENSOR: [[dtype_str, shape], values]}
    raise TypeError(
        f"TcServerDispatcher: cannot encode tensor of type {type(tensor).__name__}; "
        "expected numpy.ndarray or an object with dtype_tag()/shape()/flattened_f32()/flattened_f64()"
    )


def _decode_tensor_response(payload: dict) -> np.ndarray:
    """Decode a server tensor JSON response into a numpy array.

    Handles both shorthand dtypes ("f32", "f64") and full TinyChain dtype paths.
    """
    if not isinstance(payload, dict) or _COLLECTION_TENSOR not in payload:
        raise ValueError(f"TcServerDispatcher: unexpected response format: {payload!r}")
    meta, values = payload[_COLLECTION_TENSOR]
    dtype_str = str(meta[0])
    shape = meta[1]
    dtype = np.float32 if "32" in dtype_str else np.float64
    return np.array(values, dtype=dtype).reshape(shape)


class TcServerDispatcher:
    """RouteDispatcher that executes individual ops by POSTing OpDef requests to tc-server.

    Supported op_kind values: "broadcast_reduce", "add".
    Raises AutodiffError("unsupported_operator", ...) for anything else.
    """

    def __init__(self, host: str) -> None:
        self._host = host.rstrip("/")

    def __call__(self, op_kind: str, op_params: dict, args: list) -> np.ndarray:
        if op_kind == "broadcast_reduce":
            return self._broadcast_reduce(op_params, args)
        if op_kind == "add":
            return self._add(op_params, args)
        raise AutodiffError(
            "unsupported_operator",
            f"TcServerDispatcher: no handler for op_kind '{op_kind}'",
        )

    def _broadcast_reduce(self, op_params: dict, args: list) -> np.ndarray:
        body = [
            ["x", _encode_tensor(args[0])],
            ["result", {"$x/broadcast_reduce": {"target_shape": op_params["target_shape"]}}],
        ]
        return self._post(body)

    def _add(self, op_params: dict, args: list) -> np.ndarray:
        body = [
            ["x", _encode_tensor(args[0])],
            ["y", _encode_tensor(args[1])],
            ["result", {"$x/add": {"r": {"$y": []}}}],
        ]
        return self._post(body)

    def _post(self, body: list) -> np.ndarray:
        url = f"{self._host}{_OP_POST_ROUTE}"
        response = requests.post(
            url,
            data=json.dumps({_OP_POST_ROUTE: body}, separators=(",", ":")),
            headers={"content-type": "application/json", "accept": "application/json"},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"TcServerDispatcher: server error {response.status_code}: {response.text}"
            )
        return _decode_tensor_response(response.json())
