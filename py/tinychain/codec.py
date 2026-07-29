from __future__ import annotations

import json

from .state.value import Number
from .uri import URI

def _decode_tensor(payload: object) -> object:
    if not (isinstance(payload, list) and len(payload) == 2):
        return payload

    meta, values = payload
    if not (isinstance(meta, list) and len(meta) == 2 and isinstance(meta[0], str) and isinstance(meta[1], list)):
        return payload

    dtype = meta[0]
    try:
        shape = [int(dim) for dim in meta[1]]
    except (TypeError, ValueError):
        return payload
    decoded_values = [decode_payload(value) for value in values] if isinstance(values, list) else values

    from . import _local
    from .collection.tensor import Tensor

    try:
        local = _local.backend()
        native_tensor = getattr(local, "Tensor", None)
    except ImportError:
        native_tensor = None
        tensor_type = None
    else:
        tensor_type = Tensor

    if native_tensor is not None and tensor_type is not None:
        try:
            if dtype == str(URI(Number, "float", "32")):
                return tensor_type(native=native_tensor.dense_f32(shape, [float(value) for value in decoded_values]))
            if dtype == str(URI(Number, "float", "64")):
                return tensor_type(native=native_tensor.dense_f64(shape, [float(value) for value in decoded_values]))
            if dtype == str(URI(Number, "uint", "64")):
                return tensor_type(native=native_tensor.dense_u64(shape, [int(value) for value in decoded_values]))
        except (AttributeError, TypeError, ValueError):
            pass

    if dtype in (
        str(URI(Number, "float", "32")),
        str(URI(Number, "float", "64")),
        str(URI(Number, "uint", "64")),
    ):
        raise TypeError(f"cannot decode tensor dtype {dtype} into local Tensor backend")

    raise TypeError(f"unsupported tensor dtype {dtype}")


def _decode_collections(payload: object) -> object:
    from .collection.tensor import Tensor

    if isinstance(payload, dict) and len(payload) == 1:
        (key, value), = payload.items()
        if key == str(URI(Tensor)):
            return _decode_tensor(value)
    if isinstance(payload, dict):
        return {k: _decode_collections(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_decode_collections(item) for item in payload]
    return payload


def decode_payload(payload: object) -> object:
    from .state import OpDef, TCRef
    from .state.value import Map, Tuple, Value, form_of as value_form_of

    def _project_value(value: Value) -> object:
        if isinstance(value, Map):
            map_form = value_form_of(value)
            assert isinstance(map_form, dict)
            return {k: _project_value(v) for k, v in map_form.items()}

        if isinstance(value, Tuple):
            tuple_form = value_form_of(value)
            assert isinstance(tuple_form, list)
            return [_project_value(v) for v in tuple_form]

        return value_form_of(value)

    unwrapped = _decode_collections(payload)
    if unwrapped is None or isinstance(unwrapped, (bool, int, float, str)):
        return unwrapped

    if isinstance(unwrapped, dict):
        if len(unwrapped) == 1:
            (key, _value), = unwrapped.items()
            if isinstance(key, str) and key.startswith(str(URI(OpDef))):
                try:
                    return OpDef.from_json(unwrapped)
                except (TypeError, ValueError):
                    pass

            if isinstance(key, str) and (key.startswith("/") or key.startswith("$")):
                try:
                    return TCRef.from_json(unwrapped)
                except (TypeError, ValueError):
                    pass

            try:
                value = Value.from_json(unwrapped)
                return _project_value(value)
            except (TypeError, ValueError):
                pass

        return {k: decode_payload(v) for k, v in unwrapped.items()}

    if isinstance(unwrapped, list):
        return [decode_payload(item) for item in unwrapped]

    return unwrapped


def decode_response_body(response: object) -> object:
    body = getattr(response, "body", None)
    if body is None:
        raise AssertionError("response missing body")

    value = body.value()
    text = value.to_json() if hasattr(value, "to_json") else value
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="replace")
    if isinstance(text, str):
        payload = json.loads(text)
    else:
        payload = text
    return decode_payload(payload)
