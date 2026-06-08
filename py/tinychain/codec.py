from __future__ import annotations

import json

from .uri import uri


_COLLECTION_TENSOR = uri("state", "collection", "tensor").path
_DTYPE_F32 = uri("state", "scalar", "value", "number", "float", "32").path
_DTYPE_U64 = uri("state", "scalar", "value", "number", "uint", "64").path


def _decode_tensor(payload: object) -> object:
    if not (isinstance(payload, list) and len(payload) == 2):
        return payload

    meta, values = payload
    if not (isinstance(meta, list) and len(meta) == 2 and isinstance(meta[0], str) and isinstance(meta[1], list)):
        return payload

    dtype = meta[0]
    try:
        shape = [int(dim) for dim in meta[1]]
    except Exception:
        return payload
    decoded_values = [decode_payload(value) for value in values] if isinstance(values, list) else values

    try:
        import tinychain as tc
    except Exception:
        tc = None

    if tc is not None and hasattr(tc, "Tensor"):
        try:
            if dtype == _DTYPE_F32:
                return tc.Tensor.dense_f32(shape, [float(value) for value in decoded_values])
            if dtype == _DTYPE_U64:
                return tc.Tensor.dense_u64(shape, [int(value) for value in decoded_values])
        except Exception:
            pass

    return {
        "dtype": dtype,
        "shape": shape,
        "values": decoded_values,
    }


def _decode_collections(payload: object) -> object:
    if isinstance(payload, dict) and len(payload) == 1:
        (key, value), = payload.items()
        if key == _COLLECTION_TENSOR:
            return _decode_tensor(value)
    if isinstance(payload, dict):
        return {k: _decode_collections(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_decode_collections(item) for item in payload]
    return payload


def decode_payload(payload: object) -> object:
    from .state import OpDef, TCRef
    from .state.value import Value

    unwrapped = _decode_collections(payload)
    if unwrapped is None or isinstance(unwrapped, (bool, int, float, str)):
        return unwrapped

    if isinstance(unwrapped, dict):
        if len(unwrapped) == 1:
            (key, _value), = unwrapped.items()
            if isinstance(key, str) and key.startswith(uri("state", "scalar", "op").path):
                try:
                    return OpDef.from_json(unwrapped)
                except Exception:
                    pass

            if isinstance(key, str) and (key.startswith("/") or key.startswith("$")):
                try:
                    return TCRef.from_json(unwrapped)
                except Exception:
                    pass

            try:
                value = Value.from_json(unwrapped)
                return value.value
            except Exception:
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
