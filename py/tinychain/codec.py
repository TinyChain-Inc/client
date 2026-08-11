from __future__ import annotations

import json

from .uri import URI


def decode_payload(payload: object) -> object:
    from .state import OpDef, TCRef
    from .state.collection import Collection
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

    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload

    if isinstance(payload, dict):
        if len(payload) == 1:
            (key, _value), = payload.items()
            collection = Collection.decode(payload)
            if collection is not None:
                return collection

            if isinstance(key, str) and key.startswith(str(URI(OpDef))):
                try:
                    return OpDef.from_json(payload)
                except (TypeError, ValueError):
                    pass

            if isinstance(key, str) and (key.startswith("/") or key.startswith("$")):
                try:
                    return TCRef.from_json(payload)
                except (TypeError, ValueError):
                    pass

                try:
                    value = Value.from_json(payload)
                    return _project_value(value)
                except (TypeError, ValueError):
                    pass

        return {k: decode_payload(v) for k, v in payload.items()}

    if isinstance(payload, list):
        return [decode_payload(item) for item in payload]

    return payload


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
