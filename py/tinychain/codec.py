from __future__ import annotations


def decode_payload(payload: object, codec=None) -> object:
    from . import _native

    return (codec or _native.Codec()).materialize(payload)


def decode_response_body(response: object) -> object:
    body = getattr(response, "body", None)
    if body is None:
        raise AssertionError("response missing body")

    return decode_payload(body.value())
