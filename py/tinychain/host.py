"""Lightweight HTTP host client for TinyChain."""

from __future__ import annotations

import json
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests

from .auth import bearer_token as _bearer_token
from .opref import OpRef
from .ref import Ref
from .codec import decode_payload
from .uri import URI, uri


class Host:
    """A TinyChain HTTP host."""

    def __init__(
        self,
        address: str,
        *,
        token: object | None = None,
    ):
        if "://" not in address:
            raise ValueError(f"host address missing protocol: {address}")
        self.__uri__ = URI.parse(address)
        if self.__uri__.path:
            raise ValueError(
                f"Host address should not include a path: {self.__uri__.path}"
            )
        self._bearer_token = _bearer_token(token)

    def __repr__(self) -> str:
        return f"host at {self.__uri__}"

    def link(self, path: object) -> URI:
        if isinstance(path, URI):
            target = path
        else:
            target = uri(path)
            if not isinstance(target, URI):
                raise TypeError("expected a URI path for HTTP host link")

        if target.host is not None:
            return target

        return URI(
            path=target.path,
            scheme=self.__uri__.scheme,
            host=self.__uri__.host,
            port=self.__uri__.port,
        )

    def execute(self, opref: OpRef | Ref) -> object:
        if hasattr(opref, "op"):
            opref = opref.op
        if not isinstance(opref, OpRef):
            raise TypeError(f"expected OpRef, got {type(opref).__name__}")
        return self.request(
            opref.method,
            opref.path,
            body=opref.body,
            headers=opref.headers,
        )

    def url(self, target: object, route: str | None = None, **query: object) -> str:
        _reject_transaction_query(query)
        path = self.link(uri(target, *([route] if route else []))).absolute()
        _reject_transaction_control(path)
        if not query:
            return path
        encoded = urlencode(_url_query(query), doseq=True)
        return f"{path}?{encoded}"

    def request(
        self,
        method: str,
        path: object,
        *,
        body: object | None = None,
        headers: Optional[Iterable[tuple[str, str]]] = None,
    ) -> object:
        target = self.link(path)
        _reject_transaction_control(target.absolute())
        payload = None if body is None else _encode_body(body)
        merged_headers = _merge_headers(headers, self._bearer_token, payload is not None)
        response = requests.request(
            method.upper(),
            target.absolute(),
            data=payload,
            headers=merged_headers,
        )
        return _handle_response(response)


def _reject_transaction_query(query: dict[str, object]) -> None:
    if any(key.lower() == "txn_id" for key in query):
        raise ValueError("Python client does not expose transaction controls")


def _reject_transaction_control(target: str) -> None:
    query = urlsplit(target).query
    if not query:
        return
    if any(key.lower() == "txn_id" for key, _ in parse_qsl(query, keep_blank_values=True)):
        raise ValueError("Python client does not expose transaction controls")


def _encode_payload(value: object) -> object:
    from .executor import _encode_payload as _executor_encode

    return _executor_encode(value)


def _encode_body(value: object) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return json.dumps(_encode_payload(value), separators=(",", ":")).encode("utf-8")


def _url_query(query: dict[str, object]) -> dict[str, object]:
    if len(query) == 1 and "key" in query:
        return {"key": _url_value(query["key"])}
    return {
        "key": json.dumps(
            {key: _encode_payload(value) for key, value in query.items()},
            separators=(",", ":"),
        )
    }


def _url_value(value: object) -> object:
    if isinstance(value, URI):
        return value.absolute()
    if isinstance(value, (list, tuple)):
        return [_url_value(item) for item in value]
    return value


def _merge_headers(
    headers: Optional[Iterable[tuple[str, str]]],
    bearer_token: Optional[str],
    has_body: bool,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    if headers:
        for key, value in headers:
            merged[key] = value
    if bearer_token and not any(k.lower() == "authorization" for k in merged):
        merged["authorization"] = f"Bearer {bearer_token}"
    if has_body and not any(k.lower() == "content-type" for k in merged):
        merged["content-type"] = "application/json"
    merged.setdefault("accept", "application/json")
    return merged


def _handle_response(response: requests.Response) -> object:
    status = response.status_code
    if status == 204:
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"invalid JSON response: {response.text}") from exc
    if status == 200:
        return decode_payload(payload)
    raise RuntimeError(f"unexpected HTTP {status}: {payload}")
