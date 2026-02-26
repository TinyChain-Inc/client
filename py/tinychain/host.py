"""Lightweight HTTP host client for TinyChain."""

from __future__ import annotations

import json
from typing import Iterable, Optional

import requests

from .opref import OpRef
from .uri import URI, uri


class Host:
    """A TinyChain HTTP host."""

    def __init__(self, address: str):
        if "://" not in address:
            raise ValueError(f"host address missing protocol: {address}")
        self.__uri__ = URI.parse(address)
        if self.__uri__.path:
            raise ValueError(
                f"Host address should not include a path: {self.__uri__.path}"
            )

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

    def execute(
        self,
        opref: OpRef,
        *,
        bearer_token: Optional[str] = None,
    ) -> object:
        if not isinstance(opref, OpRef):
            raise TypeError(f"expected OpRef, got {type(opref).__name__}")
        return self.request(
            opref.method,
            opref.path,
            body=opref.body,
            headers=opref.headers,
            bearer_token=bearer_token,
        )

    def request(
        self,
        method: str,
        path: object,
        *,
        body: object | None = None,
        headers: Optional[Iterable[tuple[str, str]]] = None,
        bearer_token: Optional[str] = None,
    ) -> object:
        target = self.link(path)
        payload = None if body is None else _encode_body(body)
        merged_headers = _merge_headers(headers, bearer_token, payload is not None)
        response = requests.request(
            method.upper(),
            target.absolute(),
            data=payload,
            headers=merged_headers,
        )
        return _handle_response(response)


def _encode_payload(value: object) -> object:
    from .executor import _encode_payload as _executor_encode

    return _executor_encode(value)


def _encode_body(value: object) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return json.dumps(_encode_payload(value), separators=(",", ":")).encode("utf-8")


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
        return payload
    raise RuntimeError(f"unexpected HTTP {status}: {payload}")
