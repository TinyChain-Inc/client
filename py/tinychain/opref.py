from __future__ import annotations

from typing import Any, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")
_MISSING = object()


class OpRef(Generic[T]):
    __slots__ = ()

    METHOD: str = ""

    @property
    def method(self) -> str:
        return type(self).METHOD

    @property
    def path(self) -> str:
        raise NotImplementedError()

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        raise NotImplementedError()

    @property
    def body(self) -> Any:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, OpRef)
            and type(self) is type(other)
            and self.path == other.path
            and self.headers == other.headers
            and self.body == other.body
        )

    def __hash__(self) -> int:
        return hash((type(self), self.path, self.headers, repr(self.body)))

    def _replace(
        self,
        *,
        path: str | object = _MISSING,
        headers: tuple[tuple[str, str], ...] | object = _MISSING,
        body: Any = _MISSING,
    ) -> "OpRef[T]":
        raise NotImplementedError()

    def with_headers(self, headers: Optional[Iterable[tuple[str, str]]]) -> "OpRef[T]":
        if not headers:
            return self
        return self._replace(headers=self.headers + tuple(headers))

    def with_body(self, body: Any) -> "OpRef[T]":
        return self._replace(body=body)


class GetOpRef(OpRef[T]):
    __slots__ = ("_path", "_headers", "_body")

    METHOD = "GET"

    def __init__(
        self,
        path: str,
        *,
        headers: Optional[Iterable[tuple[str, str]]] = None,
        body: Any = None,
    ):
        self._path = path
        self._headers = tuple(headers or ())
        self._body = body

    @property
    def path(self) -> str:
        return self._path

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        return self._headers

    @property
    def body(self) -> Any:
        return self._body

    def _replace(
        self,
        *,
        path: str | object = _MISSING,
        headers: tuple[tuple[str, str], ...] | object = _MISSING,
        body: Any = _MISSING,
    ) -> "GetOpRef[T]":
        return GetOpRef(
            self.path if path is _MISSING else path,
            headers=self.headers if headers is _MISSING else headers,
            body=self.body if body is _MISSING else body,
        )


class PutOpRef(OpRef[T]):
    __slots__ = ("_path", "_headers", "_body")

    METHOD = "PUT"

    def __init__(
        self,
        path: str,
        *,
        headers: Optional[Iterable[tuple[str, str]]] = None,
        body: Any = None,
    ):
        self._path = path
        self._headers = tuple(headers or ())
        self._body = body

    @property
    def path(self) -> str:
        return self._path

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        return self._headers

    @property
    def body(self) -> Any:
        return self._body

    def _replace(
        self,
        *,
        path: str | object = _MISSING,
        headers: tuple[tuple[str, str], ...] | object = _MISSING,
        body: Any = _MISSING,
    ) -> "PutOpRef[T]":
        return PutOpRef(
            self.path if path is _MISSING else path,
            headers=self.headers if headers is _MISSING else headers,
            body=self.body if body is _MISSING else body,
        )


class PostOpRef(OpRef[T]):
    __slots__ = ("_path", "_headers", "_body")

    METHOD = "POST"

    def __init__(
        self,
        path: str,
        *,
        headers: Optional[Iterable[tuple[str, str]]] = None,
        body: Any = None,
    ):
        self._path = path
        self._headers = tuple(headers or ())
        self._body = body

    @property
    def path(self) -> str:
        return self._path

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        return self._headers

    @property
    def body(self) -> Any:
        return self._body

    def _replace(
        self,
        *,
        path: str | object = _MISSING,
        headers: tuple[tuple[str, str], ...] | object = _MISSING,
        body: Any = _MISSING,
    ) -> "PostOpRef[T]":
        return PostOpRef(
            self.path if path is _MISSING else path,
            headers=self.headers if headers is _MISSING else headers,
            body=self.body if body is _MISSING else body,
        )


class DeleteOpRef(OpRef[T]):
    __slots__ = ("_path", "_headers", "_body")

    METHOD = "DELETE"

    def __init__(
        self,
        path: str,
        *,
        headers: Optional[Iterable[tuple[str, str]]] = None,
        body: Any = None,
    ):
        self._path = path
        self._headers = tuple(headers or ())
        self._body = body

    @property
    def path(self) -> str:
        return self._path

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        return self._headers

    @property
    def body(self) -> Any:
        return self._body

    def _replace(
        self,
        *,
        path: str | object = _MISSING,
        headers: tuple[tuple[str, str], ...] | object = _MISSING,
        body: Any = _MISSING,
    ) -> "DeleteOpRef[T]":
        return DeleteOpRef(
            self.path if path is _MISSING else path,
            headers=self.headers if headers is _MISSING else headers,
            body=self.body if body is _MISSING else body,
        )


def get(path: str, *, headers: Optional[Iterable[tuple[str, str]]] = None, body: Any = None) -> OpRef[Any]:
    return GetOpRef(path, headers=headers, body=body)


def put(path: str, *, headers: Optional[Iterable[tuple[str, str]]] = None, body: Any = None) -> OpRef[Any]:
    return PutOpRef(path, headers=headers, body=body)


def post(path: str, *, headers: Optional[Iterable[tuple[str, str]]] = None, body: Any = None) -> OpRef[Any]:
    return PostOpRef(path, headers=headers, body=body)


def delete(path: str, *, headers: Optional[Iterable[tuple[str, str]]] = None, body: Any = None) -> OpRef[Any]:
    return DeleteOpRef(path, headers=headers, body=body)
