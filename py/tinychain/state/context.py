from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Iterable

from .scalar import IdRef, Scalar, TCRef, autobox


@dataclass(frozen=True, slots=True)
class ContextResult:
    form: list[tuple[str, Scalar]]
    result: object


class Context:
    def __init__(self) -> None:
        object.__setattr__(self, "_form", [])
        object.__setattr__(self, "_names", set())
        object.__setattr__(self, "_bound", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_counter", 0)

    def bind(self, value: object, name: str) -> Scalar:
        original = name
        if name in self._names:
            while True:
                alias = f"{original}_{self._counter}"
                object.__setattr__(self, "_counter", self._counter + 1)
                if alias not in self._names:
                    name = alias
                    break
        self._names.add(name)
        boxed = autobox(value)
        if isinstance(boxed, Scalar):
            object.__setattr__(boxed, "_ctx", self)
        self._form.append((name, boxed))

        cls = type(boxed) if isinstance(boxed, Scalar) else Scalar
        try:
            bound = cls(ref=TCRef(IdRef(name)), ctx=self)
        except TypeError:
            try:
                bound = cls(ref=TCRef(IdRef(name)))
                if isinstance(bound, Scalar):
                    object.__setattr__(bound, "_ctx", self)
            except TypeError:
                # Fall back to an untyped scalar if a subclass constructor diverges.
                bound = Scalar(ref=TCRef(IdRef(name)), ctx=self)

        self._bound[name] = bound
        self._bound[original] = bound
        self._values[name] = boxed
        self._values[original] = boxed
        return bound

    def bind_auto(self, value: object, *, prefix: str = "_tmp") -> Scalar:
        while True:
            name = f"{prefix}{self._counter}"
            object.__setattr__(self, "_counter", self._counter + 1)
            if name in self._names:
                continue
            return self.bind(value, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self.bind(value, name)

    def __getattr__(self, name: str) -> Scalar:
        bound = self._bound.get(name)
        if bound is None:
            raise AttributeError(name)
        return bound

    def value(self, name: str) -> Scalar:
        value = self._values.get(name)
        if value is None:
            raise AttributeError(name)
        return value

    def result(self, value: object) -> ContextResult:
        return ContextResult(list(self._form), value)

    def form(self) -> Iterable[tuple[str, Scalar]]:
        return list(self._form)


class _ContextScope:
    def __init__(self) -> None:
        self._token: contextvars.Token[Context | None] | None = None

    def __enter__(self) -> Context:
        ctx = Context()
        self._token = _current_context.set(ctx)
        return ctx

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _current_context.reset(self._token)
            self._token = None


_current_context: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "tinychain_state_context",
    default=None,
)


def context() -> Context:
    ctx = _current_context.get()
    if ctx is None:
        ctx = Context()
        _current_context.set(ctx)
    return ctx


def current_context() -> Context | None:
    return _current_context.get()


def scoped_context() -> _ContextScope:
    return _ContextScope()
