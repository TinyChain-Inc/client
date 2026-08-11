from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Iterable

from .base import State
from .scalar import IdRef, autobox


@dataclass(frozen=True, slots=True)
class ContextResult:
    form: list[tuple[str, State]]
    result: object


class Context:
    def __init__(self) -> None:
        object.__setattr__(self, "_form", [])
        object.__setattr__(self, "_names", set())
        object.__setattr__(self, "_bound", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_by_identity", {})
        object.__setattr__(self, "_counter", 0)

    def bind(self, name: str, value: object) -> State:
        if not isinstance(name, str):
            raise TypeError("Context binding names must be strings")
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
        self._form.append((name, boxed))

        cls = type(boxed) if isinstance(boxed, State) else State
        bound = cls(IdRef(name))

        self._bound[name] = bound
        self._bound[original] = bound
        self._values[name] = boxed
        self._values[original] = boxed
        self._by_identity[id(value)] = name
        return bound

    def bind_auto(self, value: object, *, prefix: str = "_tmp") -> State:
        name = self._by_identity.get(id(value))
        if name is not None:
            return self._bound[name]
        while True:
            name = f"{prefix}{self._counter}"
            object.__setattr__(self, "_counter", self._counter + 1)
            if name in self._names:
                continue
            return self.bind(name, value)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self.bind(name, value)

    def __getattr__(self, name: str) -> State:
        bound = self._bound.get(name)
        if bound is None:
            raise AttributeError(name)
        return bound

    def value(self, name: str) -> State:
        value = self._values.get(name)
        if value is None:
            raise AttributeError(name)
        return value

    def result(self, value: object) -> ContextResult:
        return ContextResult(list(self._form), value)

    def form(self) -> Iterable[tuple[str, State]]:
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
