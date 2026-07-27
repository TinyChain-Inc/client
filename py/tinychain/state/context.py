from __future__ import annotations

from dataclasses import dataclass
import contextvars
from typing import Iterable

from .scalar import IdRef, Scalar, TCRef, autobox


@dataclass(frozen=True, slots=True)
class ContextResult:
    form: list[tuple[str, Scalar]]
    result: Scalar


class Context:
    def __init__(self) -> None:
        object.__setattr__(self, "_form", [])
        object.__setattr__(self, "_names", set())
        object.__setattr__(self, "_bound", {})
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
        self._form.append((name, boxed))

        cls = type(boxed) if isinstance(boxed, Scalar) else Scalar
        try:
            bound = cls(ref=TCRef.id(name))
        except TypeError:
            # Fall back to an untyped scalar if a subclass constructor diverges.
            bound = Scalar(ref=TCRef.id(name))

        self._bound[name] = bound
        self._bound[original] = bound
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

    def result(self, value: object) -> ContextResult:
        return ContextResult(list(self._form), autobox(value))

    def form(self) -> Iterable[tuple[str, Scalar]]:
        return list(self._form)


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


def scoped_context() -> _ContextScope:
    return _ContextScope()
