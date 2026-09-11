from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Callable, Iterable

from .state.base import State
from .state.scalar import IdRef, autobox, form_of
from .state.scalar.refs import TCRef, _validate_id


@dataclass(frozen=True, slots=True)
class ContextResult:
    form: tuple[tuple[str, State], ...]
    result: object


class Context:
    """A lexical, single-assignment operation-authoring context."""

    __slots__ = (
        "_form",
        "_names",
        "_bound",
        "_values",
        "_by_identity",
        "_counter",
        "_parent",
        "_normalize",
    )

    def __init__(
        self,
        parent: "Context | None" = None,
        normalize: Callable[[object], object] | None = None,
    ) -> None:
        self._form: list[tuple[str, State]] = []
        self._names: set[str] = set()
        self._bound: dict[str, State] = {}
        self._values: dict[str, State] = {}
        self._by_identity: dict[int, tuple[object, str]] = {}
        self._counter = 0
        self._parent = parent
        self._normalize = (
            normalize if normalize is not None else (parent._normalize if parent else None)
        )

    def _contains_visible(self, name: str) -> bool:
        return name in self._names or (
            self._parent is not None and self._parent._contains_visible(name)
        )

    def _visible_names(self) -> set[str]:
        names = self._parent._visible_names() if self._parent is not None else set()
        names.update(self._names)
        return names

    def _reserve(self, names: Iterable[str]) -> None:
        """Reserve invocation bindings supplied outside the operation form."""
        for name in names:
            name = _validate_id(name)
            if name == "self":
                raise ValueError("$self is reserved and cannot be bound")
            if self._contains_visible(name):
                raise ValueError(f"Context already has a visible binding named {name!r}")
            self._names.add(name)

    def _bound_identity(self, value: object) -> State | None:
        entry = self._by_identity.get(id(value))
        if entry is not None:
            source, name = entry
            if source is value:
                return self._bound[name]
        if self._parent is not None:
            return self._parent._bound_identity(value)
        return None

    def bind(self, name: str, value: object) -> State:
        name = _validate_id(name)
        if name == "self":
            raise ValueError("$self is reserved and cannot be bound")
        if self._contains_visible(name):
            raise ValueError(f"Context already has a visible binding named {name!r}")
        if self._bound_identity(value) is not None:
            raise ValueError("cannot explicitly bind one value under two names")

        normalized = self._normalize(value) if self._normalize is not None else value
        boxed = autobox(normalized)
        self._names.add(name)
        self._form.append((name, boxed))

        cls = type(boxed) if isinstance(boxed, State) else State
        bound = cls(IdRef(name))
        self._bound[name] = bound
        self._values[name] = boxed
        # Retain the source object so a recycled CPython object ID cannot resolve
        # an unrelated value later in this lexical context.
        self._by_identity[id(value)] = (value, name)
        return bound

    def bind_auto(self, value: object, *, prefix: str = "_tmp") -> State:
        bound = self._bound_identity(value)
        if bound is not None:
            return bound
        prefix = _validate_id(prefix)
        while True:
            name = f"{prefix}{self._counter}"
            self._counter += 1
            if not self._contains_visible(name):
                return self.bind(name, value)

    def __setattr__(self, name: str, value: object) -> None:
        # Public assignment declares a lexical binding; underscore-prefixed
        # attributes are Context's own implementation state.
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        self.bind(name, value)

    def __getattr__(self, name: str) -> State:
        bound = self._bound.get(name)
        if bound is None and self._parent is not None:
            return getattr(self._parent, name)
        if bound is None:
            raise AttributeError(name)
        return bound

    def value(self, name: str) -> State:
        value = self._values.get(name)
        if value is None and self._parent is not None:
            return self._parent.value(name)
        if value is None:
            raise AttributeError(name)
        return value

    def result(self, value: object) -> ContextResult:
        return ContextResult(tuple(self._form), value)

    def form(self) -> tuple[tuple[str, State], ...]:
        return tuple(self._form)


class _ContextScope:
    def __init__(self, normalize: Callable[[object], object] | None = None) -> None:
        self._token: contextvars.Token[Context | None] | None = None
        self._normalize = normalize

    def __enter__(self) -> Context:
        context = Context(_current_context.get(), self._normalize)
        self._token = _current_context.set(context)
        return context

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _current_context.reset(self._token)
            self._token = None


_current_context: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "tinychain_context",
    default=None,
)


def _active_context() -> Context | None:
    return _current_context.get()


def _require_context() -> Context:
    context = _active_context()
    if context is None:
        raise RuntimeError("TinyChain symbolic execution requires scoped_context()")
    return context


def _bind_subject(owner: object) -> str | None:
    context = _active_context()
    if context is None:
        return None

    bound = context.bind_auto(owner)
    bound_form = form_of(bound)
    if isinstance(bound_form, TCRef):
        bound_ref_form = form_of(bound_form)
        if isinstance(bound_ref_form, IdRef):
            return bound_ref_form.key()
    return None


def _scoped_context(
    normalize: Callable[[object], object] | None = None,
) -> _ContextScope:
    return _ContextScope(normalize)


def scoped_context() -> _ContextScope:
    return _scoped_context()
