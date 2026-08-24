"""Frozen exceptions that Python's exception machinery can still annotate.

A frozen dataclass that subclasses `Exception` is two things at once, and the
two disagree. `@dataclass(frozen=True)` generates a `__setattr__` that refuses
every assignment, while Python's exception protocol assigns state *on the
propagating instance*. Some of those assignments are made from Python rather
than from C -- a generator-based `@contextlib.contextmanager` re-raises by
assigning `__traceback__`, and `BaseException.add_note` assigns `__notes__` --
so the frozen `__setattr__` refuses them and the exception a consumer catches
is a `FrozenInstanceError` instead of the error that was raised. For a
categorized error type, that erases the category, which is the whole public
contract.

`allow_exception_state` reopens exactly the attributes the protocol owns and
nothing else: declared fields and any attribute a caller invents stay frozen.
"""

from __future__ import annotations

from typing import TypeVar


# The attributes Python's exception protocol owns, rather than a hand-picked
# list of the ones that happened to be seen failing. These are the language's
# own attributes, so the set does not drift with our types.
_EXCEPTION_STATE_ATTRIBUTES = frozenset(
    {
        "__traceback__",
        "__cause__",
        "__context__",
        "__suppress_context__",
        "__notes__",
    }
)

_ExceptionClass = TypeVar("_ExceptionClass", bound=type[BaseException])


def allow_exception_state(cls: _ExceptionClass) -> _ExceptionClass:
    """Let Python's exception machinery assign its own state on *cls*.

    Apply it above `@dataclass(frozen=True)`, so the frozen `__setattr__`
    already exists to be wrapped by the time this runs::

        @allow_exception_state
        @dataclass(frozen=True)
        class SomeError(Exception):
            ...

    The wrapping is installed after the class body rather than written into it
    because `@dataclass(frozen=True)` raises `TypeError: Cannot overwrite
    attribute __setattr__` at class-definition time if the body defines one.
    """
    frozen_setattr = cls.__setattr__

    def __setattr__(self: BaseException, name: str, value: object) -> None:
        if name in _EXCEPTION_STATE_ATTRIBUTES:
            object.__setattr__(self, name, value)
            return
        frozen_setattr(self, name, value)

    cls.__setattr__ = __setattr__
    return cls
