from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from . import Scalar
    from ..value import Value


class Op:
    """
    Marker base class for callable op handles.

    Use concrete method handles (`Get`, `Put`, `Post`, `Delete`) directly.
    """

    __slots__ = ()

    def __init__(self):
        if type(self) is Op:
            raise TypeError("Op is abstract; use Get, Put, Post, or Delete")


class Get(Op):
    __slots__ = ("subject",)

    def __init__(self, subject: str):
        self.subject = subject

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Get) and self.subject == other.subject

    def __hash__(self) -> int:
        return hash((Get, self.subject))

    def __call__(self, key: Value | None = None) -> "Scalar":
        from . import GetOpRef, Scalar, TCRef
        from ..value import Null, Value

        if key is None:
            key = Null()
        if not isinstance(key, Value):
            raise TypeError("Get expects key to be a Value")

        return Scalar(ref=TCRef(GetOpRef(self.subject, key)))


class Put(Op):
    __slots__ = ("subject",)

    def __init__(self, subject: str):
        self.subject = subject

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Put) and self.subject == other.subject

    def __hash__(self) -> int:
        return hash((Put, self.subject))

    def __call__(self, key: "Scalar", value: "Scalar") -> "Scalar":
        from . import PutOpRef, Scalar, TCRef

        if not isinstance(key, Scalar):
            raise TypeError("Put expects key to be State (Scalar or Value)")
        if not isinstance(value, Scalar):
            raise TypeError("Put expects value to be State (Scalar or Value)")

        return Scalar(ref=TCRef(PutOpRef(self.subject, key, value)))


class Post(Op):
    __slots__ = ("subject",)

    def __init__(self, subject: str):
        self.subject = subject

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Post) and self.subject == other.subject

    def __hash__(self) -> int:
        return hash((Post, self.subject))

    def __call__(self, params: Mapping[str, "Scalar"]) -> "Scalar":
        from . import PostOpRef, Scalar, TCRef, _sorted_items

        if not isinstance(params, Mapping):
            raise TypeError("Post expects params to be a map of State values")

        typed_params: dict[str, Scalar] = {}
        for key, value in _sorted_items(params):
            if not isinstance(key, str):
                raise TypeError("Post params keys must be strings")
            if not isinstance(value, Scalar):
                raise TypeError("Post expects params to be a map of State values")
            typed_params[key] = value

        return Scalar(ref=TCRef(PostOpRef(self.subject, typed_params)))


class Delete(Op):
    __slots__ = ("subject",)

    def __init__(self, subject: str):
        self.subject = subject

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Delete) and self.subject == other.subject

    def __hash__(self) -> int:
        return hash((Delete, self.subject))

    def __call__(self, key: Value | None = None) -> "Scalar":
        from . import DeleteOpRef, Scalar, TCRef
        from ..value import Null, Value

        if key is None:
            key = Null()
        if not isinstance(key, Value):
            raise TypeError("Delete expects key to be a Value")

        return Scalar(ref=TCRef(DeleteOpRef(self.subject, key)))
