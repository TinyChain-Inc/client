from __future__ import annotations

from typing import TYPE_CHECKING

from ..uri import URI

if TYPE_CHECKING:
    from .context import Context


class State:
    __slots__ = ("_form", "_ctx")
    __uri__: URI = URI("state")

    def __init__(self, form: object = None, *, ref: TCRef | None = None, ctx: "Context | None" = None):
        if form is not None and ref is not None:
            raise TypeError("State accepts either form or ref, not both")

        self._form = ref if ref is not None else form
        self._ctx = ctx

    def __hash__(self) -> int:
        from ._ops import _freeze_shape
        from .scalar import form_of

        return hash(_freeze_shape(form_of(self)))

    def to_json(self) -> object:
        from .scalar import _json_of, form_of

        return _json_of(form_of(self))

    @classmethod
    def _from_opref(cls, opref, *, ctx: "Context | None" = None):
        try:
            if ctx is None:
                return cls(ref=opref)
            return cls(ref=opref, ctx=ctx)
        except TypeError:
            state = cls(opref)
            if isinstance(state, State):
                state._ctx = ctx
            return state

    @classmethod
    def _get_ref(cls, subject: str, key: object = None, *, ctx: "Context | None" = None):
        from .scalar.refs import GetOpRef

        return cls._from_opref(GetOpRef(subject, key), ctx=ctx)

    @classmethod
    def _put_ref(cls, subject: str, key: object, value: object, *, ctx: "Context | None" = None):
        from .scalar.refs import PutOpRef

        return cls._from_opref(PutOpRef(subject, key, value), ctx=ctx)

    @classmethod
    def _post_ref(cls, subject: str, params: object | None = None, *, ctx: "Context | None" = None):
        from .scalar.refs import PostOpRef

        return cls._from_opref(PostOpRef(subject, params or {}), ctx=ctx)

    @classmethod
    def _delete_ref(cls, subject: str, key: object = None, *, ctx: "Context | None" = None):
        from .scalar.refs import DeleteOpRef

        return cls._from_opref(DeleteOpRef(subject, key), ctx=ctx)

    def _get(self, method: str | None = None, key: object = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        from ._ops import _state_get

        return _state_get(self, self._form, method, key, rtype=rtype, ctx=ctx)

    def _put(self, value: object, method: str | None = None, key: object = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        from ._ops import _state_put

        return _state_put(self, self._form, value, method, key, rtype=rtype, ctx=ctx)

    def _post(self, method: str | None = None, params: object | None = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        from ._ops import _state_post

        return _state_post(self, self._form, method, params, rtype=rtype, ctx=ctx)

    def _delete(self, method: str | None = None, key: object = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        from ._ops import _state_delete

        return _state_delete(self, self._form, method, key, rtype=rtype, ctx=ctx)
