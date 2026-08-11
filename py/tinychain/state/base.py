from __future__ import annotations

from ..uri import URI


class State:
    __slots__ = ("_form",)
    __uri__: URI = URI("state")

    def __init__(self, form: object = None):
        self._form = form

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_ctx":
            raise AttributeError("State bindings belong to Context, not State instances")
        object.__setattr__(self, name, value)

    def __hash__(self) -> int:
        from ._ops import _freeze_shape
        from .scalar import form_of

        return hash(_freeze_shape(form_of(self)))

    def to_json(self) -> object:
        from .scalar import _json_of, form_of

        return _json_of(form_of(self))

    @classmethod
    def _from_opref(cls, opref):
        return cls(opref)

    @classmethod
    def _get_ref(cls, subject: str, key: object = None):
        from .scalar.refs import GetOpRef

        return cls._from_opref(GetOpRef(subject, key))

    @classmethod
    def _put_ref(cls, subject: str, key: object, value: object):
        from .scalar.refs import PutOpRef

        return cls._from_opref(PutOpRef(subject, key, value))

    @classmethod
    def _post_ref(cls, subject: str, params: object | None = None):
        from .scalar.refs import PostOpRef

        return cls._from_opref(PostOpRef(subject, params or {}))

    @classmethod
    def _delete_ref(cls, subject: str, key: object = None):
        from .scalar.refs import DeleteOpRef

        return cls._from_opref(DeleteOpRef(subject, key))

    def _get(self, method: str | None = None, key: object = None, *, rtype: type | None = None):
        from ._ops import _state_get

        return _state_get(self, self._form, method, key, rtype=rtype)

    def _put(self, value: object, method: str | None = None, key: object = None, *, rtype: type | None = None):
        from ._ops import _state_put

        return _state_put(self, self._form, value, method, key, rtype=rtype)

    def _post(self, method: str | None = None, params: object | None = None, *, rtype: type | None = None):
        from ._ops import _state_post

        return _state_post(self, self._form, method, params, rtype=rtype)

    def _delete(self, method: str | None = None, key: object = None, *, rtype: type | None = None):
        from ._ops import _state_delete

        return _state_delete(self, self._form, method, key, rtype=rtype)
