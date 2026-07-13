from __future__ import annotations

from typing import TYPE_CHECKING

from ..scalar import DeleteOpRef, GetOpRef, IdRef, OpRef, PostOpRef, PutOpRef, Scalar, TCRef, form_of

if TYPE_CHECKING:
    from ..context import Context


class Collection:
    __slots__ = ("_form", "_ctx")

    def __init__(self, form: object = None, *, ref: TCRef | None = None, ctx: "Context | None" = None):
        if form is not None and ref is not None:
            raise TypeError("Collection accepts either form or ref, not both")

        self._form = ref if ref is not None else form
        self._ctx = ctx

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Collection):
            return False

        return form_of(self) == form_of(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(repr(self.to_json()))

    def to_json(self) -> object:
        return Scalar(self._form).to_json()

    @classmethod
    def _from_opref(cls, opref, *, ctx: "Context | None" = None):
        try:
            return cls(ref=TCRef(opref), ctx=ctx)
        except TypeError:
            collection = cls(opref)
            if isinstance(collection, Collection):
                collection._ctx = ctx
            return collection

    @classmethod
    def _get_ref(cls, subject: str, key: object = None, *, ctx: "Context | None" = None):
        return cls._from_opref(GetOpRef(subject, key), ctx=ctx)

    @classmethod
    def _put_ref(cls, subject: str, key: object, value: object, *, ctx: "Context | None" = None):
        return cls._from_opref(PutOpRef(subject, key, value), ctx=ctx)

    @classmethod
    def _post_ref(cls, subject: str, params: object | None = None, *, ctx: "Context | None" = None):
        return cls._from_opref(PostOpRef(subject, params or {}), ctx=ctx)

    @classmethod
    def _delete_ref(cls, subject: str, key: object = None, *, ctx: "Context | None" = None):
        return cls._from_opref(DeleteOpRef(subject, key), ctx=ctx)

    def _subject(self, *, ctx: "Context | None" = None) -> str:
        form = self._form
        if isinstance(form, OpRef):
            return form.subject
        runtime_form = OpRef.from_runtime(form)
        if runtime_form is not None:
            return runtime_form.subject
        if isinstance(form, TCRef):
            ref_form = form_of(form)
            if isinstance(ref_form, IdRef):
                return ref_form.key()
            if isinstance(ref_form, OpRef):
                return ref_form.subject
            runtime_ref = OpRef.from_runtime(ref_form)
            if runtime_ref is not None:
                return runtime_ref.subject

        active_ctx = ctx if ctx is not None else self._ctx
        if active_ctx is not None:
            bound = active_ctx.bind_auto(self)
            bound_form = form_of(bound)
            if isinstance(bound_form, TCRef):
                bound_ref_form = form_of(bound_form)
                if isinstance(bound_ref_form, IdRef):
                    return bound_ref_form.key()

        raise TypeError("expected a Collection id ref for an op subject")

    def _subject_ref(self, method: str | None = None, *, ctx: "Context | None" = None) -> str:
        subject = self._subject(ctx=ctx)
        return f"{subject}/{method}" if method else subject

    def _get(self, method: str | None = None, key: object = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._get_ref(self._subject_ref(method, ctx=active_ctx), key, ctx=active_ctx)

    def _put(self, value: object, method: str | None = None, key: object = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._put_ref(self._subject_ref(method, ctx=active_ctx), key, value, ctx=active_ctx)

    def _post(self, method: str | None = None, params: object | None = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._post_ref(self._subject_ref(method, ctx=active_ctx), params, ctx=active_ctx)

    def _delete(self, method: str | None = None, key: object = None, *, rtype: type | None = None, ctx: "Context | None" = None):
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._delete_ref(self._subject_ref(method, ctx=active_ctx), key, ctx=active_ctx)
