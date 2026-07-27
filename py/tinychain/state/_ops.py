from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..uri import path, uri
from .scalar.refs import (
    Cond,
    ForEach,
    IdRef,
    OpRef,
    TCRef,
    While,
)

if TYPE_CHECKING:
    from .context import Context
    from .scalar import Scalar
    from .value import Value


def _freeze_shape(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return tuple((key, _freeze_shape(val)) for key, val in _sorted_items(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_shape(item) for item in value)
    return value


def _sorted_items(obj: Mapping[str, object]) -> list[tuple[str, object]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _owner_ctx(owner: object, ctx: "Context | None" = None) -> "Context | None":
    return ctx if ctx is not None else getattr(owner, "_ctx", None)


def _active_subject_ctx(owner: object, ctx: "Context | None" = None) -> "Context | None":
    active_ctx = _owner_ctx(owner, ctx)
    from .context import resolve_context

    return resolve_context(active_ctx)


def _stage_subject(owner: object, active_ctx: "Context | None") -> str | None:
    if active_ctx is None:
        return None

    try:
        bound = active_ctx.bind_auto(owner)
    except TypeError:
        return None

    from .scalar import form_of

    bound_form = form_of(bound)
    if isinstance(bound_form, TCRef):
        bound_ref_form = form_of(bound_form)
        if isinstance(bound_ref_form, IdRef):
            return bound_ref_form.key()

    return None


def _subject_from_candidate(candidate: object, active_ctx: "Context | None", owner: object) -> str | None:
    if isinstance(candidate, OpRef):
        subject = candidate.subject
    else:
        runtime_candidate = OpRef.from_runtime(candidate)
        if runtime_candidate is None:
            return None
        subject = runtime_candidate.subject

    staged = _stage_subject(owner, active_ctx)
    return staged if staged is not None else subject


def _state_subject(owner: object, form: object, *, ctx: "Context | None" = None) -> str:
    from .scalar import form_of

    active_ctx = _active_subject_ctx(owner, ctx)

    if isinstance(form, OpRef):
        subject = _subject_from_candidate(form, active_ctx, owner)
        if subject is not None:
            return subject

    subject = _subject_from_candidate(form, active_ctx, owner)
    if subject is not None:
        return subject

    if isinstance(form, TCRef):
        ref_form = form_of(form)
        if isinstance(ref_form, IdRef):
            return ref_form.key()

        subject = _subject_from_candidate(ref_form, active_ctx, owner)
        if subject is not None:
            return subject

        if isinstance(ref_form, Cond):
            return path(uri(TCRef, "cond"))
        if isinstance(ref_form, While):
            return path(uri(TCRef, "while"))
        if isinstance(ref_form, ForEach):
            return path(uri(TCRef, "for_each"))

    if isinstance(form, Mapping) and len(form) == 1:
        (key, _value), = form.items()
        if isinstance(key, str) and key.startswith("$"):
            return key

    if isinstance(form, Sequence) and not isinstance(form, (str, bytes, bytearray)):
        staged = _stage_subject(owner, active_ctx)
        if staged is not None:
            return staged

    staged = _stage_subject(owner, active_ctx)
    if staged is not None:
        return staged

    raise TypeError("expected a State id ref for an op subject")


def _state_subject_ref(owner: object, form: object, method: str | None = None, *, ctx: "Context | None" = None) -> str:
    subject = _state_subject(owner, form, ctx=ctx)
    return f"{subject}/{method}" if method else subject


def _state_get(
    owner: object,
    form: object,
    method: str | None = None,
    key: "Scalar | Value | object" = None,
    *,
    rtype: type["Scalar"] | None = None,
    ctx: "Context | None" = None,
) -> "Scalar":
    cls = rtype or type(owner)
    owner_ctx = _owner_ctx(owner, ctx)
    subject = _state_subject_ref(owner, form, method, ctx=owner_ctx)
    return cls._get_ref(subject, key, ctx=owner_ctx)


def _state_put(
    owner: object,
    form: object,
    value: "Scalar | Value | object",
    method: str | None = None,
    key: "Scalar | Value | object" = None,
    *,
    rtype: type["Scalar"] | None = None,
    ctx: "Context | None" = None,
) -> "Scalar":
    cls = rtype or type(owner)
    owner_ctx = _owner_ctx(owner, ctx)
    subject = _state_subject_ref(owner, form, method, ctx=owner_ctx)
    return cls._put_ref(subject, key, value, ctx=owner_ctx)


def _state_post(
    owner: object,
    form: object,
    method: str | None = None,
    params: Mapping[str, "Scalar | Value | object"] | None = None,
    *,
    rtype: type["Scalar"] | None = None,
    ctx: "Context | None" = None,
) -> "Scalar":
    cls = rtype or type(owner)
    owner_ctx = _owner_ctx(owner, ctx)
    subject = _state_subject_ref(owner, form, method, ctx=owner_ctx)
    return cls._post_ref(subject, params, ctx=owner_ctx)


def _state_delete(
    owner: object,
    form: object,
    method: str | None = None,
    key: "Scalar | Value | object" = None,
    *,
    rtype: type["Scalar"] | None = None,
    ctx: "Context | None" = None,
) -> "Scalar":
    cls = rtype or type(owner)
    owner_ctx = _owner_ctx(owner, ctx)
    subject = _state_subject_ref(owner, form, method, ctx=owner_ctx)
    return cls._delete_ref(subject, key, ctx=owner_ctx)
