from __future__ import annotations

from collections.abc import Mapping, Sequence

from .scalar.refs import (
    ControlRef,
    IdRef,
    OpRef,
    TCRef,
)


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


def _stage_subject(owner: object) -> str | None:
    from ..context import _bind_subject

    return _bind_subject(owner)


def _subject_from_candidate(candidate: object, owner: object) -> str | None:
    if not isinstance(candidate, (OpRef, ControlRef)):
        return None
    staged = _stage_subject(owner)
    if staged is None:
        raise TypeError("an intermediate operation requires a lexical Context; use scoped_context() or bind its result")
    return staged


def _state_subject(owner: object, form: object) -> str:
    from .scalar import form_of
    from .value import Link

    if isinstance(form, Link):
        return str(form_of(form))

    subject = _subject_from_candidate(form, owner)
    if subject is not None:
        return subject

    if isinstance(form, TCRef):
        ref_form = form_of(form)
        if isinstance(ref_form, IdRef):
            return ref_form.key()

        subject = _subject_from_candidate(ref_form, owner)
        if subject is not None:
            return subject

    if isinstance(form, Mapping) and len(form) == 1:
        (key, _value), = form.items()
        if isinstance(key, str) and key.startswith("$"):
            return key

    if isinstance(form, Sequence) and not isinstance(form, (str, bytes, bytearray)):
        staged = _stage_subject(owner)
        if staged is not None:
            return staged

    staged = _stage_subject(owner)
    if staged is not None:
        return staged

    raise TypeError("expected a State id ref for an op subject")


def _state_subject_ref(owner: object, form: object, method: str | None = None) -> str:
    subject = _state_subject(owner, form)
    return f"{subject}/{method}" if method else subject


def subject_of(owner: object) -> str:
    from .scalar import form_of

    return _state_subject(owner, form_of(owner))


def _state_get(
    owner: object,
    form: object,
    method: str | None = None,
    key: object = None,
    *,
    rtype: type | None = None,
) -> object:
    cls = rtype or type(owner)
    subject = _state_subject_ref(owner, form, method)
    return cls._get_ref(subject, key)


def _state_put(
    owner: object,
    form: object,
    value: object,
    method: str | None = None,
    key: object = None,
    *,
    rtype: type | None = None,
) -> object:
    cls = rtype or type(owner)
    subject = _state_subject_ref(owner, form, method)
    return cls._put_ref(subject, key, value)


def _state_post(
    owner: object,
    form: object,
    method: str | None = None,
    params: Mapping[str, object] | None = None,
    *,
    rtype: type | None = None,
) -> object:
    cls = rtype or type(owner)
    subject = _state_subject_ref(owner, form, method)
    return cls._post_ref(subject, params)


def _state_delete(
    owner: object,
    form: object,
    method: str | None = None,
    key: object = None,
    *,
    rtype: type | None = None,
) -> object:
    cls = rtype or type(owner)
    subject = _state_subject_ref(owner, form, method)
    return cls._delete_ref(subject, key)
