from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from .refs import IdRef, OpRef, TCRef

if TYPE_CHECKING:
    from . import OpDef, Scalar


def infer_reduce_item_name(
    op: "OpDef | Scalar | object",
    value: "Scalar | object",
) -> str:
    from . import form_of

    resolved_op = _resolve_reduce_opdef(op)
    if resolved_op is None:
        raise TypeError("reduce requires a concrete OpDef to infer item binding")

    state_keys = _reduce_state_keys(value)
    referenced_ids: set[str] = set()
    resolved_op.requires(referenced_ids)
    subject_ids: set[str] = set()
    for _, scalar in resolved_op.form:
        _collect_subject_ids(form_of(scalar), subject_ids)

    subject_candidates = sorted(
        name
        for name in subject_ids
        if name in referenced_ids
        and name not in state_keys
    )
    if len(subject_candidates) == 1:
        return subject_candidates[0]
    if len(subject_candidates) > 1:
        raise TypeError(
            "reduce item binding is ambiguous; reducer references multiple subject inputs: "
            + ", ".join(subject_candidates)
        )

    # A reducer which ignores its item still needs a collision-free callback
    # binder. Captures remain ordinary unresolved requirements and are never
    # reverse-resolved through an ambient authoring Context.
    used = referenced_ids | state_keys | {name for name, _ in resolved_op.form}
    index = 0
    while f"_item{index}" in used:
        index += 1
    return f"_item{index}"


def _reduce_state_keys(value: "Scalar | object") -> set[str]:
    from . import Scalar, form_of
    from ..value import Map as value_map
    from ..value import Value
    from ..value import form_of as value_form_of

    if isinstance(value, Scalar):
        value_form = form_of(value)
        if isinstance(value_form, Mapping):
            return set(value_form.keys())
        if isinstance(value_form, value_map):
            map_form = value_form_of(value_form)
            if isinstance(map_form, dict):
                return set(map_form.keys())
        return set()

    if isinstance(value, Value):
        if isinstance(value, value_map):
            map_form = value_form_of(value)
            if isinstance(map_form, dict):
                return set(map_form.keys())
        return set()

    if isinstance(value, Mapping):
        keys: set[str] = set()
        for key in value.keys():
            if not isinstance(key, str):
                raise TypeError("reduce state map keys must be strings")
            keys.add(key)
        return keys

    return set()


def _resolve_reduce_opdef(op: "OpDef | Scalar | object") -> "OpDef | None":
    from . import OpDef, Scalar, form_of

    if isinstance(op, OpDef):
        return op

    if not isinstance(op, Scalar):
        return None

    node: object = op
    seen: set[int] = set()
    while isinstance(node, Scalar) and id(node) not in seen:
        seen.add(id(node))
        node = form_of(node)
        if isinstance(node, OpDef):
            return node

    return None


def _record_subject_token(subject: str, subject_ids: set[str]) -> None:
    if not subject.startswith("$"):
        return

    head, sep, _tail = subject[1:].partition("/")
    if not head:
        return

    if sep:
        subject_ids.add(head)


def _collect_subject_ids(node: object, subject_ids: set[str]) -> None:
    from . import OpDef, Scalar, form_of

    if isinstance(node, OpRef):
        _record_subject_token(node.subject, subject_ids)
        _collect_subject_ids(node.args, subject_ids)
        return

    if isinstance(node, IdRef):
        return

    if isinstance(node, TCRef):
        ref_form = form_of(node)
        if ref_form is node:
            return
        _collect_subject_ids(ref_form, subject_ids)
        return

    if isinstance(node, Scalar):
        _collect_subject_ids(form_of(node), subject_ids)
        return

    if isinstance(node, OpDef):
        for _name, scalar in node.form:
            _collect_subject_ids(form_of(scalar), subject_ids)
        return

    if isinstance(node, Mapping):
        for key, value in node.items():
            if isinstance(key, str):
                _record_subject_token(key, subject_ids)
            _collect_subject_ids(value, subject_ids)
        return

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for item in node:
            _collect_subject_ids(item, subject_ids)
