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
    defined_ids = {name for name, _ in resolved_op.form}
    referenced_ids: set[str] = set()
    subject_ids: set[str] = set()
    for _, scalar in resolved_op.form:
        _collect_ref_ids_from_form(form_of(scalar), referenced_ids, subject_ids)

    subject_candidates = sorted(
        name for name in subject_ids if name not in defined_ids and name not in state_keys
    )
    if len(subject_candidates) == 1:
        return subject_candidates[0]
    if len(subject_candidates) > 1:
        raise TypeError(
            "reduce item binding is ambiguous; reducer references multiple subject inputs: "
            + ", ".join(subject_candidates)
        )

    external_ids = sorted(name for name in referenced_ids if name not in defined_ids)
    candidates = [name for name in external_ids if name not in state_keys]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise TypeError(
            "reduce could not infer item binding: reducer must reference exactly one external input not present in state"
        )

    raise TypeError(
        "reduce item binding is ambiguous; reducer references multiple external inputs: "
        + ", ".join(candidates)
    )


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

    op_form = form_of(op)
    if isinstance(op_form, OpDef):
        return op_form

    if not isinstance(op_form, TCRef):
        return None
    op_ref_form = form_of(op_form)
    if not isinstance(op_ref_form, IdRef):
        return None

    from ..context import current_context

    active_ctx = current_context()
    if active_ctx is None:
        return None

    target = op_ref_form.name
    for name, scalar in active_ctx.form():
        if name != target:
            continue
        scalar_form = form_of(scalar)
        if isinstance(scalar_form, OpDef):
            return scalar_form
        return _resolve_reduce_opdef(scalar)

    return None


def _record_subject_token(subject: str, out: set[str], subject_ids: set[str]) -> None:
    if not subject.startswith("$"):
        return

    head, sep, _tail = subject[1:].partition("/")
    if not head:
        return

    out.add(head)
    if sep:
        subject_ids.add(head)


def _collect_ref_ids_from_form(node: object, out: set[str], subject_ids: set[str]) -> None:
    from . import OpDef, Scalar, form_of

    if isinstance(node, OpRef):
        _record_subject_token(node.subject, out, subject_ids)
        _collect_ref_ids_from_form(node.args, out, subject_ids)
        return

    if isinstance(node, IdRef):
        out.add(node.name)
        return

    if isinstance(node, TCRef):
        ref_form = form_of(node)
        if ref_form is node:
            return
        _collect_ref_ids_from_form(ref_form, out, subject_ids)
        return

    if isinstance(node, Scalar):
        _collect_ref_ids_from_form(form_of(node), out, subject_ids)
        return

    runtime_op = OpRef.from_runtime(node)
    if runtime_op is not None:
        _collect_ref_ids_from_form(runtime_op, out, subject_ids)
        return

    if isinstance(node, OpDef):
        for _name, scalar in node.form:
            _collect_ref_ids_from_form(form_of(scalar), out, subject_ids)
        return

    if isinstance(node, Mapping):
        for key, value in node.items():
            if isinstance(key, str):
                _record_subject_token(key, out, subject_ids)
            _collect_ref_ids_from_form(value, out, subject_ids)
        return

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for item in node:
            _collect_ref_ids_from_form(item, out, subject_ids)
