from __future__ import annotations

from collections.abc import Sequence
import json
from typing import TYPE_CHECKING

from ... import _local
from ...executor import try_current
from ...uri import path, uri
from ..scalar import Bool, Number, Scalar, TCRef, Tuple, autobox, form_of
from ..value import Null
from .base import Collection

if TYPE_CHECKING:
    from ..context import Context


BTREE_PATH = path("state", "collection", "btree")


def _normalize_schema(schema: object) -> list[list[object]]:
    if not isinstance(schema, list):
        raise TypeError("BTree schema must be a list of column definitions")

    normalized: list[list[object]] = []
    for column in schema:
        if not isinstance(column, (list, tuple)):
            raise TypeError("BTree schema columns must be tuple/list entries")

        if len(column) not in (2, 3):
            raise TypeError("BTree schema column entries must have 2 or 3 elements")

        name = column[0]
        dtype = column[1]
        if not isinstance(name, str):
            raise TypeError("BTree schema column name must be a string")
        if not isinstance(dtype, str):
            raise TypeError("BTree schema column dtype must be a string URI")

        if len(column) == 3:
            normalized.append([name, dtype, column[2]])
        else:
            normalized.append([name, dtype])

    return normalized


def _normalize_rows(rows: object) -> list[object]:
    if not isinstance(rows, list):
        raise TypeError("BTree rows payload must be a list")
    return list(rows)


class BTree(Collection):
    """TinyChain BTree native type wrapper."""

    __uri__ = uri(BTREE_PATH)

    def __init__(
        self,
        form: object = None,
        *,
        ref: TCRef | None = None,
        rows: object = None,
        native: object = None,
        ctx: "Context | None" = None,
    ):
        if native is not None and (form is not None or ref is not None):
            raise TypeError("BTree accepts either native or symbolic form/ref")

        if native is not None:
            super().__init__(None, ctx=ctx)
            self._native = native
            return

        if ref is None and _looks_like_schema(form):
            schema = _normalize_schema(form)
            normalized_rows = _normalize_rows(rows or [])

            if _is_imperative_mode(ctx=ctx):
                super().__init__(None, ctx=ctx)
                self._native = _require_local_btree(schema, normalized_rows)
                return

            super().__init__({BTREE_PATH: [schema, normalized_rows]}, ctx=ctx)
            self._native = None
            return

        super().__init__(form, ref=ref, ctx=ctx)
        self._native = None

    @classmethod
    def from_json(cls, form: object) -> "BTree":
        if not isinstance(form, dict):
            raise TypeError("BTree.from_json expects a map-shaped BTree payload")

        if BTREE_PATH in form:
            payload = form[BTREE_PATH]
            if not isinstance(payload, list):
                raise TypeError("BTree payload must be a list")

            if len(payload) != 2:
                raise TypeError("BTree payload must be [schema, rows]")

            schema = _normalize_schema(payload[0])
            rows = _normalize_rows(payload[1])
            return cls({BTREE_PATH: [schema, rows]})

        # Symbolic BTree refs/oprefs are map-shaped TCRef forms.
        if len(form) == 1:
            return cls(form=form_of(Scalar.from_json(form)))

        raise TypeError("BTree.from_json expects {uri: [keys]} or a symbolic ref map")

    def to_json(self) -> object:
        if self._native is not None:
            return self._native.to_json()

        form = self._form
        if isinstance(form, TCRef):
            return Scalar(ref=form).to_json()

        if isinstance(form, dict):
            if BTREE_PATH in form:
                payload = form[BTREE_PATH]
                if not isinstance(payload, list):
                    raise TypeError("BTree payload must be a list")

                if len(payload) != 2:
                    raise TypeError("BTree payload must be [schema, rows]")

                schema = _normalize_schema(payload[0])
                rows = _normalize_rows(payload[1])
                return {BTREE_PATH: [schema, rows]}

            return Scalar(form).to_json()

        return Scalar(form).to_json()

    def contains(self, row: object, *, ctx: "Context | None" = None) -> Bool:
        if self._native is not None:
            return self._native.contains(_native_row(row, _native_key_arity(self.to_json())))
        return self._get("contains", autobox(row), rtype=Bool, ctx=ctx)

    def count(self, key: object = None, *, ctx: "Context | None" = None) -> Number:
        if self._native is not None:
            rows = _native_rows(self.to_json())
            if key is None:
                return len(rows)

            start, end, _ = _slice_bounds(key)
            return len(_slice_rows(rows, start, end, reverse=False))
        return self._get("count", autobox(key), rtype=Number, ctx=ctx)

    def is_empty(self, key: object = None, *, ctx: "Context | None" = None) -> Bool:
        if self._native is not None:
            return self.count(key) == 0
        return self._get("is_empty", autobox(key), rtype=Bool, ctx=ctx)

    def insert(self, row: object, *, ctx: "Context | None" = None) -> Tuple:
        if self._native is not None:
            self._native.insert(_native_row(row, _native_key_arity(self.to_json())))
            return None
        return self._post("insert", {"row": autobox(row)}, rtype=Tuple, ctx=ctx)

    def delete(self, row: object, *, ctx: "Context | None" = None) -> Null:
        if self._native is not None:
            self._native.delete(_native_row(row, _native_key_arity(self.to_json())))
            return None
        return self._delete(key=autobox(row), rtype=Null, ctx=ctx)

    def slice(
        self,
        start: object = None,
        end: object = None,
        *,
        reverse: bool = False,
        ctx: "Context | None" = None,
    ) -> "BTree":
        if self._native is not None:
            payload = self.to_json()
            schema = _native_schema(payload)
            rows = _native_rows(payload)
            sliced = _slice_rows(rows, start, end, reverse)
            return BTree(schema, rows=sliced)

        key = {
            "start": autobox(start),
            "end": autobox(end),
            "reverse": autobox(reverse),
        }
        return self._get("slice", key, rtype=BTree, ctx=ctx)


def _looks_like_schema(form: object) -> bool:
    if not isinstance(form, list) or len(form) == 0:
        return False

    return all(isinstance(column, (list, tuple)) for column in form)


def _is_imperative_mode(*, ctx: "Context | None" = None) -> bool:
    if ctx is not None:
        return False

    executor = try_current()
    if executor is not None and not executor.is_eager():
        return False

    return True


def _require_local_btree(schema: list[list[object]], rows: list[object]):
    try:
        return _new_local_btree(schema, rows)
    except ImportError as exc:
        raise ImportError(
            "BTree construction in imperative mode requires `tinychain-local`; "
            "install it or construct inside a deferred planning context"
        ) from exc


def _new_local_btree(schema: list[list[object]], rows: list[object]):
    local = _local.backend()
    local_schema = [(column[0], column[1]) for column in schema]
    arity = len(local_schema)
    local_rows = [_native_row(row, arity) for row in rows]
    return local.BTree(local_schema, rows=local_rows)


def _native_key_arity(payload: object) -> int:
    schema = _native_schema(payload)
    return len(schema)


def _native_schema(payload: object) -> list[list[object]]:
    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict) or BTREE_PATH not in payload:
        raise TypeError("native BTree payload must be map-shaped")

    raw = _native_btree_payload(payload[BTREE_PATH])
    if not isinstance(raw, list) or len(raw) != 2:
        raise TypeError("native BTree payload must be [schema, rows]")

    return _normalize_schema(raw[0])


def _native_rows(payload: object) -> list[list[object]]:
    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict) or BTREE_PATH not in payload:
        raise TypeError("native BTree payload must be map-shaped")

    raw = _native_btree_payload(payload[BTREE_PATH])
    if not isinstance(raw, list) or len(raw) != 2:
        raise TypeError("native BTree payload must be [schema, rows]")

    schema = _normalize_schema(raw[0])
    rows = _normalize_rows(raw[1])
    arity = len(schema)
    return [_native_row(row, arity) for row in rows]


def _native_row(row: object, arity: int) -> list[object]:
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        values = list(row)
    else:
        values = [row]

    if arity != len(values):
        raise ValueError(f"BTree row arity {len(values)} does not match schema arity {arity}")

    return values


def _native_btree_payload(raw: object) -> object:
    # tinychain_local currently returns {uri: [[schema, rows]]}; normalize that
    # to the canonical {uri: [schema, rows]} form expected by the wrapper.
    if (
        isinstance(raw, list)
        and len(raw) == 1
        and isinstance(raw[0], list)
        and len(raw[0]) == 2
    ):
        return raw[0]

    return raw


def _slice_bounds(bounds: object) -> tuple[object, object, bool]:
    if bounds is None:
        return None, None, False

    if isinstance(bounds, dict):
        return bounds.get("start"), bounds.get("end"), bool(bounds.get("reverse", False))

    raise TypeError("BTree slice/count/is_empty bounds must be a dict with start/end/reverse")


def _slice_rows(rows: list[list[object]], start: object, end: object, reverse: bool) -> list[list[object]]:
    def in_range(row: list[object]) -> bool:
        key = tuple(row)
        if start is not None:
            start_key = tuple(_coerce_bound(start, len(row)))
            if key < start_key:
                return False
        if end is not None:
            end_key = tuple(_coerce_bound(end, len(row)))
            if key >= end_key:
                return False
        return True

    sliced = [row for row in rows if in_range(row)]
    if reverse:
        sliced.reverse()
    return sliced


def _coerce_bound(bound: object, arity: int) -> list[object]:
    return _native_row(bound, arity)
