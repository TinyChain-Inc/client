from __future__ import annotations

from ..state.collection import Collection
from ..state.scalar import Bool, Number, Scalar, Tuple, autobox
from ..uri import URI


def _normalize_schema(schema: object) -> list[list[object]]:
    if not isinstance(schema, list):
        raise TypeError("BTree schema must be a list of column definitions")

    normalized: list[list[object]] = []
    for column in schema:
        if not isinstance(column, (list, tuple)):
            raise TypeError("BTree schema columns must be tuple/list entries")
        if len(column) not in (2, 3):
            raise TypeError("BTree schema column entries must have 2 or 3 elements")

        normalized.append(list(column))

    return normalized


def _normalize_rows(rows: object) -> list[object]:
    if not isinstance(rows, list):
        raise TypeError("BTree rows payload must be a list")
    return list(rows)


class BTree(Collection):
    """TinyChain BTree native type wrapper."""

    __uri__: URI = URI(Collection, "collection", "btree")

    def __init__(
        self,
        form: object = None,
        rows: object = None,
    ):
        if _looks_like_schema(form):
            schema = _normalize_schema(form)
            normalized_rows = _normalize_rows([] if rows is None else rows)

            super().__init__({str(URI(BTree)): [schema, normalized_rows]})
            return

        super().__init__(form)

    @classmethod
    def _normalize_payload(cls, payload: object) -> object:
        if not isinstance(payload, list) or len(payload) != 2:
            raise TypeError("BTree payload must be [schema, rows]")
        return [_normalize_schema(payload[0]), _normalize_rows(payload[1])]

    def contains(self, row: object) -> Bool:
        return self._get("contains", autobox(row), rtype=Bool)

    def count(self, key: object = None) -> Number:
        return self._get("count", autobox(key), rtype=Number)

    def is_empty(self, key: object = None) -> Bool:
        return self._get("is_empty", autobox(key), rtype=Bool)

    def insert(self, row: object) -> Tuple:
        return self._post("insert", {"row": autobox(row)}, rtype=Tuple)

    def delete(self, row: object) -> Scalar:
        return self._delete(key=autobox(row), rtype=Scalar)

    def slice(
        self,
        start: object = None,
        end: object = None,
        *,
        reverse: bool = False,
    ) -> "BTree":
        key = {
            "start": autobox(start),
            "end": autobox(end),
            "reverse": autobox(reverse),
        }
        return self._get("slice", key, rtype=BTree)


def _looks_like_schema(form: object) -> bool:
    if not isinstance(form, list) or len(form) == 0:
        return False

    return all(isinstance(column, (list, tuple)) for column in form)
