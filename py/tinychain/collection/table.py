from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..state.collection import Collection
from ..state.scalar import Bool, Number, Scalar, Tuple, autobox
from ..uri import URI
from .bound import Range
from .schema import Column


class Schema:
    """The primary key, value columns, and auxiliary indices of a Table."""

    def __init__(
        self,
        key: Iterable[Column],
        values: Iterable[Column] = (),
        indices: Iterable[tuple[str, Iterable[str]]] = (),
    ):
        self.key = _columns(key, "key")
        if not self.key:
            raise ValueError("Table schema requires at least one key column")

        self.values = _columns(values, "value")
        self.indices: list[tuple[str, list[str]]] = []
        for name, columns in indices:
            self.create_index(name, columns)

    def columns(self) -> list[Column]:
        return [*self.key, *self.values]

    def create_index(self, name: str, columns: Iterable[str]) -> "Schema":
        if not isinstance(name, str) or not name:
            raise TypeError("Table index name must be a non-empty string")
        column_names = list(columns)
        if not column_names or not all(isinstance(column, str) for column in column_names):
            raise TypeError("Table index columns must be a non-empty sequence of names")
        self.indices.append((name, column_names))
        return self

    def to_json(self) -> list[object]:
        columns = [
            [column.to_json() for column in self.key],
            [column.to_json() for column in self.values],
        ]
        indices = [[name, list(index)] for name, index in self.indices]
        return [columns, indices]


class Table(Collection):
    """A native TinyChain Table literal, reference, or lazy view."""

    __uri__: URI = URI(Collection, "collection", "table")

    def __init__(self, form: object = None, rows: object = None):
        if isinstance(form, Schema):
            schema = form.to_json()
            normalized_rows = _rows([] if rows is None else rows)
            super().__init__({str(URI(Table)): [schema, normalized_rows]})
            return

        if rows is not None:
            raise TypeError("Table rows require a Table Schema")
        super().__init__(form)

    @classmethod
    def _normalize_payload(cls, payload: object) -> object:
        return list(_payload(payload))

    def __getitem__(self, key: object) -> Tuple:
        return self._get(key=autobox(key), rtype=Tuple)

    def contains(self, key: object = None) -> Bool:
        return self._get("contains", autobox(key), rtype=Bool)

    def columns(self) -> Tuple:
        return self._get("columns", rtype=Tuple)

    def count(self, key: object = None) -> Number:
        return self._get("count", autobox(key), rtype=Number)

    def is_empty(self, key: object = None) -> Bool:
        return self._get("is_empty", autobox(key), rtype=Bool)

    def key_columns(self) -> Tuple:
        return self._get("key_columns", rtype=Tuple)

    def key_names(self) -> Tuple:
        return self._get("key_names", rtype=Tuple)

    def limit(self, limit: object) -> "Table":
        return self._get("limit", autobox(limit), rtype=Table)

    def order_by(self, columns: object, reverse: object = False) -> "Table":
        return self._get("order", autobox((columns, reverse)), rtype=Table)

    def select(self, columns: object) -> "Table":
        return self._get("select", autobox(columns), rtype=Table)

    def where(self, **bounds: object) -> "Table":
        if not bounds:
            return self
        return self._post(params={name: _bound(bound) for name, bound in bounds.items()}, rtype=Table)

    def insert(self, key: object, values: object = ()) -> Scalar:
        return self._post(
            "insert",
            {"key": autobox(key), "values": autobox(values)},
            rtype=Scalar,
        )

    def upsert(self, key: object, values: object) -> Scalar:
        return self._put(autobox(values), key=autobox(key), rtype=Scalar)

    def update(self, **values: object) -> Scalar:
        return self._put(autobox(values), key=autobox(None), rtype=Scalar)

    def delete(self, key: object) -> Scalar:
        return self._delete(key=autobox(key), rtype=Scalar)

    def truncate(self) -> Scalar:
        return self._delete(key=autobox(None), rtype=Scalar)


def _columns(columns: Iterable[Column], label: str) -> list[Column]:
    result = list(columns)
    if not all(isinstance(column, Column) for column in result):
        raise TypeError(f"Table {label} columns must be Column instances")
    return result


def _rows(rows: object) -> list[list[object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("Table rows must be a sequence")
    result: list[list[object]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError("each Table row must be a sequence")
        result.append(list(row))
    return result


def _payload(payload: object) -> tuple[list[object], list[object]]:
    if not isinstance(payload, list) or len(payload) != 2:
        raise TypeError("Table payload must be [schema, rows]")
    schema_payload, rows = payload
    if not isinstance(schema_payload, list) or len(schema_payload) != 2:
        raise TypeError("Table schema payload must be [columns, indices]")
    columns, indices = schema_payload
    if not isinstance(columns, list) or len(columns) != 2:
        raise TypeError("Table schema columns must be [key, values]")
    if not all(isinstance(group, list) for group in columns) or not isinstance(indices, list):
        raise TypeError("Table schema indices must be a list")
    return schema_payload, _rows(rows)


def _bound(bound: object) -> object:
    return autobox(Range.from_slice(bound).to_json() if isinstance(bound, slice) else bound)
