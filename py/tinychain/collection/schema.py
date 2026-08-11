from __future__ import annotations

from ..state.value import Value
from ..uri import URI, uri


class Column:
    """A named collection column with a canonical TinyChain value type."""

    def __init__(self, name: str, dtype: object):
        if not isinstance(name, str) or not name:
            raise TypeError("column name must be a non-empty string")

        if isinstance(dtype, type) and issubclass(dtype, Value):
            dtype = next(
                (value_type for value_type in Value.__subclasses__() if issubclass(dtype, value_type)),
                dtype,
            )

        dtype_uri = URI.parse(dtype) if isinstance(dtype, str) else uri(dtype)
        if not isinstance(dtype_uri, URI):
            raise TypeError("column dtype must have a canonical URI")

        self.name = name
        self.dtype = dtype_uri

    def to_json(self) -> list[str]:
        return [self.name, self.dtype.canonical()]
