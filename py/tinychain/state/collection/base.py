from __future__ import annotations

from ..base import State
from ...uri import URI


class Collection(State):
    """A native TinyChain collection literal, reference, or lazy view."""

    @classmethod
    def _from_payload(cls, payload: object) -> "Collection":
        return cls({str(URI(cls)): cls._normalize_payload(payload)})

    @classmethod
    def _normalize_payload(cls, payload: object) -> object:
        return payload

    @classmethod
    def _type_for(cls, path: object) -> type["Collection"] | None:
        return next(
            (collection_type for collection_type in cls.__subclasses__() if path == str(URI(collection_type))),
            None,
        )

    @classmethod
    def decode(cls, form: object) -> "Collection" | None:
        if not isinstance(form, dict) or len(form) != 1:
            return None
        (path, payload), = form.items()
        if cls is Collection:
            collection_type = cls._type_for(path)
        else:
            collection_type = cls if path == str(URI(cls)) else None
        return None if collection_type is None else collection_type._from_payload(payload)

    @classmethod
    def from_json(cls, form: object) -> "Collection":
        collection = cls.decode(form)
        if collection is not None:
            return collection

        if cls is Collection:
            raise TypeError("unknown collection payload")
        from ..scalar import Scalar, form_of

        return cls(form_of(Scalar.from_json(form)))
