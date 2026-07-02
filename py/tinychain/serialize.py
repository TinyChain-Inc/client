from __future__ import annotations

import dataclasses
from typing import Callable

_SERIALIZERS: dict[type, Callable[[object], object]] = {}


def register_serializer(type_: type, handler: Callable[[object], object]) -> None:
    """Register a serializer function for a specific type.

    Args:
        type_: The type to register the handler for.
        handler: A function that takes an instance of type_ and returns a
            JSON-serializable representation.
    """
    _SERIALIZERS[type_] = handler


def serialize(obj: object) -> object:
    """Recursively convert objects to JSON-serializable primitives.

    Handles:
    - Registered types: dispatched to registered handlers via exact type match
      or isinstance fallback against registered base classes
    - Dataclasses: recursively serialized field-by-field
    - Lists/tuples: recursively serialized element-by-element (tuples become
      lists — JSON has no tuple type, so this is intentional normalization,
      not data loss)
    - Dicts: recursively serialized key-value pairs
    - Scalars (str, int, float, bool, None): passed through unchanged

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable representation of obj.
    """
    # Check for exact type match in registry
    obj_type = type(obj)
    if obj_type in _SERIALIZERS:
        return _SERIALIZERS[obj_type](obj)

    # Fall back to isinstance checks against registered base classes
    for registered_type, handler in _SERIALIZERS.items():
        if isinstance(obj, registered_type):
            return handler(obj)

    # Generic cases
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: serialize(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }

    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]

    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    return obj
