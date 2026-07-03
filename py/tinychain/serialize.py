from __future__ import annotations

import dataclasses


def serialize(obj: object) -> object:
    """Recursively convert objects to JSON-serializable primitives.

    Handles:
    - Objects with a __serialize__ hook: dispatched to the hook, then re-serialized
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
    # Check for __serialize__ hook on the type (not the instance)
    hook = getattr(type(obj), "__serialize__", None)
    if hook is not None:
        return serialize(hook(obj))

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
