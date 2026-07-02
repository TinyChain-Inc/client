from __future__ import annotations

import dataclasses


def _serialize(obj: object) -> object:
    """Recursively convert autodiff types to JSON-serializable primitives.
    
    Handles:
    - TensorOperator instances: serialized as {"type": class_name, "route_name": ...}
    - Dataclasses: recursively serialized field-by-field
    - Lists/tuples: recursively serialized element-by-element
    - Dicts: recursively serialized key-value pairs
    - Scalars (str, int, float, bool, None): passed through unchanged
    
    Args:
        obj: The object to serialize.
    
    Returns:
        A JSON-serializable representation of obj.
    """
    # Local import to avoid circular dependency at module load time.
    # By the time _serialize() is called, both modules have finished loading.
    from .graph import TensorOperator

    if isinstance(obj, TensorOperator):
        return {"type": type(obj).__name__, "route_name": obj.route_name}
    
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _serialize(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    
    return obj
