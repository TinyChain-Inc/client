"""Canonical lexical inspection for TinyChain IR.

This module inspects immutable syntax. It does not resolve names or schedule
execution, and it never consults an authoring Context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def requires(node: object, required: set[str]) -> None:
    """Add the unresolved lexical inputs referenced by ``node``."""
    from .state.collection import Collection
    from .state.scalar import OpDef, Scalar, form_of
    from .state.scalar.refs import After, Cond, ForEach, IdRef, OpRef, TCRef, While

    if isinstance(node, OpDef):
        local: set[str] = set()
        for _, scalar in node.form:
            requires(scalar, local)
        local.difference_update(name for name, _ in node.form)
        local.difference_update(node._parameters())
        required.update(local)
    elif isinstance(node, IdRef):
        if node.name != "self":
            required.add(node.name)
    elif isinstance(node, OpRef):
        if node.subject.startswith("$"):
            name = node.subject[1:].partition("/")[0]
            if name and name != "self":
                required.add(name)
        requires(node._argument_form(), required)
    elif isinstance(node, After):
        requires(node.when, required)
        requires(node.then, required)
    elif isinstance(node, Cond):
        requires(node.cond, required)
        requires(node.then, required)
        requires(node.or_else, required)
    elif isinstance(node, While):
        callback: set[str] = set()
        requires(node.cond, callback)
        requires(node.op, callback)
        callback.discard("state")
        required.update(callback)
        requires(node.state, required)
    elif isinstance(node, ForEach):
        requires(node.items, required)
        body: set[str] = set()
        requires(node.op, body)
        body.discard(node.item_name)
        required.update(body)
    elif isinstance(node, TCRef):
        ref_form = form_of(node)
        if ref_form is not node:
            requires(ref_form, required)
    elif isinstance(node, (Collection, Scalar)):
        scalar_form = form_of(node)
        if scalar_form is not node:
            requires(scalar_form, required)
    elif isinstance(node, Mapping):
        for value in node.values():
            requires(value, required)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for value in node:
            requires(value, required)


def validate(opdef: object) -> None:
    """Validate an OpDef as an immutable, single-assignment lexical graph."""
    _validate_node(opdef, frozenset())


def bindings(node: object, names: set[str]) -> None:
    """Collect binders nested in ``node`` for compiler-generated name safety."""
    from .state.collection import Collection
    from .state.scalar import OpDef, Scalar, form_of
    from .state.scalar.refs import After, Cond, ForEach, OpRef, TCRef, While

    if isinstance(node, OpDef):
        names.update(node._parameters())
        names.update(name for name, _ in node.form)
        for _, scalar in node.form:
            bindings(scalar, names)
    elif isinstance(node, While):
        names.add("state")
        bindings(node.cond, names)
        bindings(node.op, names)
        bindings(node.state, names)
    elif isinstance(node, ForEach):
        names.add(node.item_name)
        bindings(node.items, names)
        bindings(node.op, names)
    elif isinstance(node, After):
        bindings(node.when, names)
        bindings(node.then, names)
    elif isinstance(node, Cond):
        bindings(node.cond, names)
        bindings(node.then, names)
        bindings(node.or_else, names)
    elif isinstance(node, OpRef):
        bindings(node._argument_form(), names)
    elif isinstance(node, (Collection, Scalar, TCRef)):
        scalar_form = form_of(node)
        if scalar_form is not node:
            bindings(scalar_form, names)
    elif isinstance(node, Mapping):
        for value in node.values():
            bindings(value, names)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for value in node:
            bindings(value, names)


def _validate_node(node: object, visible: frozenset[str]) -> None:
    from .state.collection import Collection
    from .state.scalar import OpDef, Scalar, form_of
    from .state.scalar.refs import After, Cond, ForEach, IdRef, OpRef, TCRef, While, _validate_id

    if isinstance(node, OpDef):
        local: set[str] = set()
        binders = (*node._parameters(), *(name for name, _ in node.form))
        for name in binders:
            _validate_id(name)
            if name == "self":
                raise ValueError("$self is reserved and cannot be bound")
            if name in local:
                raise ValueError(f"duplicate OpDef binding {name!r}")
            if name in visible:
                raise ValueError(f"OpDef binding {name!r} shadows an enclosing binding")
            local.add(name)
        scope = visible | frozenset(local)
        for _, scalar in node.form:
            _validate_node(scalar, scope)
    elif isinstance(node, IdRef):
        _validate_id(node.name)
    elif isinstance(node, OpRef):
        if node.subject.startswith("$"):
            _validate_id(node.subject[1:].partition("/")[0])
        _validate_node(node._argument_form(), visible)
    elif isinstance(node, After):
        _validate_node(node.when, visible)
        _validate_node(node.then, visible)
    elif isinstance(node, Cond):
        _validate_node(node.cond, visible)
        _validate_node(node.then, visible)
        _validate_node(node.or_else, visible)
    elif isinstance(node, While):
        _validate_node(node.state, visible)
        if "state" in visible:
            raise ValueError("While callback binding 'state' shadows an enclosing binding")
        callback = visible | frozenset(("state",))
        _validate_node(node.cond, callback)
        _validate_node(node.op, callback)
    elif isinstance(node, ForEach):
        _validate_node(node.items, visible)
        _validate_id(node.item_name)
        if node.item_name == "self":
            raise ValueError("$self is reserved and cannot be bound")
        if node.item_name in visible:
            raise ValueError(
                f"ForEach binding {node.item_name!r} shadows an enclosing binding"
            )
        _validate_node(node.op, visible | frozenset((node.item_name,)))
    elif isinstance(node, TCRef):
        ref_form = form_of(node)
        if ref_form is not node:
            _validate_node(ref_form, visible)
    elif isinstance(node, (Collection, Scalar)):
        scalar_form = form_of(node)
        if scalar_form is not node:
            _validate_node(scalar_form, visible)
    elif isinstance(node, Mapping):
        for value in node.values():
            _validate_node(value, visible)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for value in node:
            _validate_node(value, visible)
