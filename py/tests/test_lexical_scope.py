from __future__ import annotations

import asyncio
import gc
import weakref

import pytest
import tinychain as tc


def _scalar(value: object) -> tc.state.Scalar:
    return tc.state.Scalar.from_json(value)


def test_context_is_single_assignment_and_auto_binding_is_idempotent() -> None:
    value = tc.Number(1)
    cxt = tc.Context()
    first = cxt.bind("first", value)

    with pytest.raises(ValueError, match="visible binding"):
        cxt.bind("first", tc.Number(2))
    with pytest.raises(ValueError, match="under two names"):
        cxt.bind("second", value)

    assert cxt.bind_auto(value) is first
    generated = cxt.bind_auto(tc.Number(2), prefix="_value")
    assert tc.state.form_of(generated) == tc.state.IdRef("_value0")


def test_context_private_state_is_slotted_and_not_a_binding() -> None:
    cxt = tc.Context()

    assert not hasattr(cxt, "__dict__")
    assert cxt.form() == ()

    cxt._counter = 7
    assert cxt.form() == ()

    cxt.total = 1
    assert [name for name, _value in cxt.form()] == ["total"]


def test_context_result_is_an_immutable_snapshot() -> None:
    cxt = tc.Context()
    cxt.bind("first", 1)
    result = cxt.result(tc.state.id("first"))
    cxt.bind("second", 2)

    assert isinstance(result.form, tuple)
    assert [name for name, _ in result.form] == ["first"]


def test_context_retains_identity_keys_for_its_lifetime() -> None:
    class WeakList(list):
        pass

    value = WeakList([1])
    reference = weakref.ref(value)
    cxt = tc.Context()
    cxt.bind("value", value)
    del value
    gc.collect()

    assert reference() is not None


def test_nested_context_captures_but_cannot_shadow() -> None:
    with tc.scoped_context() as outer:
        outer.bind("captured", 1)
        with tc.scoped_context() as inner:
            assert inner.captured is outer.captured
            with pytest.raises(ValueError, match="visible binding"):
                inner.bind("captured", 2)


def test_scope_restores_after_exception_and_cancellation() -> None:
    from tinychain.context import _active_context

    with pytest.raises(RuntimeError):
        with tc.scoped_context():
            raise RuntimeError("stop")
    assert _active_context() is None

    with pytest.raises(asyncio.CancelledError):
        with tc.scoped_context():
            raise asyncio.CancelledError()
    assert _active_context() is None


def test_opdef_rejects_duplicate_parameters_and_recursive_shadowing() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        tc.state.PutOpDef("value", "value", [])

    nested = tc.state.PostOpDef([("outer", _scalar(2))])
    with pytest.raises(ValueError, match="shadows"):
        tc.state.PostOpDef([("outer", _scalar(1)), ("result", tc.state.Scalar(nested))])


def test_opdef_form_is_immutable() -> None:
    op = tc.state.PostOpDef([("result", 1)])
    assert isinstance(op.form, tuple)
    with pytest.raises(AttributeError):
        op.form.append(("other", _scalar(2)))


def test_control_binders_are_lexical() -> None:
    body = tc.state.PostOpDef([("result", tc.state.id("item"))])
    loop = tc.state.for_each([], item_name="item", op=body)
    required: set[str] = set()
    tc.state.PostOpDef([("outer", loop)]).requires(required)
    assert required == set()

    with pytest.raises(ValueError, match="ForEach.*shadows"):
        tc.state.PostOpDef([("item", _scalar(1)), ("result", loop)])

    callback = tc.state.PostOpDef([("result", tc.state.id("state"))])
    while_ref = tc.state.while_loop(callback, callback, tc.Number(0))
    required = set()
    tc.state.PostOpDef([("outer", while_ref)]).requires(required)
    assert required == set()

    with pytest.raises(ValueError, match="While.*shadows"):
        tc.state.PostOpDef([("state", _scalar(1)), ("result", while_ref)])


@pytest.mark.parametrize("name", ["", "bad name", "bad/name", "self"])
def test_invalid_or_reserved_binding_names_are_rejected(name: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        tc.state.PostOpDef([(name, _scalar(1))])


def test_requires_reports_only_unbound_lexical_inputs() -> None:
    op = tc.state.GetOpDef(
        "key",
        [
            ("local", tc.state.id("input")),
            ("result", tc.state.id("local")),
        ],
    )
    required: set[str] = set()
    op.requires(required)
    assert required == {"input"}


def test_context_is_owned_by_the_top_level_authoring_api() -> None:
    assert not hasattr(tc.state, "Context")
    assert not hasattr(tc.state, "scoped_context")


def test_route_compilation_rejects_undefined_lexical_inputs() -> None:
    class Invalid(tc.Library):
        publisher = "example-devco"
        resource_name = "invalid"
        version = "1.0.0"

        @tc.post
        def route(self, supplied: tc.Number):
            return tc.state.id("missing")

    with pytest.raises(ValueError, match=r"undefined input.*\$missing"):
        tc.library.compile_ir(Invalid)


def test_route_mapping_keys_do_not_enter_lexical_scope() -> None:
    class Valid(tc.Library):
        publisher = "example-devco"
        resource_name = "valid"
        version = "1.0.0"

        @tc.post
        def route(self, supplied: tc.Number):
            return {"supplied": 1}

    definition = tc.library.compile_ir(Valid)
    route = definition[Valid.class_id().path]["route"]
    form = next(iter(route.values()))
    assert form == [["result", {"supplied": 1}]]


def test_symbolic_reducer_requires_an_explicit_item_name() -> None:
    with tc.scoped_context() as cxt:
        reducer = cxt.bind("reducer", tc.state.PostOpDef([("result", 1)]))
        items = tc.state.autobox([1])
        cxt.bind("items", items)

        with pytest.raises(TypeError, match="concrete OpDef"):
            items.reduce(op=reducer, value={})

        reduced = items.reduce(op=reducer, value={}, item_name="item")

    (_subject, params), = reduced.to_json().items()
    assert params["item_name"] == "item"
