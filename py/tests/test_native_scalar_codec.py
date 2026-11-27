import json

import pytest

import tinychain as tc
from tinychain import _native
from tinychain.codec import decode_payload


@pytest.mark.parametrize("text", [
    "https://example.com/resource", "/state/scalar/value/tuple",
])
def test_native_wire_links_and_explicit_strings_round_trip(text):
    decoded = tc.state.Value.from_json(text)
    assert isinstance(decoded, tc.Link)
    assert decoded.to_json() == tc.Link(text).to_json()
    string = tc.String(text)
    wire = string.to_json()
    assert wire == {str(tc.URI(tc.String)): text}
    restored = tc.state.Value.from_json(wire)
    assert isinstance(restored, tc.String)
    assert tc.state.form_of(restored) == text
    assert tc.state.autobox(text).to_json() == wire
    assert decode_payload(wire) == text


def test_native_wire_retains_non_link_strings():
    text = "plain"
    assert isinstance(tc.state.Value.from_json(text), tc.String)
    assert tc.String(text).to_json() == text


def test_malformed_explicit_tags_are_errors_in_every_entrypoint():
    payload = {str(tc.URI(tc.String)): 17}
    for decoder in [tc.state.Value.from_json, tc.state.Scalar.from_json, decode_payload]:
        with pytest.raises(ValueError):
            decoder(payload)


@pytest.mark.parametrize("payload", [
    {str(tc.URI(tc.Null)): 7},
    {str(tc.URI(tc.String)): "value", "ignored": []},
])
def test_native_readers_reject_malformed_singleton_records(payload):
    for decoder in [tc.state.Value.from_json, tc.state.Scalar.from_json, decode_payload]:
        with pytest.raises(ValueError):
            decoder(payload)


def test_explicit_link_validation_is_owned_by_rust():
    text = "ftp://example.com/x"
    with pytest.raises(ValueError):
        tc.Link(text)


def test_native_invocations_do_not_collide_with_typed_state_paths():
    ref = tc.state.GetOpRef(tc.URI(tc.String), tc.String("https://example.com/"))
    wire = ref.to_json()
    assert set(wire) == {str(tc.URI(tc.state.OpRef, "get"))}
    restored = tc.state.TCRef.from_json(wire)
    assert isinstance(restored, tc.state.GetOpRef)
    assert restored.to_json() == wire
    hosted = {"https://example.com/run": {"key": 7}}
    restored = tc.state.TCRef.from_json(hosted)
    assert isinstance(restored, tc.state.PostOpRef)
    assert restored.to_json() == hosted
    assert isinstance(decode_payload(hosted), tc.state.PostOpRef)


def test_native_bytes_and_variant_preserving_tuples():
    from tinychain.state.value import Bytes

    value = Bytes(b"\0\xff")
    restored = tc.state.Value.from_json(value.to_json())
    assert tc.state.form_of(restored) == b"\0\xff"
    explicit = tc.state.Scalar(tc.Tuple([tc.Link("/state/scalar/value/tuple")]))
    restored = tc.state.Scalar.from_json(explicit.to_json())
    assert isinstance(tc.state.form_of(restored), tc.Tuple)
    assert restored.to_json() == explicit.to_json()


def test_native_transport_encodes_once_without_python_json(monkeypatch):
    from tinychain.host import _encode_body

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected intermediate serialization")

    monkeypatch.setattr(json, "dumps", forbidden)
    wire = _encode_body({"link": tc.Link("https://example.com/x"), "literal": tc.String("/x")})
    assert json.loads(wire) == {"link": "https://example.com/x", "literal": {str(tc.URI(tc.String)): "/x"}}


def test_native_materialization_bounds_and_cycles():
    codec = _native.Codec(max_bytes=32)
    with pytest.raises(ValueError):
        codec.scalar_from_wire([None] * 100)
    cycle = []
    cycle.append(cycle)
    for method in [_native.Codec().scalar_from_wire, _native.Codec().encode_wire]:
        with pytest.raises(ValueError):
            method(cycle)
    with pytest.raises(ValueError):
        _native.Codec().to_wire(_native.ScalarHandle.literal(float("nan")))


def test_native_builders_enforce_scalar_depth_before_publication():
    value = _native.ScalarHandle.literal(None)
    for _ in range(16):
        value = _native.ScalarHandle.tuple([value])
    assert _native.Codec().scalar_from_wire(_native.Codec().to_wire(value)) is not None
    with pytest.raises(ValueError):
        _native.ScalarHandle.tuple([value])


def test_response_projection_delegates_nested_collection_payloads():
    path = str(tc.URI(tc.collection.BTree))
    payload = {path: [[["key", str(tc.URI(tc.String))]], [["value"]]]}
    response = decode_payload({"items": [payload]})
    collection = response["items"][0]
    assert isinstance(collection, tc.collection.BTree)
    assert collection.to_json() == payload
