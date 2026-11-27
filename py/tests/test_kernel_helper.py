from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tinychain as tc
import tinychain._local as tc_local
from tinychain.library import library_definition

_TOKEN = SimpleNamespace(
    host="/host",
    actor_id="test",
    alg="falcon512",
    public_key_b64="test",
    bearer_token="test-token",
)


def _use_kernel_handle(monkeypatch, handle: type) -> None:
    monkeypatch.setattr(tc_local, "kernel_handle", lambda: handle)
    monkeypatch.setattr(tc_local, "state_handle", lambda value: value)
    monkeypatch.setattr(
        tc_local,
        "kernel_request",
        lambda method, path, headers, body: (method, path, headers, body),
    )


def _materialized_definition(request) -> dict:
    handle = request[3]
    assert isinstance(handle, tc_local.backend().StateHandle)
    identity, definition = handle.value()
    return {identity: definition}


def test_with_library_passes_only_the_canonical_literal(tmp_path, monkeypatch):
    class FakeKernelHandle:
        calls: list[tuple[object | None, str | None, str | None]] = []
        request: tuple | None = None

        @classmethod
        def local(cls, *, token=None, data_dir=None, workspace=None):
            cls.calls.append((token, data_dir, workspace))
            return cls()

        def dispatch(self, request):
            type(self).request = request

    _use_kernel_handle(monkeypatch, FakeKernelHandle)

    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

        @tc.get
        def answer(self) -> tc.Number:
            return 42

    assert isinstance(tc.kernel.with_library(A(), data_dir=tmp_path, token=_TOKEN), FakeKernelHandle)
    token, data_dir, workspace = FakeKernelHandle.calls.pop()
    literal = library_definition(A())
    assert _materialized_definition(FakeKernelHandle.request) == literal
    assert token is _TOKEN
    assert data_dir == str(tmp_path)
    assert workspace == str(tmp_path.with_name(f"{tmp_path.name}-workspace"))


def test_with_library_compiles_authority_into_dependency_reference(tmp_path, monkeypatch):
    class FakeKernelHandle:
        definition: dict | None = None

        @classmethod
        def local(cls, **_kwargs):
            return cls()

        def dispatch(self, request):
            type(self).definition = _materialized_definition(request)

    _use_kernel_handle(monkeypatch, FakeKernelHandle)

    class Remote(tc.Library):
        publisher = "example-devco"
        resource_name = "remote"
        version = "0.1.0"

        @tc.get
        def ping(self, name: str) -> tc.String:
            ...

    class Local(tc.Library):
        publisher = "example-devco"
        resource_name = "local"
        version = "0.1.0"

        remote = Remote(authority=tc.URI.parse("https://api.example.test"))

        @tc.get
        def ping(self, name: str) -> tc.String:
            return self.remote.ping(name=name)

    tc.kernel.with_library(Local(), data_dir=tmp_path, token=_TOKEN)
    encoded = json.dumps(FakeKernelHandle.definition, separators=(",", ":"))
    assert "https://api.example.test/lib/example-devco/remote/0.1.0/ping" in encoded


def test_with_library_installs_classes_before_linking_the_library(tmp_path, monkeypatch):
    class FakeKernelHandle:
        requests: list[tuple] = []

        @classmethod
        def local(cls, **_kwargs):
            return cls()

        def dispatch(self, request):
            type(self).requests.append(request)

    _use_kernel_handle(monkeypatch, FakeKernelHandle)

    class Point(tc.Class, tc.Map):
        publisher = "example-devco"
        resource_name = "point"
        version = "1.0.0"

    class Geometry(tc.Library):
        publisher = "example-devco"
        resource_name = "geometry"
        version = "1.0.0"
        classes = (Point,)

    tc.kernel.with_library(Geometry(), data_dir=tmp_path, token=_TOKEN)

    assert [request[1] for request in FakeKernelHandle.requests] == ["/class", "/lib"]
    library_put = _materialized_definition(FakeKernelHandle.requests[-1])
    assert next(iter(library_put.values()))["point"] == Point.class_id().path


def test_with_library_rejects_dependency_override_argument(tmp_path):
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

    with pytest.raises(TypeError, match="dependency"):
        tc.kernel.with_library(
            A(), data_dir=tmp_path, token=_TOKEN, dependency=tc.URI("lib")
        )


def test_for_library_alias_is_removed():
    assert not hasattr(tc.kernel, "for_library")
