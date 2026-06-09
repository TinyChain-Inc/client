from __future__ import annotations

import pytest

import tinychain as tc


def _clear_token_env(monkeypatch) -> None:
    monkeypatch.delenv("TC_TOKEN_HOST", raising=False)
    monkeypatch.delenv("TC_ACTOR_ID", raising=False)
    monkeypatch.delenv("TC_PUBLIC_KEY_B64", raising=False)


def test_with_library_infers_route_from_declared_dependency_authority(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep = tc.URI.parse("https://deps.example.test:9443/lib/example-devco/b/0.1.0")

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (dep,)

    kernel = tc.kernel.with_library(A(), data_dir=tmp_path)
    assert kernel == "kernel"
    assert len(FakeKernelHandle.calls) == 1
    assert FakeKernelHandle.calls[0][1:] == (
        [("/lib/example-devco/b/0.1.0", "deps.example.test:9443")],
        None,
        str(tmp_path),
    )


def test_with_library_rejects_dependency_override_argument(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)

    dependency = tc.URI.parse("http://deps.example.test:8702/lib/example-devco/c/0.1.0")
    with pytest.raises(TypeError, match="dependency"):
        tc.kernel.with_library(A(), data_dir=tmp_path, dependency=dependency)

    assert FakeKernelHandle.calls == []


def test_with_library_uses_multi_route_constructor_when_available(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep_b = tc.URI.parse("http://left.example.test:8702/lib/example-devco/b/0.1.0")
    dep_c = tc.URI.parse("http://right.example.test:8703/lib/example-devco/c/0.1.0")

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (dep_b, dep_c)

    tc.kernel.with_library(A(), data_dir=tmp_path)

    assert len(FakeKernelHandle.calls) == 1
    assert FakeKernelHandle.calls[0][1:] == (
        [
            ("/lib/example-devco/b/0.1.0", "left.example.test:8702"),
            ("/lib/example-devco/c/0.1.0", "right.example.test:8703"),
        ],
        None,
        str(tmp_path),
    )


def test_with_library_keeps_each_declared_dependency_route(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep_b = tc.URI.parse("http://deps.example.test:8702/lib/example-devco/b/0.1.0")
    dep_c = tc.URI.parse("http://deps.example.test:8702/lib/example-devco/c/0.1.0")

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (dep_b, dep_c)

    tc.kernel.with_library(A(), data_dir=tmp_path)

    assert FakeKernelHandle.calls[0][1] == [
        ("/lib/example-devco/b/0.1.0", "deps.example.test:8702"),
        ("/lib/example-devco/c/0.1.0", "deps.example.test:8702"),
    ]


def test_with_library_infers_authority_from_runtime_dependency_binding(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    class B(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        authority = tc.URI.parse("http://deps.example.test:8702")

    class Local(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)
        dep = B()

    tc.kernel.with_library(Local(), data_dir=tmp_path)

    assert FakeKernelHandle.calls[0][1] == [
        ("/lib/example-devco/b/0.1.0", "deps.example.test:8702")
    ]


def test_with_library_missing_authority_is_clear(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        pass

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)

    with pytest.raises(ValueError, match="missing dependency authority"):
        tc.kernel.with_library(A(), data_dir=tmp_path)


def test_with_library_rejects_ambiguous_runtime_binding_authority(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        pass

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    LeftB = type(
        "B",
        (tc.Library,),
        {
            "publisher": "example-devco",
            "version": "0.1.0",
            "authority": tc.URI.parse("http://left.example.test:8702"),
        },
    )
    RightB = type(
        "B",
        (tc.Library,),
        {
            "publisher": "example-devco",
            "version": "0.1.0",
            "authority": tc.URI.parse("http://right.example.test:8703"),
        },
    )

    class Local(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)
        left = LeftB()
        right = RightB()

    with pytest.raises(ValueError, match="ambiguous dependency authority"):
        tc.kernel.with_library(Local(), data_dir=tmp_path)


def test_for_library_alias_is_removed():
    assert not hasattr(tc.kernel, "for_library")


def test_with_library_ignores_env_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("TC_TOKEN_HOST", "https://tokens.example.test")
    monkeypatch.setenv("TC_ACTOR_ID", "example-admin")
    monkeypatch.setenv("TC_PUBLIC_KEY_B64", "pubkey")

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep = tc.URI.parse("https://deps.example.test:9443/lib/example-devco/b/0.1.0")

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (dep,)

    tc.kernel.with_library(A(), data_dir=tmp_path)
    assert len(FakeKernelHandle.calls) == 1
    assert FakeKernelHandle.calls[0][1:] == (
        [("/lib/example-devco/b/0.1.0", "deps.example.test:9443")],
        None,
        str(tmp_path),
    )


def test_with_library_accepts_single_token_object(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, list[tuple[str, str]] | None, object | None, str | None]] = []

        @classmethod
        def with_library_definition(cls, definition_json, *, routes=None, token=None, data_dir=None):
            cls.calls.append((definition_json, routes, token, data_dir))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep = tc.URI.parse("https://deps.example.test:9443/lib/example-devco/b/0.1.0")

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (dep,)

    token = tc.auth.SignedBearerToken(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        public_key_b64="pubkey",
        secret_key_b64="secret",
        bearer_token="bearer",
    )

    assert tc.kernel.with_library(A(), data_dir=tmp_path, token=token) == "kernel"
    assert len(FakeKernelHandle.calls) == 1
    assert FakeKernelHandle.calls[0][1:] == (
        [("/lib/example-devco/b/0.1.0", "deps.example.test:9443")],
        token,
        str(tmp_path),
    )
