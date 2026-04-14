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
        calls: list[tuple[str, str, dict]] = []

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            cls.calls.append((dependency_root, authority, kwargs))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep = tc.URI.parse("https://deps.example.test:9443/lib/example-devco/b/0.1.0")
    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(dep,),
    )

    kernel = tc.kernel.with_library(library, data_dir=tmp_path)
    assert kernel == "kernel"
    assert FakeKernelHandle.calls == [
        (
            "/lib/example-devco/b/0.1.0",
            "deps.example.test:9443",
            {
                "token_host": None,
                "actor_id": None,
                "public_key_b64": None,
                "data_dir": str(tmp_path),
            },
        )
    ]


def test_with_library_rejects_dependency_override_argument(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, str, dict]] = []

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            cls.calls.append((dependency_root, authority, kwargs))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(tc.uri("lib", "example-devco", "b", "0.1.0"),),
    )
    dependency = tc.URI.parse("http://deps.example.test:8702/lib/example-devco/c/0.1.0")
    with pytest.raises(TypeError, match="dependency"):
        tc.kernel.with_library(library, data_dir=tmp_path, dependency=dependency)

    assert FakeKernelHandle.calls == []


def test_with_library_uses_multi_route_constructor_when_available(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[list[tuple[str, str]], dict]] = []

        @classmethod
        def local_with_dependency_routes(cls, dependency_routes, **kwargs):
            cls.calls.append((dependency_routes, kwargs))
            return "kernel"

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            raise AssertionError("expected multi-route constructor")

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(
            tc.URI.parse("http://left.example.test:8702/lib/example-devco/b/0.1.0"),
            tc.URI.parse("http://right.example.test:8703/lib/example-devco/c/0.1.0"),
        ),
    )

    tc.kernel.with_library(library, data_dir=tmp_path)

    assert FakeKernelHandle.calls == [
        (
            [
                ("/lib/example-devco/b/0.1.0", "left.example.test:8702"),
                ("/lib/example-devco/c/0.1.0", "right.example.test:8703"),
            ],
            {
                "token_host": None,
                "actor_id": None,
                "public_key_b64": None,
                "data_dir": str(tmp_path),
            },
        )
    ]


def test_with_library_collapses_routes_with_shared_authority(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, str, dict]] = []

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            cls.calls.append((dependency_root, authority, kwargs))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(
            tc.URI.parse("http://deps.example.test:8702/lib/example-devco/b/0.1.0"),
            tc.URI.parse("http://deps.example.test:8702/lib/example-devco/c/0.1.0"),
        ),
    )

    tc.kernel.with_library(library, data_dir=tmp_path)

    assert FakeKernelHandle.calls[0][0] == "/lib/example-devco"
    assert FakeKernelHandle.calls[0][1] == "deps.example.test:8702"


def test_with_library_rejects_multi_authority_without_multi_route_support(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            raise AssertionError("should fail before invoking single-route constructor")

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(
            tc.URI.parse("http://left.example.test:8702/lib/example-devco/b/0.1.0"),
            tc.URI.parse("http://right.example.test:8703/lib/example-devco/c/0.1.0"),
        ),
    )

    with pytest.raises(ValueError, match="multiple dependency authorities"):
        tc.kernel.with_library(library, data_dir=tmp_path)


def test_with_library_infers_authority_from_runtime_dependency_binding(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, str, dict]] = []

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            cls.calls.append((dependency_root, authority, kwargs))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    class Dep(tc.Library):
        publisher = "example-devco"
        name = "b"
        version = "0.1.0"
        authority = tc.URI.parse("http://deps.example.test:8702")

    class Local(tc.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)
        dep = Dep()

    tc.kernel.with_library(Local(), data_dir=tmp_path)

    assert FakeKernelHandle.calls[0][0] == "/lib/example-devco/b/0.1.0"
    assert FakeKernelHandle.calls[0][1] == "deps.example.test:8702"


def test_with_library_missing_authority_is_clear(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            raise AssertionError("should fail before invoking kernel constructor")

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(tc.uri("lib", "example-devco", "b", "0.1.0"),),
    )

    with pytest.raises(ValueError, match="missing dependency authority"):
        tc.kernel.with_library(library, data_dir=tmp_path)


def test_with_library_rejects_ambiguous_runtime_binding_authority(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            raise AssertionError("should fail before invoking kernel constructor")

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    class DepA(tc.Library):
        publisher = "example-devco"
        name = "b"
        version = "0.1.0"
        authority = tc.URI.parse("http://left.example.test:8702")

    class DepB(tc.Library):
        publisher = "example-devco"
        name = "b"
        version = "0.1.0"
        authority = tc.URI.parse("http://right.example.test:8703")

    class Local(tc.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)
        left = DepA()
        right = DepB()

    with pytest.raises(ValueError, match="ambiguous dependency authority"):
        tc.kernel.with_library(Local(), data_dir=tmp_path)


def test_for_library_alias_is_removed():
    assert not hasattr(tc.kernel, "for_library")


def test_with_library_ignores_env_auth_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TC_TOKEN_HOST", "https://tokens.example.test")
    monkeypatch.setenv("TC_ACTOR_ID", "example-admin")
    monkeypatch.setenv("TC_PUBLIC_KEY_B64", "pubkey")

    class FakeKernelHandle:
        calls: list[tuple[str, str, dict]] = []

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            cls.calls.append((dependency_root, authority, kwargs))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep = tc.URI.parse("https://deps.example.test:9443/lib/example-devco/b/0.1.0")
    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(dep,),
    )

    tc.kernel.with_library(library, data_dir=tmp_path)
    assert FakeKernelHandle.calls == [
        (
            "/lib/example-devco/b/0.1.0",
            "deps.example.test:9443",
            {
                "token_host": None,
                "actor_id": None,
                "public_key_b64": None,
                "data_dir": str(tmp_path),
            },
        )
    ]


def test_with_library_accepts_single_token_object(tmp_path, monkeypatch):
    _clear_token_env(monkeypatch)

    class FakeKernelHandle:
        calls: list[tuple[str, str, dict]] = []

        @classmethod
        def local_with_dependency_route(cls, dependency_root, authority, **kwargs):
            cls.calls.append((dependency_root, authority, kwargs))
            return "kernel"

    monkeypatch.setattr(tc, "KernelHandle", FakeKernelHandle)

    dep = tc.URI.parse("https://deps.example.test:9443/lib/example-devco/b/0.1.0")
    library = tc.Library(
        publisher="example-devco",
        name="a",
        version="0.1.0",
        dependencies=(dep,),
    )
    token = tc.auth.SignedBearerToken(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        public_key_b64="pubkey",
        secret_key_b64="secret",
        bearer_token="bearer",
    )

    assert tc.kernel.with_library(library, data_dir=tmp_path, token=token) == "kernel"
    assert FakeKernelHandle.calls == [
        (
            "/lib/example-devco/b/0.1.0",
            "deps.example.test:9443",
            {
                "token_host": "http://127.0.0.1:8702",
                "actor_id": "example-admin",
                "public_key_b64": "pubkey",
                "data_dir": str(tmp_path),
            },
        )
    ]
