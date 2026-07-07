from __future__ import annotations

from types import SimpleNamespace

import pytest

import tinychain as tc
from tinychain.autodiff import AutodiffError
from tinychain.autodiff.routes import RouteDerivativeIdentity, extract_route_identity


class RouteIdentityLibrary(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.get
    def fetch(self, key: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")

    @tc.post
    def create(self, value: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")

    @tc.put
    def replace(self, value: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")

    @tc.delete
    def remove(self, key: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")


@pytest.mark.parametrize(
    ("route_name", "expected_method"),
    [
        ("fetch", "GET"),
        ("create", "POST"),
        ("replace", "PUT"),
        ("remove", "DELETE"),
    ],
)
def test_extract_route_identity_for_bound_route_decorators(
    route_name: str,
    expected_method: str,
) -> None:
    library = RouteIdentityLibrary()
    target = getattr(library, route_name)

    identity = extract_route_identity(target)

    assert identity == RouteDerivativeIdentity.from_dict(identity.to_dict())
    assert identity.to_dict() == {
        "publisher": "example-devco",
        "library_name": "route_identity_library",
        "library_version": "0.1.0",
        "library_path": "/lib/example-devco/route_identity_library/0.1.0",
        "library_uri": "/lib/example-devco/route_identity_library/0.1.0",
        "route_name": route_name,
        "route_path": f"/{route_name}",
        "route_uri": f"/lib/example-devco/route_identity_library/0.1.0/path/{route_name}",
        "http_method": expected_method,
    }


def test_extract_route_identity_respects_authority_aware_library_uris() -> None:
    library = RouteIdentityLibrary(authority=tc.URI.parse("https://example.test:8443"))

    identity = extract_route_identity(library.create)

    assert identity.library_path == "/lib/example-devco/route_identity_library/0.1.0"
    assert (
        identity.library_uri
        == "https://example.test:8443/lib/example-devco/route_identity_library/0.1.0"
    )
    assert (
        identity.route_uri
        == "https://example.test:8443/lib/example-devco/route_identity_library/0.1.0/path/create"
    )


def test_extract_route_identity_rejects_unbound_route_descriptor() -> None:
    with pytest.raises(TypeError, match="bound TinyChain route target"):
        extract_route_identity(RouteIdentityLibrary.fetch)


def test_extract_route_identity_rejects_plain_callables() -> None:
    def plain_target() -> None:
        raise AssertionError("plain target must not execute")

    with pytest.raises(TypeError, match="call-site transform"):
        extract_route_identity(plain_target)


def test_extract_route_identity_rejects_malformed_route_metadata() -> None:
    def fake_target() -> None:
        raise AssertionError("fake target must not execute")

    fake_target.__tc_route__ = SimpleNamespace(name="", method="POST")
    fake_target.__tc_instance__ = RouteIdentityLibrary()

    with pytest.raises(AutodiffError) as exc_info:
        extract_route_identity(fake_target)

    assert exc_info.value.category == "non_differentiable_route"
    assert "missing route name" in str(exc_info.value)


def test_extract_route_identity_does_not_call_route_compile_install_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"body": 0}

    class SideEffectLibrary(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def mutate(self, value: tc.String) -> tc.String:
            calls["body"] += 1
            raise AssertionError("route body must not execute")

    def forbidden_behavior(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compile/install/dispatch behavior must not run")

    monkeypatch.setattr("tinychain.library.compile_ir", forbidden_behavior)
    monkeypatch.setattr("tinychain.library.install", forbidden_behavior)
    monkeypatch.setattr(tc, "execute", forbidden_behavior)

    identity = extract_route_identity(SideEffectLibrary().mutate)

    assert calls == {"body": 0}
    assert identity.route_name == "mutate"
    assert identity.http_method == "POST"
