from __future__ import annotations

import base64
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_type_hints

from .opref import OpRef
from .ref import Ref
from . import uri as _uri

IR_ARTIFACT_CONTENT_TYPE = "application/tinychain+json"


def _is_method(form: Callable[..., Any]) -> bool:
    names = list(getattr(form, "__code__", None).co_varnames or ())
    return bool(names and names[0] == "self")


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    form: Callable[..., Any]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not callable(self.form):
            raise TypeError(f"expected a callable, got {type(self.form).__name__}")
        if not _is_method(self.form):
            raise TypeError(
                "tc.define route decorators are only valid on instance methods (first arg must be `self`)"
            )

    def __set_name__(self, _owner: type, attr_name: str) -> None:
        if self.name is None:
            object.__setattr__(self, "name", attr_name)

    def _return_type(self) -> Optional[type[Ref]]:
        try:
            rtype = get_type_hints(self.form, globalns=self.form.__globals__).get("return")
        except Exception:
            return None

        if isinstance(rtype, type) and issubclass(rtype, Ref):
            return rtype

        return None

    def _opref(self, instance: object) -> OpRef[Any]:
        route_name = self.name or self.form.__name__
        route = getattr(instance, "route", None)
        if route is None or not callable(route):
            raise TypeError("expected instance to define a .route(...) method")
        path = route(route_name)
        return OpRef(method=self.method, path=path)

    def __get__(self, instance: object, owner: type | None = None):
        if instance is None:
            return self

        def bound(*args, **kwargs):
            body = None
            if args and kwargs:
                raise TypeError("TinyChain route stubs accept either args or kwargs, not both")

            if args:
                if len(args) != 1:
                    raise TypeError("TinyChain route stubs accept at most one positional argument")
                body = args[0]
            elif kwargs:
                if len(kwargs) != 1 or "body" not in kwargs:
                    raise TypeError("TinyChain route stubs accept only the keyword argument `body`")
                body = kwargs["body"]

            opref = self._opref(instance)
            if body is not None:
                opref = OpRef(method=opref.method, path=opref.path, headers=opref.headers, body=body)
            rtype = self._return_type()
            return rtype(opref) if rtype is not None else opref

        bound.__name__ = self.name or self.form.__name__
        bound.__doc__ = self.form.__doc__
        return bound


def _decorate(
    method: str,
    form: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
):
    if form is None:
        return lambda actual: Route(method=method.upper(), form=actual, name=name)
    return Route(method=method.upper(), form=form, name=name)


def get(
    form: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
):
    return _decorate("GET", form, name=name)


def put(
    form: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
):
    return _decorate("PUT", form, name=name)


def post(
    form: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
):
    return _decorate("POST", form, name=name)


def delete(
    form: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
):
    return _decorate("DELETE", form, name=name)


class Library:
    """
    A v1-style Library definition surface: decorators define route stubs whose calls return typed
    reference values (deferred graph nodes), driven by return type hints.

    This is intentionally separate from `tc.Library` (which is a runtime stub for executing requests).
    """

    def __init__(
        self,
        *,
        publisher: str,
        name: str,
        version: str,
        dependencies: tuple[_uri.URI, ...] = (),
        authority: _uri.URI | None = None,
    ) -> None:
        self.publisher = publisher
        self.name = name
        self.version = version
        self.dependencies = dependencies
        self.authority = authority

    def id(self) -> _uri.URI:
        return _uri.library(publisher=self.publisher, name=self.name, version=self.version)

    def route(self, *path: str) -> str:
        return _uri.library(
            publisher=self.publisher,
            name=self.name,
            version=self.version,
            path=list(path) if path else None,
        ).path

    def link(self) -> _uri.URI:
        return _uri.library_link(
            publisher=self.publisher,
            name=self.name,
            version=self.version,
            authority=self.authority,
        )

    def schema(self) -> dict:
        return {
            "id": self.id().path,
            "version": self.version,
            "dependencies": [dep.path for dep in self.dependencies],
        }

    def schema_json(self) -> str:
        return json.dumps(self.schema(), separators=(",", ":"))


def _to_opref(value: object) -> Optional[OpRef[Any]]:
    if isinstance(value, Ref):
        return value.op  # type: ignore[return-value]
    if isinstance(value, OpRef):
        return value
    return None


def compile_ir(library: Library) -> dict:
    routes: list[dict] = []
    for name, attr in list(library.__class__.__dict__.items()):
        if not isinstance(attr, Route):
            continue

        if attr.method != "GET":
            raise ValueError("only GET methods are supported for tc.define compilation right now")

        result = attr.form(library)
        op = _to_opref(result)
        if op is not None:
            routes.append(
                {
                    "path": f"/{name}",
                    "op": {"method": op.method, "path": op.path},
                }
            )
        else:
            routes.append({"path": f"/{name}", "value": result})

    return {"schema": library.schema(), "routes": routes}


def install(
    library: Library,
    *,
    kernel: Optional[object] = None,
    data_dir: Optional[pathlib.Path] = None,
) -> object:
    try:
        import tinychain_local as local  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("`tc.define.install` requires the optional `tinychain-local` backend") from exc

    if kernel is None:
        if data_dir is None:
            raise ValueError("expected either `kernel` or `data_dir`")
        kernel = local.KernelHandle.local(data_dir=str(data_dir))

    payload = compile_ir(library)
    ir_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ir_b64 = base64.b64encode(ir_bytes).decode("ascii")

    install_payload = json.dumps(
        {
            "schema": library.schema(),
            "artifacts": [
                {
                    "path": "/lib/ir",
                    "content_type": IR_ARTIFACT_CONTENT_TYPE,
                    "bytes": ir_b64,
                }
            ],
        },
        separators=(",", ":"),
    )

    request = local.KernelRequest("PUT", "/lib", None, local.StateHandle(install_payload))
    return kernel.dispatch(request)
