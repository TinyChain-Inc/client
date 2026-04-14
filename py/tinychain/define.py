from __future__ import annotations

import base64
import inspect
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

from .opref import OpRef
from .ref import Ref
from . import autograph
from .state import ContextResult, OpDef, Scalar, TCRef, autobox, context, current_context, scoped_context
from .uri import URI, _segment, uri as _uri

IR_ARTIFACT_CONTENT_TYPE = "application/tinychain+json"


def _is_method(form: Callable[..., Any]) -> bool:
    names = list(getattr(form, "__code__", None).co_varnames or ())
    return bool(names and names[0] == "self")


def self_subject(*path: str) -> str:
    if not path:
        return "$self"
    return "$self/" + "/".join(path)

def _route_path(subject: object, route_name: str) -> str:
    for attr in ("publisher", "name", "version"):
        if not hasattr(subject, attr):
            raise TypeError("expected library class with publisher/name/version fields")
    publisher = getattr(subject, "publisher")
    name = getattr(subject, "name")
    version = getattr(subject, "version")
    route_uri = URI(
        "/" + "/".join(
            [
                "lib",
                _segment("publisher", publisher),
                _segment("name", name),
                _segment("version", version),
                _segment("path", route_name),
            ]
        )
    )

    authority = getattr(subject, "authority", None)
    authority_uri: URI | None = None
    if isinstance(authority, URI):
        authority_uri = authority
    elif isinstance(authority, str):
        authority_uri = URI.parse(authority)

    if authority_uri is not None and authority_uri.host is not None:
        route_uri = URI(
            path=route_uri.path,
            scheme=authority_uri.scheme,
            host=authority_uri.host,
            port=authority_uri.port,
        )
        return route_uri.absolute()

    return route_uri.path


def _library_class(library: "Library | type[Library]") -> type["Library"]:
    if isinstance(library, type) and issubclass(library, Library):
        return library
    if isinstance(library, Library):
        return type(library)
    raise TypeError("expected a Library class")


def _validate_library_class(library: type["Library"]) -> None:
    if "__init__" in library.__dict__:
        raise TypeError("Library subclasses must not define __init__")
    if not getattr(library, "publisher", None) or not getattr(library, "name", None) or not getattr(library, "version", None):
        raise TypeError("Library requires publisher, name, and version")


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    form: Callable[..., Any]
    name: Optional[str] = field(default=None, init=False)

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
        path = _route_path(instance, route_name)
        return OpRef(method=self.method, path=path)

    def opdef(self, instance: object) -> OpDef:
        sig = inspect.signature(self.form)
        params = list(sig.parameters.values())
        if not params or params[0].name != "self":
            raise TypeError("route form must begin with a `self` parameter")
        library_cls = _library_class(instance)
        _validate_library_class(library_cls)
        if isinstance(instance, Library):
            if (
                instance.publisher != getattr(library_cls, "publisher", None)
                or instance.name != getattr(library_cls, "name", None)
                or instance.version != getattr(library_cls, "version", None)
            ):
                raise TypeError("Library instance fields must match class attributes")
        return _compile_opdef_route(self, library_cls, params)

    def __get__(self, instance: object, owner: type | None = None):
        if instance is None:
            return self

        def bound(*args, **kwargs):
            body = None
            if args and kwargs:
                raise TypeError("TinyChain route stubs accept either args or kwargs, not both")

            if args:
                if self.method == "POST":
                    sig = inspect.signature(self.form)
                    params = list(sig.parameters.values())
                    injected = {"cxt", "ctx", "txn"}
                    param_names = [
                        param.name
                        for param in params[1:]
                        if param.kind in (
                            param.POSITIONAL_ONLY,
                            param.POSITIONAL_OR_KEYWORD,
                            param.KEYWORD_ONLY,
                        )
                        and param.name not in injected
                    ]
                    if len(args) != len(param_names):
                        raise TypeError(
                            "TinyChain POST stubs require one positional argument per parameter"
                        )
                    body = dict(zip(param_names, args, strict=True))
                else:
                    if len(args) != 1:
                        raise TypeError("TinyChain route stubs accept at most one positional argument")
                    body = args[0]
            elif kwargs:
                if self.method == "POST" and not (len(kwargs) == 1 and "body" in kwargs):
                    body = kwargs
                else:
                    if len(kwargs) != 1 or "body" not in kwargs:
                        raise TypeError("TinyChain route stubs accept only the keyword argument `body`")
                    body = kwargs["body"]

            opref = self._opref(instance)
            if body is not None:
                opref = OpRef(method=opref.method, path=opref.path, headers=opref.headers, body=body)
            rtype = self._return_type()
            result = rtype(opref) if rtype is not None else opref

            from .executor import try_current

            exec_ctx = try_current()
            if exec_ctx is not None and exec_ctx.should_auto_execute():
                import tinychain as tc

                return tc.execute(result)

            return result

        bound.__name__ = self.name or self.form.__name__
        bound.__doc__ = self.form.__doc__
        bound.__tc_route__ = self
        bound.__tc_instance__ = instance
        return bound


def _decorate(
    method: str,
    form: Optional[Callable[..., Any]] = None,
):
    if form is None:
        return lambda actual: Route(method=method.upper(), form=actual)
    if _is_method(form):
        return Route(method=method.upper(), form=form)
    return _compile_opdef_callable(form, method=method.upper())


def get(
    form: Optional[Callable[..., Any]] = None,
):
    return _decorate("GET", form)


def put(
    form: Optional[Callable[..., Any]] = None,
):
    return _decorate("PUT", form)


def post(
    form: Optional[Callable[..., Any]] = None,
):
    return _decorate("POST", form)


def delete(
    form: Optional[Callable[..., Any]] = None,
):
    return _decorate("DELETE", form)

def _compile_opdef_callable(form: Callable[..., Any], *, method: str) -> OpDef:
    sig = inspect.signature(form)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        raise TypeError("expected a standalone callable, not a method")

    injected_names = {"cxt", "ctx", "txn"}
    arg_names = []
    with scoped_context():
        args: list[Scalar] = []
        kwargs: dict[str, Scalar] = {}
        for param in params:
            if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
                raise TypeError("variadic opdef callables are not supported")
            if param.name in injected_names:
                raise TypeError(f"reserved parameter '{param.name}' is not supported in opdef callables")
            placeholder = Scalar.id(param.name)
            arg_names.append(param.name)
            if param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = placeholder
            else:
                args.append(placeholder)

        result = form(*args, **kwargs)
        ctx = current_context()
        if ctx is not None and ctx.form() and not isinstance(result, ContextResult):
            result = ctx.result(result)

    if isinstance(result, OpDef):
        _validate_opdef_method(method, result)
        _validate_opdef(result, set(arg_names))
        return result

    if isinstance(result, ContextResult):
        form_items = list(result.form)
        scalar = autobox(result.result)
        form_items.append(("result", scalar))
        opdef = _opdef_from_method(method, arg_names, form_items)
        _validate_opdef(opdef, set(arg_names))
        return opdef

    if isinstance(result, dict):
        form_items: list[tuple[str, Scalar]] = []
        for key, value in result.items():
            if not isinstance(key, str):
                raise TypeError("opdef form keys must be strings")
            form_items.append((key, autobox(value)))
        opdef = _opdef_from_method(method, arg_names, form_items)
        _validate_opdef(opdef, set(arg_names))
        return opdef

    if _to_opref(result) is not None:
        raise TypeError("opdef callables must return an OpDef or Scalar, not an OpRef")

    scalar = autobox(result)
    opdef = _opdef_from_method(method, arg_names, [("result", scalar)])
    _validate_opdef(opdef, set(arg_names))
    return opdef


def _opdef_from_method(method: str, arg_names: list[str], form: list[tuple[str, Scalar]]) -> OpDef:
    if method == "GET":
        key_name = arg_names[0] if arg_names else "key"
        return OpDef.get(key_name, form)
    if method == "PUT":
        key_name = arg_names[0] if len(arg_names) > 0 else "key"
        value_name = arg_names[1] if len(arg_names) > 1 else "value"
        return OpDef.put(key_name, value_name, form)
    if method == "POST":
        return OpDef.post(form)
    if method == "DELETE":
        key_name = arg_names[0] if arg_names else "key"
        return OpDef.delete(key_name, form)
    raise ValueError(f"unsupported opdef method {method}")


def _class_dependencies(cls: type) -> tuple[URI, ...]:
    deps: list[URI] = []
    class_deps = getattr(cls, "dependencies", ())
    if class_deps:
        deps.extend(class_deps)
    for value in vars(cls).values():
        if hasattr(value, "id") and callable(getattr(value, "id")):
            try:
                dep = value.id()
            except Exception:
                continue
            if dep not in deps:
                deps.append(dep)
    return tuple(deps)


class Library:
    """
    A v1-style Library definition surface: decorators define route stubs whose calls return typed
    reference values (deferred graph nodes), driven by return type hints.

    This is intentionally separate from `tc.Library` (which is a runtime stub for executing requests).
    """

    def __init__(
        self,
        *,
        publisher: str | None = None,
        name: str | None = None,
        version: str | None = None,
        dependencies: tuple[URI, ...] | None = None,
        authority: URI | None = None,
    ) -> None:
        cls = type(self)
        publisher = publisher or getattr(cls, "publisher", None)
        name = name or getattr(cls, "name", None)
        version = version or getattr(cls, "version", None)
        if not publisher or not name or not version:
            raise TypeError("Library requires publisher, name, and version")
        self.publisher = publisher
        self.name = name
        self.version = version
        if dependencies is None:
            dependencies = _class_dependencies(type(self))
        self.dependencies = dependencies
        self.authority = authority or getattr(cls, "authority", None)

    def id(self) -> URI:
        return URI(
            "/" + "/".join(
                [
                    "lib",
                    _segment("publisher", self.publisher),
                    _segment("name", self.name),
                    _segment("version", self.version),
                ]
            )
        )

    def link(self) -> URI:
        base = self.id()
        if self.authority is None:
            return base
        return URI(
            path=base.path,
            scheme=self.authority.scheme,
            host=self.authority.host,
            port=self.authority.port,
        )

    def schema(self) -> dict:
        return {
            "id": self.id().path,
            "version": self.version,
            "dependencies": [dep.path for dep in self.dependencies],
        }

    def schema_json(self) -> str:
        return json.dumps(self.schema(), separators=(",", ":"))

    @classmethod
    def class_id(cls) -> URI:
        publisher = getattr(cls, "publisher", None)
        name = getattr(cls, "name", None)
        version = getattr(cls, "version", None)
        if not publisher or not name or not version:
            raise TypeError("Library requires publisher, name, and version")
        return URI(
            "/" + "/".join(
                [
                    "lib",
                    _segment("publisher", publisher),
                    _segment("name", name),
                    _segment("version", version),
                ]
            )
        )

    @classmethod
    def class_schema(cls) -> dict:
        deps = _class_dependencies(cls)
        return {
            "id": cls.class_id().path,
            "version": getattr(cls, "version", None),
            "dependencies": [dep.path for dep in deps],
        }

    @classmethod
    def class_schema_json(cls) -> str:
        return json.dumps(cls.class_schema(), separators=(",", ":"))


def _to_opref(value: object) -> Optional[OpRef[Any]]:
    if isinstance(value, Ref):
        return value.op  # type: ignore[return-value]
    if isinstance(value, OpRef):
        return value
    return None


def compile_ir(library: Library | type[Library]) -> dict:
    library_cls = _library_class(library)
    _validate_library_class(library_cls)
    routes: list[dict] = []
    for name, attr in list(library_cls.__dict__.items()):
        if not isinstance(attr, Route):
            continue

        result = _compile_route(attr, library_cls)
        op = _to_opref(result)
        if op is not None:
            routes.append(
                {
                    "path": f"/{name}",
                    "op": {"method": op.method, "path": op.path},
                }
            )
            continue

        if isinstance(result, OpDef):
            routes.append({"path": f"/{name}", "opdef": result.to_json()})
            continue

        routes.append({"path": f"/{name}", "value": result})

    return {"schema": library_cls.class_schema(), "routes": routes}


def _compile_route(route: Route, library: type[Library]) -> object:
    sig = inspect.signature(route.form)
    params = list(sig.parameters.values())
    if not params or params[0].name != "self":
        raise TypeError("route form must begin with a `self` parameter")

    if len(params) > 1:
        return _compile_opdef_route(route, library, params)

    result = route.form(library)
    if isinstance(result, (OpDef, Scalar)):
        return _compile_opdef_route(route, library, params)

    return result


def _compile_opdef_route(
    route: Route,
    library: type[Library],
    params: list[inspect.Parameter],
) -> OpDef:

    injected_names = {"cxt", "ctx", "txn"}
    form = route.form
    if not any(param.name in injected_names for param in params[1:]):
        form = autograph.transform(route.form)

    arg_names = [
        param.name
        for param in params[1:]
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and param.name not in injected_names
    ]

    with scoped_context():
        args: list[Scalar] = []
        kwargs: dict[str, Scalar] = {}
        for idx, param in enumerate(params[1:], start=1):
            if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
                raise TypeError("variadic route parameters are not supported in tc.define compilation")
            if param.name in injected_names:
                if idx != 1:
                    raise TypeError(
                        f"reserved parameter '{param.name}' must be the first argument after self"
                    )
                injected = context()
                if param.kind == param.KEYWORD_ONLY:
                    kwargs[param.name] = injected
                else:
                    args.append(injected)
                continue
            placeholder = Scalar.id(param.name)
            if param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = placeholder
            else:
                args.append(placeholder)
        result = form(library, *args, **kwargs)
        ctx = current_context()
        if ctx is not None and ctx.form() and not isinstance(result, ContextResult):
            result = ctx.result(result)

    if isinstance(result, OpDef):
        result = _inline_opref_refs(result)
        _validate_opdef_method(route.method, result)
        _validate_opdef(result, _allowed_inputs_from_params(route, params))
        return result

    if isinstance(result, ContextResult):
        form = list(result.form)
        scalar = autobox(result.result)
        form.append(("result", scalar))
        opdef = _opdef_from_method(route.method, arg_names, form)
        opdef = _inline_opref_refs(opdef)
        _validate_opdef(opdef, set(arg_names))
        return opdef

    if isinstance(result, dict):
        form: list[tuple[str, Scalar]] = []
        for key, value in result.items():
            if not isinstance(key, str):
                raise TypeError("opdef form keys must be strings")
        form.append((key, autobox(value)))
        opdef = _opdef_from_method(route.method, arg_names, form)
        opdef = _inline_opref_refs(opdef)
        _validate_opdef(opdef, set(arg_names))
        return opdef

    if _to_opref(result) is not None:
        raise TypeError("opdef routes must return an OpDef or Scalar, not an OpRef")

    scalar = autobox(result)
    form = [("result", scalar)]
    opdef = _opdef_from_method(route.method, arg_names, form)
    opdef = _inline_opref_refs(opdef)
    _validate_opdef(opdef, set(arg_names))
    return opdef


def _validate_opdef_method(method: str, opdef: OpDef) -> None:
    expected = opdef.method.upper()
    if expected != method.upper():
        raise ValueError(f"route method {method} does not match OpDef {expected}")


def _allowed_inputs_from_params(route: Route, params: list[inspect.Parameter]) -> set[str]:
    arg_names = [
        param.name
        for param in params[1:]
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]

    if route.method == "GET":
        return {arg_names[0]} if arg_names else {"key"}
    if route.method == "PUT":
        if len(arg_names) >= 2:
            return {arg_names[0], arg_names[1]}
        if len(arg_names) == 1:
            return {arg_names[0]}
        return {"key", "value"}
    if route.method == "DELETE":
        return {arg_names[0]} if arg_names else {"key"}
    if route.method == "POST":
        return set(arg_names)
    return set()


def _validate_opdef(opdef: OpDef, allowed_inputs: set[str]) -> None:
    defined = set()
    for name, _ in opdef.form:
        if name in defined:
            raise ValueError(f"duplicate OpDef id {name}")
        defined.add(name)
    allowed = set(allowed_inputs)
    allowed.add("self")
    allowed |= defined
    form_map = {name: scalar for name, scalar in opdef.form}

    for _, scalar in opdef.form:
        for node in _walk_scalars_with_opdef(scalar):
            if node.ref is not None:
                _validate_tcref(node.ref, allowed, form_map)


def _inline_opref_refs(opdef: OpDef) -> OpDef:
    form_map = {name: scalar for name, scalar in opdef.form}

    def resolve_scalar(scalar: Scalar) -> Scalar:
        if scalar.ref is not None and scalar.ref.op is not None and isinstance(scalar.ref.op, Scalar):
            op_scalar = scalar.ref.op
            if op_scalar.ref is not None and op_scalar.ref.id is not None:
                target = form_map.get(op_scalar.ref.id.name)
                if target is not None and target.ref is not None and isinstance(target.ref.op, OpRef):
                    return Scalar(ref=TCRef(op=target.ref.op))

        if scalar.op is not None:
            return Scalar(op=_inline_opref_refs(scalar.op))
        if scalar.map is not None:
            return Scalar.map_of({key: resolve_scalar(value) for key, value in scalar.map.items()})
        if scalar.tuple is not None:
            return Scalar.tuple_of([resolve_scalar(value) for value in scalar.tuple])
        return scalar

    return OpDef(
        method=opdef.method,
        form=[(name, resolve_scalar(scalar)) for name, scalar in opdef.form],
        key=opdef.key,
        value=opdef.value,
    )


def _validate_tcref(tcref, allowed: set[str], form_map: dict[str, Scalar]) -> None:
    if tcref.id is not None:
        name = tcref.id.name
        if name not in allowed:
            logging.info("OpDef depends on undefined id $%s", name)

    if tcref.op is not None:
        if isinstance(tcref.op, OpRef):
            _validate_opref(tcref.op)
        elif isinstance(tcref.op, Scalar):
            resolved = _resolve_opref_ref(tcref.op, form_map)
            if resolved is not None:
                _validate_opref(resolved)


def _validate_opref(opref: OpRef[Any]) -> None:
    method = opref.method.upper()
    if method == "GET":
        if not isinstance(opref.args, list) or len(opref.args) != 1:
            raise ValueError("GET OpRef expects a single-element list args")
    elif method == "PUT":
        if not isinstance(opref.args, list) or len(opref.args) != 2:
            raise ValueError("PUT OpRef expects a two-element list args")
    elif method == "POST":
        if not isinstance(opref.args, dict):
            raise ValueError("POST OpRef expects a dict args")
    elif method == "DELETE":
        pass
    else:
        raise ValueError(f"unsupported OpRef method {opref.method}")


def _resolve_opref_ref(value: Scalar, form_map: dict[str, Scalar]) -> OpRef | None:
    if value.ref is not None and isinstance(value.ref.op, OpRef):
        return value.ref.op
    if value.ref is not None and value.ref.id is not None:
        target = form_map.get(value.ref.id.name)
        if target is not None and target is not value:
            return _resolve_opref_ref(target, form_map)
    return None


def _walk_scalars_with_opdef(root: Scalar):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node

        if node.op is not None:
            for _, inner in node.op.form:
                stack.append(inner)
        if node.map is not None:
            for value in reversed(list(node.map.values())):
                stack.append(value)
        if node.tuple is not None:
            for value in reversed(list(node.tuple)):
                stack.append(value)


def install(
    library: Library | type[Library],
    *,
    kernel: Optional[object] = None,
    data_dir: Optional[pathlib.Path] = None,
    bearer_token: Optional[str] = None,
) -> object:
    try:
        import tinychain_local as local  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("`tc.define.install` requires the optional `tinychain-local` backend") from exc

    if bearer_token is None:
        raise ValueError("expected `bearer_token` for library installs")

    if kernel is None:
        if data_dir is None:
            raise ValueError("expected either `kernel` or `data_dir`")
        kernel = local.KernelHandle.local(data_dir=str(data_dir))

    library_cls = _library_class(library)
    _validate_library_class(library_cls)
    if isinstance(library, Library):
        if (
            library.publisher != getattr(library_cls, "publisher", None)
            or library.name != getattr(library_cls, "name", None)
            or library.version != getattr(library_cls, "version", None)
        ):
            raise TypeError("Library instance fields must match class attributes")

    payload = compile_ir(library_cls)
    ir_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ir_b64 = base64.b64encode(ir_bytes).decode("ascii")

    install_payload = json.dumps(
        {
            "schema": library_cls.class_schema(),
            "artifacts": [
                {
                    "path": _uri("lib", "ir").path,
                    "content_type": IR_ARTIFACT_CONTENT_TYPE,
                    "bytes": ir_b64,
                }
            ],
        },
        separators=(",", ":"),
    )

    headers = [("authorization", f"Bearer {bearer_token}")]

    install_path = _uri("lib").path
    request = local.KernelRequest(
        "PUT",
        install_path,
        headers,
        local.StateHandle(install_payload),
    )
    response = kernel.dispatch(request)
    return _finalize_implicit_txn(local, kernel, response, install_path, bearer_token)


def _header_value(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    needle = name.lower()
    for key, value in headers:
        if str(key).lower() == needle:
            return str(value)
    return None


def _finalize_implicit_txn(
    local: object,
    kernel: object,
    response: object,
    path: str,
    bearer_token: str | None,
) -> object:
    txn_id = _header_value(response, "x-tc-txn-id")
    if not txn_id:
        return response

    headers = [("authorization", f"Bearer {bearer_token}")] if bearer_token else None
    commit = local.KernelRequest("POST", f"{path}?txn_id={txn_id}", headers, None)
    return kernel.dispatch(commit)
