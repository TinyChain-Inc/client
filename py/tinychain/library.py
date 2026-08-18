from __future__ import annotations

import inspect
import json
import logging
import pathlib
import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, get_args, get_origin, get_type_hints

from .auth import bearer_token as _bearer_token
from . import opref as runtime_opref
from .opref import OpRef
from .ref import Ref
from . import _autograph
from .state import Collection, ContextResult, DeleteOpDef, DeleteOpRef, GetOpDef, GetOpRef, IdRef, OpDef, OpRef as StateOpRef, PostOpDef, PostOpRef, PutOpDef, PutOpRef, Scalar, TCRef, autobox, context, current_context, form_of, map_of as scalar_map_of, scalar_for_hint, scoped_context, tuple_of as scalar_tuple_of
from .state.value import Bool, Map, Number, String, Tuple, Value
from .uri import URI, _segment, validate_resource_name


_INJECTED_ROUTE_PARAM_NAMES = {"cxt", "ctx", "txn"}
_LIB_ROOT_URI = URI("lib")
_LIB_WASM_URI = URI("lib", "wasm")

def _is_method(form: Callable[..., Any]) -> bool:
    names = list(getattr(form, "__code__", None).co_varnames or ())
    return bool(names and names[0] == "self")


def _greatest_common_superclass(*types: type) -> type:
    classes = [t for t in types if isinstance(t, type)]
    if not classes:
        return Value
    for candidate in classes[0].mro():
        if candidate is object:
            return Value
        if all(candidate in cls.mro() for cls in classes):
            return candidate
    return Value


def _runtime_type_hint(type_hint: object, default: type = Value) -> type:
    if type_hint in (inspect.Parameter.empty, None):
        return default
    if type_hint is Any:
        return Value
    if type_hint is str:
        return String
    if type_hint is bool:
        return Bool
    if type_hint in (int, float):
        return Number
    if type_hint is dict:
        return Map
    if type_hint in (list, tuple):
        return Tuple

    origin = get_origin(type_hint)
    if origin is not None:
        if origin is dict:
            return Map
        if origin in (list, tuple):
            return Tuple
        if origin is Optional:
            args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        else:
            args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        resolved = [_runtime_type_hint(arg, default) for arg in args]
        return _greatest_common_superclass(*resolved)

    if isinstance(type_hint, type):
        if issubclass(type_hint, Collection):
            return type_hint
        if issubclass(type_hint, Ref):
            return type_hint
        if issubclass(type_hint, Value):
            return type_hint
        if issubclass(type_hint, Scalar):
            return type_hint

    return default


def self_subject(*path: str) -> str:
    if not path:
        return "$self"
    return "$self/" + "/".join(path)


def _compile_self_instance(library: type["Library"]) -> "Library":
    return library()


def _route_path(subject: object, route_name: str) -> str:
    # Identity is always resolved from the class, never from (possibly mutated)
    # instance attributes.
    cls = subject if isinstance(subject, type) else type(subject)
    publisher, resource_name, version = _class_identity(cls)
    route_uri = URI(
        "/" + "/".join(
            [
                "lib",
                _segment("publisher", publisher),
                _segment("resource_name", resource_name),
                _segment("version", version),
                _segment("path", route_name),
            ]
        )
    )
    if not isinstance(route_uri, URI):
        raise TypeError("expected URI route path")

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


def _contains_symbolic(value: object) -> bool:
    if isinstance(value, (Scalar, Collection)):
        return isinstance(form_of(value), TCRef)

    if isinstance(value, dict):
        return any(_contains_symbolic(v) for v in value.values())

    if isinstance(value, (list, tuple)):
        return any(_contains_symbolic(v) for v in value)

    return False


def _route_arg_param_names(params: list[inspect.Parameter], *, skip_first_self: bool) -> list[str]:
    iterable = params[1:] if skip_first_self else params
    return [
        param.name
        for param in iterable
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and param.name not in _INJECTED_ROUTE_PARAM_NAMES
    ]


def _route_body_from_call(route: "Route", args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    body = None
    if args:
        raise TypeError("TinyChain route stubs require keyword arguments or one explicit body=")

    if kwargs:
        if len(kwargs) == 1 and "body" in kwargs:
            body = kwargs["body"]
        elif route.method == "POST":
            body = kwargs
        else:
            params = list(inspect.signature(route.form).parameters.values())
            param_names = _route_arg_param_names(params, skip_first_self=True)
            if len(param_names) != 1 or set(kwargs) != set(param_names):
                raise TypeError(
                    f"TinyChain {route.method} stubs require the keyword argument "
                    f"`{param_names[0]}` or `body`"
                    if len(param_names) == 1
                    else f"TinyChain {route.method} stubs require one explicit `body`"
                )
            body = kwargs[param_names[0]]

    return body


def _execute_route_result_if_needed(result: object, body: object) -> object:
    if _contains_symbolic(result):
        return result

    if _contains_symbolic(body):
        return result

    from .executor import try_current

    exec_ctx = try_current()
    if exec_ctx is not None and not exec_ctx.is_eager():
        return result

    import tinychain as tc

    return tc.execute(result)


def _append_context_result_form(route: "Route", form: list[tuple[str, Scalar]], raw_result: object) -> None:
    # Contract: mapping return values become named OpDef entries unless the route
    # is explicitly typed as generic Ref, in which case we preserve the mapping
    # as a single value under "result".
    if isinstance(raw_result, Mapping):
        declared_rtype = route._return_type()
        if declared_rtype is Ref:
            form.append(("result", autobox(raw_result)))
            return

        form.extend((key, autobox(value)) for key, value in raw_result.items())
        return

    form.append(("result", autobox(raw_result)))


def _library_class(library: "Library | type[Library]") -> type["Library"]:
    if isinstance(library, type) and issubclass(library, Library):
        return library
    if isinstance(library, Library):
        return type(library)
    raise TypeError("expected a Library class")


def _class_identity(cls: type["Library"]) -> tuple[str, str, str]:
    """Return the validated canonical ``(publisher, resource_name, version)``.

    ``resource_name`` is the single source of truth for the library name path
    component ``/lib/{publisher}/{resource_name}/{version}``. It is class-level
    metadata with the same inheritance semantics as ``publisher`` and
    ``version``; there is no derivation from the Python class name.
    """
    # A raw ``name`` field is never a source of Library identity, regardless of
    # its value or type, and even when a valid ``resource_name`` is also
    # declared. Walk the MRO (nearest declaration wins, mirroring attribute
    # resolution) so inherited raw declarations cannot bypass validation. A
    # decorated route method named ``name`` is a ``Route`` descriptor, not a raw
    # identity field, and remains valid.
    for klass in cls.__mro__:
        if klass is Library:
            break
        if "name" in klass.__dict__:
            if not isinstance(klass.__dict__["name"], Route):
                raise TypeError(
                    "Library identity comes from resource_name; a raw class-level "
                    "'name' field is not supported"
                )
            break
    publisher = getattr(cls, "publisher", None)
    resource_name = getattr(cls, "resource_name", None)
    version = getattr(cls, "version", None)
    if not publisher or not resource_name or not version:
        raise TypeError("Library requires class publisher, resource_name, and version")
    return (
        _segment("publisher", publisher),
        validate_resource_name(resource_name),
        _segment("version", version),
    )


def _validate_library_class(library: type["Library"]) -> None:
    if "__init__" in library.__dict__:
        raise TypeError("Library subclasses must not define __init__")
    _class_identity(library)


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    form: Callable[..., Any]
    source: Optional[str] = None
    name: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not callable(self.form):
            raise TypeError(f"expected a callable, got {type(self.form).__name__}")
        if not _is_method(self.form):
            raise TypeError(
                "TinyChain route decorators are only valid on instance methods (first arg must be `self`)"
            )

    def __set_name__(self, _owner: type, attr_name: str) -> None:
        if self.name is None:
            object.__setattr__(self, "name", attr_name)

    def _return_type(self) -> Optional[type]:
        try:
            rtype = get_type_hints(self.form, globalns=self.form.__globals__).get("return")
        except (NameError, TypeError):
            return None
        if rtype is None:
            sig_rtype = inspect.signature(self.form).return_annotation
            if sig_rtype is inspect.Signature.empty:
                return None
            rtype = sig_rtype

        if isinstance(rtype, type) and issubclass(rtype, Ref):
            return rtype
        resolved = _runtime_type_hint(rtype, default=Value)

        if (
            isinstance(resolved, type)
            and issubclass(resolved, Collection)
            and resolved is not Collection
        ):
            return resolved
        if isinstance(resolved, type) and issubclass(resolved, Scalar) and resolved is not Scalar:
            return resolved
        if isinstance(resolved, type) and issubclass(resolved, Value) and resolved is not Value:
            return resolved

        return None

    def _bound_signature(self) -> inspect.Signature:
        sig = inspect.signature(self.form)
        try:
            hints = get_type_hints(self.form, globalns=self.form.__globals__)
        except (NameError, TypeError):
            hints = {}

        params: list[inspect.Parameter] = []
        for index, param in enumerate(sig.parameters.values()):
            if index == 0 and param.name == "self":
                continue
            annotation = hints.get(param.name, param.annotation)
            params.append(
                param.replace(annotation=_runtime_type_hint(annotation, default=Value))
            )

        return sig.replace(
            parameters=params,
            return_annotation=_runtime_type_hint(hints.get("return", sig.return_annotation), default=Value),
        )

    def _opref(self, instance: object) -> OpRef[Any]:
        route_name = self.name or self.form.__name__
        path = _route_path(instance, route_name)
        method = self.method.upper()
        if method == "GET":
            return runtime_opref.get(path)
        if method == "PUT":
            return runtime_opref.put(path)
        if method == "POST":
            return runtime_opref.post(path)
        if method == "DELETE":
            return runtime_opref.delete(path)
        raise ValueError(f"unsupported route method {self.method}")

    def opdef(self, instance: object) -> OpDef:
        sig = inspect.signature(self.form)
        params = list(sig.parameters.values())
        if not params or params[0].name != "self":
            raise TypeError("route form must begin with a `self` parameter")
        if getattr(instance, "_bind_class_route", None) is not None:
            return _compile_opdef_route(self, instance, params)
        library_cls = _library_class(instance)
        _validate_library_class(library_cls)
        # Identity is class-authoritative: route compilation always uses a fresh
        # class-derived instance, so instance-level mutation cannot affect it.
        return _compile_opdef_route(self, _compile_self_instance(library_cls), params)

    def __get__(self, instance: object, owner: type | None = None):
        if instance is None:
            return self

        # Class instances use state-subject refs so construction and bound
        # method invocation remain one canonical deferred graph.
        bind_class_route = getattr(instance, "_bind_class_route", None)
        if bind_class_route is not None:
            return bind_class_route(self)

        def bound(*args, **kwargs):
            body = _route_body_from_call(self, args, kwargs)

            opref = self._opref(instance)
            if body is not None:
                opref = opref.with_body(body)
            rtype = self._return_type()
            result = rtype(opref) if rtype is not None else opref

            return _execute_route_result_if_needed(result, body)

        bound.__name__ = self.name or self.form.__name__
        bound.__doc__ = self.form.__doc__
        bound.__signature__ = self._bound_signature()
        bound.__annotations__ = {
            name: param.annotation
            for name, param in bound.__signature__.parameters.items()
            if param.annotation is not inspect.Parameter.empty
        }
        if bound.__signature__.return_annotation is not inspect.Signature.empty:
            bound.__annotations__["return"] = bound.__signature__.return_annotation
        bound.__tc_route__ = self
        bound.__tc_instance__ = instance
        return bound


def _decorate(
    method: str,
    form: Optional[Callable[..., Any]] = None,
):
    if form is None:
        return lambda actual: Route(
            method=method.upper(),
            form=actual,
            source=_capture_source(actual),
        )
    if _is_method(form):
        return Route(
            method=method.upper(),
            form=form,
            source=_capture_source(form),
        )
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


def _capture_source(form: Callable[..., Any]) -> str | None:
    try:
        return inspect.getsource(form)
    except (OSError, TypeError):
        return None


def _compile_opdef_callable(form: Callable[..., Any], *, method: str) -> OpDef:
    sig = inspect.signature(form)
    try:
        hints = get_type_hints(form, globalns=form.__globals__)
    except (NameError, TypeError):
        hints = {}
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        raise TypeError("expected a standalone callable, not a method")

    injected_names = _INJECTED_ROUTE_PARAM_NAMES
    arg_names = []
    with scoped_context() as cxt:
        args: list[Scalar] = []
        kwargs: dict[str, Scalar] = {}
        for param in params:
            if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
                raise TypeError("variadic opdef callables are not supported")
            if param.name in injected_names:
                raise TypeError(f"reserved parameter '{param.name}' is not supported in opdef callables")
            annotation = hints.get(param.name, param.annotation)
            placeholder = scalar_for_hint(param.name, _runtime_type_hint(annotation, default=Value))
            arg_names.append(param.name)
            if param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = placeholder
            else:
                args.append(placeholder)

        result = form(*args, **kwargs)
        if cxt.form() and not isinstance(result, ContextResult):
            result = cxt.result(result)

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
        return GetOpDef(key_name, form)
    if method == "PUT":
        key_name = arg_names[0] if len(arg_names) > 0 else "key"
        value_name = arg_names[1] if len(arg_names) > 1 else "value"
        return PutOpDef(key_name, value_name, form)
    if method == "POST":
        return PostOpDef(form)
    if method == "DELETE":
        key_name = arg_names[0] if arg_names else "key"
        return DeleteOpDef(key_name, form)
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
            except (TypeError, AttributeError):
                continue
            if dep not in deps:
                deps.append(dep)
    return tuple(deps)


class Library:
    """
    Canonical Library surface.

    Subclasses use class-level manifest metadata and route decorators.
    """

    publisher: str
    resource_name: str
    version: str

    # Canonical identity metadata is class-authoritative and read-only on
    # instances: assigning it on an instance would create a misleading shadow
    # value that identity resolution (``_class_identity``) ignores.
    _IDENTITY_FIELDS = frozenset({"publisher", "resource_name", "version"})

    def __setattr__(self, name: str, value: object) -> None:
        if name in Library._IDENTITY_FIELDS:
            raise AttributeError(
                f"{name!r} is class-level Library identity metadata and is "
                "read-only on instances; declare it on the class"
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        publisher: str | None = None,
        resource_name: str | None = None,
        name: str | None = None,
        version: str | None = None,
        authority: URI | None = None,
    ) -> None:
        cls = type(self)
        if cls is Library:
            raise TypeError(
                "Library must be subclassed with class publisher, resource_name, and version metadata"
            )

        if (
            publisher is not None
            or resource_name is not None
            or name is not None
            or version is not None
        ):
            raise TypeError(
                "Library manifest metadata must be declared on the class"
            )
        # Validate class-level identity eagerly, but do not copy it onto the
        # instance: ``publisher``/``resource_name``/``version`` remain
        # class-authoritative and are always resolved via ``_class_identity``.
        _class_identity(cls)
        self.dependencies = _class_dependencies(cls)
        self.authority = authority or getattr(cls, "authority", None)

    def id(self) -> URI:
        return type(self).class_id()

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

    @classmethod
    def class_id(cls) -> URI:
        publisher, resource_name, version = _class_identity(cls)
        return URI("/" + "/".join(["lib", publisher, resource_name, version]))

def _to_opref(value: object) -> Optional[OpRef[Any]]:
    if hasattr(value, "op"):
        return value.op  # type: ignore[return-value]
    if isinstance(value, OpRef):
        return value
    return None


def _class_schema(cls: type["Library"]) -> dict:
    _, _, version = _class_identity(cls)
    deps = _class_dependencies(cls)
    return {
        "id": cls.class_id().path,
        "version": version,
        "dependencies": [dep.path for dep in deps],
    }


def _library_schema(library: "Library") -> dict:
    _, _, version = _class_identity(type(library))
    return {
        "id": library.id().path,
        "version": version,
        "dependencies": [dep.path for dep in library.dependencies],
    }


def compile_ir(library: Library | type[Library]) -> dict:
    library_cls = _library_class(library)
    _validate_library_class(library_cls)
    from .classdef import Class, class_definition

    declared_classes = getattr(library_cls, "classes", ()) or ()
    classes: list[dict[str, object]] = []
    for declared in declared_classes:
        if not isinstance(declared, type) or not issubclass(declared, Class):
            raise TypeError("Library classes must contain tc.Class subclasses")
        classes.append(class_definition(declared))
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

    return {"schema": _class_schema(library_cls), "classes": classes, "routes": routes}


def _compile_route(route: Route, library: type[Library]) -> object:
    compile_subject = _compile_self_instance(library)
    sig = inspect.signature(route.form)
    params = list(sig.parameters.values())
    if not params or params[0].name != "self":
        raise TypeError("route form must begin with a `self` parameter")

    if len(params) > 1:
        return _compile_opdef_route(route, compile_subject, params)

    result = route.form(compile_subject)
    if isinstance(result, (OpDef, Scalar, Collection)):
        return _compile_opdef_route(route, compile_subject, params)

    return result


def _compile_opdef_route(
    route: Route,
    library: Library,
    params: list[inspect.Parameter],
) -> OpDef:

    try:
        hints = get_type_hints(route.form, globalns=route.form.__globals__)
    except (NameError, TypeError):
        hints = {}

    injected_names = _INJECTED_ROUTE_PARAM_NAMES
    form = route.form
    uses_autograph = not any(param.name in injected_names for param in params[1:])
    if uses_autograph:
        form = _autograph.transform(route.form, source=route.source)

    arg_names = _route_arg_param_names(params, skip_first_self=True)

    with scoped_context() as cxt:
        args: list[Scalar] = []
        kwargs: dict[str, Scalar] = {}
        for idx, param in enumerate(params[1:], start=1):
            if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
                raise TypeError("variadic route parameters are not supported in TinyChain route compilation")
            if param.name in injected_names:
                if idx != 1:
                    raise TypeError(
                        f"reserved parameter '{param.name}' must be the first argument after self"
                    )
                injected = cxt
                if param.kind == param.KEYWORD_ONLY:
                    kwargs[param.name] = injected
                else:
                    args.append(injected)
                continue
            annotation = hints.get(param.name, param.annotation)
            placeholder = scalar_for_hint(param.name, _runtime_type_hint(annotation, default=Value))
            if param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = placeholder
            else:
                args.append(placeholder)
        result = form(library, *args, **kwargs)
        if cxt.form() and not isinstance(result, ContextResult):
            result = cxt.result(result)

    if isinstance(result, OpDef):
        result = _inline_opref_refs(result)
        _validate_opdef_method(route.method, result)
        _validate_opdef(result, _allowed_inputs_from_params(route, params))
        return result

    if isinstance(result, ContextResult):
        form = list(result.form)
        _append_context_result_form(route, form, result.result)
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
    arg_names = _route_arg_param_names(params, skip_first_self=True)

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
            node_form = form_of(node)
            if isinstance(node_form, TCRef):
                _validate_tcref(node_form, allowed, form_map)


def _inline_opref_refs(opdef: OpDef) -> OpDef:
    form_map = {name: scalar for name, scalar in opdef.form}

    def resolve_scalar(scalar: Scalar) -> Scalar:
        scalar_form = form_of(scalar)
        if isinstance(scalar_form, TCRef):
            ref_form = form_of(scalar_form)
            if isinstance(ref_form, Scalar) and not isinstance(ref_form, TCRef):
                op_scalar = ref_form
            else:
                op_scalar = None
        else:
            op_scalar = None

        if isinstance(op_scalar, Scalar):
            op_form = form_of(op_scalar)
            if isinstance(op_form, TCRef) and isinstance(form_of(op_form), IdRef):
                target_id = form_of(op_form)
                target = form_map.get(target_id.name)
                target_form = form_of(target) if target is not None else None
                if isinstance(target_form, TCRef) and isinstance(form_of(target_form), StateOpRef):
                    return Scalar._from_opref(TCRef(form_of(target_form)))

        if isinstance(scalar_form, OpDef):
            return Scalar(_inline_opref_refs(scalar_form))
        if isinstance(scalar_form, dict):
            return scalar_map_of({key: resolve_scalar(value) for key, value in scalar_form.items()})
        if isinstance(scalar_form, (list, tuple)):
            return scalar_tuple_of([resolve_scalar(value) for value in scalar_form])
        return scalar

    resolved_form = [(name, resolve_scalar(scalar)) for name, scalar in opdef.form]
    if isinstance(opdef, GetOpDef):
        return GetOpDef(opdef.key, resolved_form)
    if isinstance(opdef, PutOpDef):
        return PutOpDef(opdef.key, opdef.value, resolved_form)
    if isinstance(opdef, DeleteOpDef):
        return DeleteOpDef(opdef.key, resolved_form)
    if isinstance(opdef, PostOpDef):
        return PostOpDef(resolved_form)

    raise TypeError(f"unsupported OpDef type {type(opdef).__name__}")


def _validate_tcref(tcref, allowed: set[str], form_map: dict[str, Scalar]) -> None:
    ref_form = form_of(tcref)
    if isinstance(ref_form, IdRef):
        name = ref_form.name
        if name not in allowed:
            logging.info("OpDef depends on undefined id $%s", name)

    if isinstance(ref_form, StateOpRef):
        _validate_opref(ref_form)
    elif isinstance(ref_form, Scalar):
        resolved = _resolve_opref_ref(ref_form, form_map)
        if resolved is not None:
            _validate_opref(resolved)


def _validate_opref(opref: StateOpRef) -> None:
    if isinstance(opref, (GetOpRef, PutOpRef, PostOpRef, DeleteOpRef)):
        return

    raise ValueError(f"unsupported OpRef type {type(opref).__name__}")


def _resolve_opref_ref(value: Scalar, form_map: dict[str, Scalar]) -> StateOpRef | None:
    value_form = form_of(value)
    if isinstance(value_form, TCRef):
        ref_form = form_of(value_form)
        if isinstance(ref_form, StateOpRef):
            return ref_form
        if isinstance(ref_form, IdRef):
            target = form_map.get(ref_form.name)
            if target is not None and target is not value:
                return _resolve_opref_ref(target, form_map)
    return None


def _walk_scalars_with_opdef(root: Scalar):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node

        node_form = form_of(node)
        if isinstance(node_form, OpDef):
            for _, inner in node_form.form:
                stack.append(inner)
        if isinstance(node_form, dict):
            for value in reversed(list(node_form.values())):
                stack.append(value)
        if isinstance(node_form, (list, tuple)):
            for value in reversed(list(node_form)):
                stack.append(value)


def install(
    library: Library | type[Library],
    *,
    wasm: pathlib.Path | None = None,
    remote: object | None = None,
    kernel: Optional[object] = None,
    data_dir: Optional[pathlib.Path] = None,
    token: object | None = None,
) -> object:
    library_cls = _library_class(library)
    _validate_library_class(library_cls)
    # Identity is class-authoritative; the install definition is derived from
    # the class, so instance-level mutation cannot change what is installed.

    if wasm is not None:
        if remote is not None:
            raise ValueError("remote WASM installs are not supported by the canonical /lib install path")
        return _install_compiled_wasm_library(
            library,
            wasm,
            kernel=kernel,
            data_dir=data_dir,
            token=token,
        )

    if _bearer_token(token) is None and remote is None:
        raise ValueError("expected `token` for library installs")

    ir = compile_ir(library_cls)
    definition = ir if ir["classes"] else library_definition(library_cls)

    if remote is not None:
        return _submit_remote_library_definition(remote, definition, token=token)

    kernel = _kernel_for_library_install(kernel=kernel, data_dir=data_dir)
    bearer_token = _bearer_token(token)
    if bearer_token is None:
        raise ValueError("expected `token` for library installs")
    return _submit_local_library_definition(kernel, definition, bearer_token=bearer_token)


def library_definition(library: Library | type[Library]) -> dict:
    """Return the canonical v1-style JSON definition of a Library."""

    library_cls = _library_class(library)
    ir = compile_ir(library_cls)
    return {
        library_cls.class_id().path: {
            route["path"].strip("/"): route["opdef"]
            if "opdef" in route
            else route["op"]
            if "op" in route
            else route["value"]
            for route in ir["routes"]
        }
    }


def _submit_remote_library_definition(
    remote: object, definition: dict, *, token: object | None
) -> object:
    from .host import Host

    if isinstance(remote, Host):
        host = remote if token is None else Host(remote.__uri__.absolute(), token=token)
    else:
        host = Host(str(remote), token=token)
    return host.request("PUT", _LIB_ROOT_URI.path, body=definition)


def _kernel_for_library_install(
    *,
    kernel: object | None,
    data_dir: pathlib.Path | None,
    library: Library | type[Library] | None = None,
    token: object | None = None,
) -> object:
    if kernel is not None:
        return kernel

    if data_dir is None:
        raise ValueError("expected either `kernel` or `data_dir`")

    if library is not None:
        library_id = (
            library.id().path
            if isinstance(library, Library)
            else _library_class(library).class_id().path
        )
        from . import _local

        return _local.kernel_with_library_definition(
            json.dumps({library_id: {}}, separators=(",", ":")),
            token=token,
            data_dir=str(data_dir),
        )

    from . import _local

    return _local.local_kernel(data_dir=str(data_dir))


def _header_value(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    needle = name.lower()
    for key, value in headers:
        if str(key).lower() == needle:
            return str(value)
    return None


def _submit_local_library_definition(kernel: object, definition: dict, *, bearer_token: str) -> object:
    from . import _local

    install_path = _LIB_ROOT_URI.path
    body = json.dumps(definition, separators=(",", ":"))
    headers = [("authorization", f"Bearer {bearer_token}")]
    request = _local.kernel_request("PUT", install_path, headers, _local.state_handle(body))
    return kernel.dispatch(request)


def _read_wasm_b64(path: pathlib.Path) -> str:
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"WASM binary {path} is empty")
    return base64.b64encode(data).decode("ascii")


def _compiled_library_package_for_wasm(
    library: Library | type[Library], wasm_path: pathlib.Path
) -> dict:
    schema = (
        _library_schema(library)
        if isinstance(library, Library)
        else _class_schema(_library_class(library))
    )
    return {
        "schema": schema,
        "artifacts": [
            {
                "path": _LIB_WASM_URI.path,
                "content_type": "application/wasm",
                "bytes": _read_wasm_b64(wasm_path),
            }
        ],
    }


def _install_compiled_wasm_library(
    library: Library | type[Library],
    wasm_path: pathlib.Path,
    *,
    kernel: Optional[object] = None,
    data_dir: Optional[pathlib.Path] = None,
    token: object | None = None,
) -> object:
    bearer_token = _bearer_token(token)
    if bearer_token is None:
        raise ValueError("expected `token` for WASM installs")

    kernel = _kernel_for_library_install(
        kernel=kernel,
        data_dir=data_dir,
        library=library,
        token=token,
    )
    package = _compiled_library_package_for_wasm(library, wasm_path)
    return kernel.install_compiled_package(
        json.dumps(package, separators=(",", ":")),
        bearer_token,
    )
