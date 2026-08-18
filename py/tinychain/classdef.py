from __future__ import annotations

import inspect

from . import opref as runtime_opref
from .state import IdRef, OpDef, Scalar, State, TCRef, autobox
from .state.scalar.refs import GetOpRef, PostOpRef
from .uri import URI, _segment, validate_resource_name


_CLASS_ROOT_URI = URI("class")
_METADATA = frozenset({"publisher", "resource_name", "version", "dependencies"})


class ClassError(ValueError):
    """An actionable Class authoring error."""


class InvalidClassParent(ClassError):
    pass


class MissingClassMember(ClassError, AttributeError):
    pass


class UnsupportedClassOverride(ClassError):
    pass


def _native_parent_id(parent: type[State]) -> URI:
    parent_uri = getattr(parent, "__uri__", None)
    if not isinstance(parent_uri, URI) or not str(parent_uri).startswith("/state/"):
        raise InvalidClassParent(
            "Class parent must be a user-defined Class or a native State type"
        )
    return parent_uri


class Class(State):
    """A canonical user-defined TinyChain Class.

    Define subclasses with class-level ``publisher``, ``resource_name``, and
    ``version`` metadata and an explicit native base, for example
    ``class User(tc.Class, tc.Map)``. Python subclassing a user-defined
    Class declares Class inheritance rather than granting dependency authority.
    Calling the subclass constructs a symbolic instance.
    """

    __uri__ = _CLASS_ROOT_URI
    publisher: str
    resource_name: str
    version: str
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        user_parents = [
            base for base in cls.__bases__ if isinstance(base, type) and issubclass(base, Class) and base is not Class
        ]
        native_parents = [
            base
            for base in cls.__bases__
            if isinstance(base, type)
            and issubclass(base, State)
            and not issubclass(base, Class)
        ]
        if len(user_parents) > 1:
            raise InvalidClassParent("multiple Class inheritance is not supported")
        if user_parents:
            if native_parents:
                raise InvalidClassParent(
                    "a derived Class cannot declare a second native parent"
                )
        else:
            if len(native_parents) != 1:
                raise InvalidClassParent(
                    "a base Class must inherit exactly one native State type"
                )
            _native_parent_id(native_parents[0])
        cls._validate_identity()
        cls._validate_overrides()

    @classmethod
    def _parent_class(cls) -> type[State]:
        parents = [
            base
            for base in cls.__bases__
            if isinstance(base, type) and issubclass(base, State) and base is not Class
        ]
        if len(parents) != 1:
            raise InvalidClassParent("Class must have exactly one declared parent")
        return parents[0]

    @classmethod
    def _validate_identity(cls) -> tuple[str, str, str]:
        publisher = getattr(cls, "publisher", None)
        resource_name = getattr(cls, "resource_name", None)
        version = getattr(cls, "version", None)
        if not publisher or not resource_name or not version:
            raise TypeError("Class requires class publisher, resource_name, and version")
        return (
            _segment("publisher", publisher),
            validate_resource_name(resource_name),
            _segment("version", version),
        )

    @classmethod
    def class_id(cls) -> URI:
        publisher, resource_name, version = cls._validate_identity()
        return URI(_CLASS_ROOT_URI, publisher, resource_name, version)

    @classmethod
    def parent_id(cls) -> URI:
        parent = cls._parent_class()
        if isinstance(parent, type) and issubclass(parent, Class):
            return parent.class_id()
        return _native_parent_id(parent)

    @classmethod
    def _declared_prototype(cls) -> dict[str, object]:
        from .library import Route

        prototype: dict[str, object] = {}
        for name, value in cls.__dict__.items():
            if name.startswith("_") or name in _METADATA:
                continue
            if isinstance(value, Route):
                prototype[name] = value.opdef(cls._self_placeholder())
            elif not inspect.isroutine(value) and not isinstance(value, (classmethod, staticmethod, property)):
                prototype[name] = autobox(value)
        return prototype

    @classmethod
    def _validate_overrides(cls) -> None:
        from .library import Route

        parent = cls._parent_class()
        if not isinstance(parent, type) or not issubclass(parent, Class):
            return
        inherited = parent.prototype(include_inherited=True)
        for name, value in cls.__dict__.items():
            if name not in inherited:
                continue
            old_is_method = isinstance(inherited[name], OpDef)
            new_is_method = isinstance(value, Route)
            if old_is_method != new_is_method:
                raise UnsupportedClassOverride(
                    f"Class member {name!r} cannot change between data and method"
                )

    @classmethod
    def prototype(cls, *, include_inherited: bool = False) -> dict[str, object]:
        own = cls._declared_prototype()
        parent = cls._parent_class()
        if include_inherited and isinstance(parent, type) and issubclass(parent, Class):
            inherited = parent.prototype(include_inherited=True)
            inherited.update(own)
            return inherited
        return own

    @classmethod
    def definition(cls) -> dict[str, object]:
        return {
            "id": str(cls.class_id()),
            "parent": str(cls.parent_id()),
            "prototype": {
                name: value.to_json() if hasattr(value, "to_json") else value
                for name, value in cls.prototype().items()
            },
        }

    @classmethod
    def _self_placeholder(cls) -> "Class":
        instance = object.__new__(cls)
        State.__init__(instance, TCRef(IdRef("self")))
        return instance

    def __init__(self, parent: object = None, /, **members: object) -> None:
        if isinstance(parent, TCRef) and not members:
            State.__init__(self, parent)
            return
        if parent is not None and members:
            raise TypeError("Class construction accepts a native parent or member keywords, not both")
        if parent is not None:
            State.__init__(self, TCRef(GetOpRef(type(self).class_id(), parent)))
        else:
            State.__init__(self, TCRef(PostOpRef(type(self).class_id(), members)))

    def id(self) -> URI:
        return type(self).class_id()

    def to_json(self) -> object:
        from .state.scalar import _json_of

        return _json_of(self._form)

    def __getattribute__(self, name: str) -> object:
        if not name.startswith("_") and name not in _METADATA:
            cls = type(self)
            for base in cls.__mro__:
                if base is Class:
                    break
                if name in base.__dict__:
                    value = base.__dict__[name]
                    from .library import Route

                    if not isinstance(value, Route) and not inspect.isroutine(value):
                        return self._get(
                            name,
                            rtype=type(autobox(value)),
                        )
                    break
        return super().__getattribute__(name)

    def _bind_class_route(self, route):
        from .library import _execute_route_result_if_needed, _route_body_from_call

        def bound(*args, **kwargs):
            body = _route_body_from_call(route, args, kwargs)
            path = URI(type(self).class_id(), route.name)
            if route.method == "GET":
                opref = runtime_opref.get(path)
            elif route.method == "POST":
                opref = runtime_opref.post(path)
            elif route.method == "PUT":
                opref = runtime_opref.put(path)
            elif route.method == "DELETE":
                opref = runtime_opref.delete(path)
            else:
                raise ValueError(f"unsupported route method {route.method}")
            if body is not None:
                opref = opref.with_body(body)
            rtype = route._return_type() or Scalar
            result = rtype(opref)
            return _execute_route_result_if_needed(result, body)

        bound.__name__ = route.name or route.form.__name__
        bound.__doc__ = route.form.__doc__
        bound.__signature__ = route._bound_signature()
        return bound

    def __getattr__(self, name: str) -> object:
        prototype = type(self).prototype(include_inherited=True)
        if name not in prototype:
            raise MissingClassMember(
                f"{type(self).__name__} has no Class member {name!r}"
            )
        value = prototype[name]
        if isinstance(value, OpDef):
            # Route descriptors normally handle methods; this is only reachable
            # for an inherited method hidden from Python descriptor lookup.
            route = next(
                getattr(base, name)
                for base in type(self).__mro__[1:]
                if name in base.__dict__
            )
            return self._bind_class_route(route)
        return self._get(name, rtype=type(value) if isinstance(value, State) else Scalar)


def class_definition(cls: type[Class]) -> dict[str, object]:
    if not isinstance(cls, type) or not issubclass(cls, Class) or cls is Class:
        raise TypeError("expected a user-defined Class")
    return cls.definition()


def validate_class_definition(value: object) -> dict[str, object]:
    """Validate and normalize a canonical language-neutral Class definition."""
    if not isinstance(value, dict) or set(value) != {"id", "parent", "prototype"}:
        raise ClassError("malformed Class definition: expected id, parent, and prototype")
    identity, parent, prototype = value["id"], value["parent"], value["prototype"]
    if not isinstance(identity, str) or not identity.startswith(f"{_CLASS_ROOT_URI}/"):
        raise ClassError("malformed Class definition: expected a canonical /class identity")
    if len(identity.strip("/").split("/")) != 4:
        raise ClassError("malformed Class definition: Class identity must be versioned")
    if not isinstance(parent, str) or not (
        parent.startswith(f"{_CLASS_ROOT_URI}/") or parent.startswith("/state/")
    ):
        raise InvalidClassParent("invalid Class parent identity")
    if not isinstance(prototype, dict) or not all(
        isinstance(name, str) and name and "/" not in name for name in prototype
    ):
        raise ClassError("malformed Class definition: invalid prototype")
    return {"id": identity, "parent": parent, "prototype": dict(prototype)}
