"""Versioned Services with scalar methods and Chain-backed collection members."""

from .chain import Sync
from .library import Library, Route, _class_dependencies, self_subject
from .state import Collection, GetOpRef, autobox
from .uri import URI, _segment


class Service(Library):
    """Declare immutable members on the class; invoke them through a backend."""

    _root = URI("service")

    def __init__(self, **kwargs):
        if type(self) is Service:
            raise TypeError("Service must be subclassed")
        super().__init__(**kwargs)

    @classmethod
    def _compile_subject(cls, original, members=None):
        subject = cls.__new__(cls)
        subject._symbolic = True
        subject._initialize(members)
        return subject

    @classmethod
    def _dependencies(cls, members):
        return _class_dependencies(cls, members)

    def _route_subject(self, name):
        if getattr(self, "_symbolic", False):
            return self_subject(_segment("path", name))
        return super()._route_subject(name)

    @staticmethod
    def _encode_member(value):
        if isinstance(value, Sync):
            value = value.declaration()
        elif isinstance(value, Library):
            from .state.value import Link

            value = Link(value.link().absolute())
        return autobox(value).to_json()

    @classmethod
    def _members(cls):
        members = {}
        excluded = (
            set(vars(Library)) | set(vars(Service)) | Library._IDENTITY_FIELDS
            | {"dependencies", "authority", "classes"}
        )
        for base in reversed(cls.__mro__):
            if base in (object, Library, Service):
                continue
            if "__init__" in vars(base):
                raise TypeError("Service members must be declared on the class, not in __init__")
            for name, attr in vars(base).items():
                if name.startswith("_") or name in excluded:
                    continue
                members[name] = attr
        for name, attr in members.items():
            if isinstance(attr, Collection):
                raise TypeError(f"persistent member {name!r} must be wrapped in Sync")
            if not isinstance(attr, (Route, Sync, Library)):
                autobox(attr)
        return members

    def _initialize(self, members, authority=None):
        if members is None:
            members = type(self)._members()
        super()._initialize(members, authority)
        for name, attr in members.items():
            if isinstance(attr, Sync):
                setattr(self, name, attr._bind(self._route_subject(name)))
            elif not isinstance(attr, (Library, Route)):
                value = autobox(attr)
                form = GetOpRef(self._route_subject(name), None)
                setattr(self, name, type(value)(form))
