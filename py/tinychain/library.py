from __future__ import annotations

import json
from dataclasses import dataclass

from . import uri


@dataclass(frozen=True, slots=True)
class Library:
    publisher: str | None = None
    name: str | None = None
    version: str | None = None
    dependencies: tuple[uri.URI, ...] = ()
    authority: uri.URI | None = None

    def __post_init__(self) -> None:
        cls = type(self)
        publisher = self.publisher or getattr(cls, "publisher", None)
        name = self.name or getattr(cls, "name", None)
        version = self.version or getattr(cls, "version", None)
        if not publisher or not name or not version:
            raise TypeError("Library requires publisher, name, and version")
        object.__setattr__(self, "publisher", publisher)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

        if not self.dependencies:
            deps = getattr(cls, "dependencies", ())
            object.__setattr__(self, "dependencies", deps)

        if self.authority is None:
            object.__setattr__(self, "authority", getattr(cls, "authority", None))

    def id(self) -> uri.URI:
        return uri.URI(
            "/" + "/".join(
                [
                    "lib",
                    uri._segment("publisher", self.publisher),
                    uri._segment("name", self.name),
                    uri._segment("version", self.version),
                ]
            )
        )

    def link(self) -> uri.URI:
        base = self.id()
        if self.authority is None:
            return base
        return uri.URI(
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
    def class_id(cls) -> uri.URI:
        publisher = getattr(cls, "publisher", None)
        name = getattr(cls, "name", None)
        version = getattr(cls, "version", None)
        if not publisher or not name or not version:
            raise TypeError("Library requires publisher, name, and version")
        return uri.URI(
            "/" + "/".join(
                [
                    "lib",
                    uri._segment("publisher", publisher),
                    uri._segment("name", name),
                    uri._segment("version", version),
                ]
            )
        )

    @classmethod
    def class_schema(cls) -> dict:
        deps = getattr(cls, "dependencies", ())
        return {
            "id": cls.class_id().path,
            "version": getattr(cls, "version", None),
            "dependencies": [dep.path for dep in deps],
        }

    @classmethod
    def class_schema_json(cls) -> str:
        return json.dumps(cls.class_schema(), separators=(",", ":"))
