from __future__ import annotations

import json
from dataclasses import dataclass

from . import uri


@dataclass(frozen=True, slots=True)
class Library:
    publisher: str
    name: str
    version: str
    dependencies: tuple[uri.URI, ...] = ()
    authority: uri.URI | None = None

    def id(self) -> uri.URI:
        return uri.library(
            publisher=self.publisher,
            name=self.name,
            version=self.version,
        )

    def route(self, *path: str) -> str:
        return uri.library(
            publisher=self.publisher,
            name=self.name,
            version=self.version,
            path=list(path) if path else None,
        ).path

    def link(self) -> uri.URI:
        return uri.library_link(
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
