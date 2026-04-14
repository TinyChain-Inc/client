from __future__ import annotations

from .library import Library
from .executor import Executor, backend
from .executor import execute as _dispatch_execute
from .opref import OpRef
from . import opref
from .ref import Ref, String
from .uri import URI, authority, origin, uri
from . import define
from . import compute
from . import state
from . import kernel
from . import testing
from . import wasm
from . import auth
from . import autograph
from .cond import cond
from .host import Host

# Convenience aliases: keep v1 ergonomics while keeping `tc.define.*` as the canonical home.
self_subject = define.self_subject
install = define.install
get = define.get
put = define.put
post = define.post
delete = define.delete

__all__ = [
    "Library",
    "Executor",
    "backend",
    "OpRef",
    "opref",
    "Ref",
    "String",
    "URI",
    "define",
    "compute",
    "state",
    "kernel",
    "self_subject",
    "install",
    "get",
    "put",
    "post",
    "delete",
    "cond",
    "uri",
    "authority",
    "origin",
    "Host",
    "testing",
    "wasm",
    "auth",
    "autograph",
]


def execute(op: "OpRef | Ref") -> object:
    if not isinstance(op, (OpRef, Ref)):
        # Backward-compatible no-op for values which may already have been
        # auto-executed in the active backend context.
        return op

    response = _dispatch_execute(op)
    status = getattr(response, "status", None)
    if status is None:
        # HTTP-host execution paths may already return decoded JSON-like values.
        return response
    if status == 200:
        return testing.decode_json_body(response)
    if status == 204:
        return None
    message = None
    try:
        body = getattr(response, "body", None)
        if body is not None:
            value = body.value()
            text = value.to_json() if hasattr(value, "to_json") else value
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", errors="replace")
            message = text if isinstance(text, str) else str(text)
    except Exception:
        message = None
    if message:
        raise AssertionError(f"unexpected status {status}: {message}")
    raise AssertionError(f"unexpected status {status}")

# Optional local (PyO3) backend. When installed, re-export its public classes at the top-level
# so user code can keep `import tinychain as tc` (v1 ergonomics) while opting into in-process speed.
try:  # pragma: no cover
    import tinychain_local as local  # type: ignore

    KernelHandle = local.KernelHandle
    Backend = local.Backend
    KernelRequest = local.KernelRequest
    KernelResponse = local.KernelResponse
    StateHandle = local.StateHandle
    State = local.State
    Scalar = local.Scalar
    Collection = local.Collection
    Tensor = local.Tensor
    Value = local.Value

    __all__ += [
        "local",
        "KernelHandle",
        "Backend",
        "KernelRequest",
        "KernelResponse",
        "StateHandle",
        "State",
        "Scalar",
        "Collection",
        "Tensor",
        "Value",
    ]
except Exception:  # pragma: no cover
    local = None

    class _MissingBackend:
        def __init__(self, name: str) -> None:
            self._name = name

        def __getattr__(self, _attr: str):
            raise ImportError(
                f"`tinychain.{self._name}` requires the optional local backend. "
                "Install `tinychain-local` (or the equivalent extra) to enable PyO3 eager execution."
            )

        def __call__(self, *args, **kwargs):
            raise ImportError(
                f"`tinychain.{self._name}` requires the optional local backend. "
                "Install `tinychain-local` (or the equivalent extra) to enable PyO3 eager execution."
            )

        def __repr__(self) -> str:
            return f"<missing tinychain-local: {self._name}>"

    KernelHandle = _MissingBackend("KernelHandle")
    Backend = _MissingBackend("Backend")
    KernelRequest = _MissingBackend("KernelRequest")
    KernelResponse = _MissingBackend("KernelResponse")
    StateHandle = _MissingBackend("StateHandle")
    State = _MissingBackend("State")
    Scalar = _MissingBackend("Scalar")
    Collection = _MissingBackend("Collection")
    Tensor = _MissingBackend("Tensor")
    Value = _MissingBackend("Value")


def __getattr__(name: str):  # pragma: no cover
    if name in {
        "KernelHandle",
        "Backend",
        "KernelRequest",
        "KernelResponse",
        "StateHandle",
        "State",
        "Scalar",
        "Collection",
        "Tensor",
        "Value",
    }:
        raise ImportError(
            f"`tinychain.{name}` requires the optional local backend. "
            "Install `tinychain-local` (or the equivalent extra) to enable PyO3 eager execution."
        )
    raise AttributeError(name)
