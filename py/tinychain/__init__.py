from __future__ import annotations

from .library import Library, delete, get, grad, install, post, put
from .codec import decode_response_body
from .executor import Executor, backend
from .executor import execute as _dispatch_execute
from .opref import OpRef
from . import opref
from .ref import Ref
from .state.tensor import Tensor, concatenate, einsum, split, tile
from .state.value import Bool, C64, C128, Complex, F32, F64, Float, I64, Integer, Link, Map, Null, Number, String, Tuple, U64
from .uri import URI, authority, origin, uri
from . import compute
from . import state
from . import kernel
from . import auth
from .cond import cond
from .after import after
from .host import Host

__all__ = [
    "Library",
    "Executor",
    "backend",
    "OpRef",
    "opref",
    "Ref",
    "Bool",
    "Null",
    "Link",
    "Number",
    "Integer",
    "I64",
    "U64",
    "Float",
    "F32",
    "F64",
    "Complex",
    "C64",
    "C128",
    "Map",
    "Tuple",
    "String",
    "Tensor",
    "concatenate",
    "einsum",
    "split",
    "tile",
    "URI",
    "compute",
    "state",
    "kernel",
    "install",
    "get",
    "grad",
    "put",
    "post",
    "delete",
    "cond",
    "after",
    "uri",
    "authority",
    "origin",
    "Host",
    "auth",
]

globals().pop("testing", None)
globals().pop("wasm", None)


def execute(op: "OpRef | Ref") -> object:
    if hasattr(op, "op"):
        op = op.op
    if not isinstance(op, (OpRef, Ref)):
        raise TypeError(f"expected OpRef or Ref, got {type(op).__name__}")

    response = _dispatch_execute(op)
    status = getattr(response, "status", None)
    if status is None:
        # HTTP-host execution paths may already return decoded JSON-like values.
        return response
    if status == 200:
        return decode_response_body(response)
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
    except (AttributeError, TypeError, ValueError):
        message = None
    if message:
        raise AssertionError(f"unexpected status {status}: {message}")
    raise AssertionError(f"unexpected status {status}")

# Optional local (PyO3) backend. When installed, re-export its public classes at the top level
# so `import tinychain as tc` works for both HTTP and in-process execution.
try:  # pragma: no cover
    import tinychain_local as local  # type: ignore

    KernelHandle = local.KernelHandle
    Backend = local.Backend
    KernelRequest = local.KernelRequest
    KernelResponse = local.KernelResponse
    StateHandle = local.StateHandle
    State = local.State
    LocalTensor = local.Tensor

    __all__ += [
        "local",
        "KernelHandle",
        "Backend",
        "KernelRequest",
        "KernelResponse",
        "StateHandle",
        "State",
        "LocalTensor",
    ]
    for _name in ("Scalar", "Collection", "Value"):
        if hasattr(local, _name):
            globals()[_name] = getattr(local, _name)
            __all__.append(_name)
except ImportError:  # pragma: no cover
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
    LocalTensor = _MissingBackend("LocalTensor")


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
        "LocalTensor",
        "Value",
    }:
        raise ImportError(
            f"`tinychain.{name}` requires the optional local backend. "
            "Install `tinychain-local` (or the equivalent extra) to enable PyO3 eager execution."
        )
    raise AttributeError(name)
