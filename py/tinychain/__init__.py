from __future__ import annotations

from .library import Library
from .executor import Executor, backend
from .executor import execute as _dispatch_execute
from .opref import OpRef
from .ref import Ref, String, Json
from .uri import URI
from . import define
from . import kernel
from . import uri
from . import testing
from . import wasm

# Convenience aliases: keep v1 ergonomics while keeping `tc.define.*` as the canonical home.
get = define.get
put = define.put
post = define.post
delete = define.delete

__all__ = [
    "Library",
    "Executor",
    "backend",
    "OpRef",
    "Ref",
    "String",
    "Json",
    "URI",
    "define",
    "kernel",
    "get",
    "put",
    "post",
    "delete",
    "uri",
    "testing",
    "wasm",
]


def execute(op: "OpRef | Ref") -> object:
    response = _dispatch_execute(op)
    status = getattr(response, "status", None)
    if status == 200:
        return testing.decode_json_body(response)
    if status == 204:
        return None
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
