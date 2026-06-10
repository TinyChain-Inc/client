from __future__ import annotations

import json
import contextvars
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, cast

from .uri import URI

_current_executor: contextvars.ContextVar["Executor | None"] = contextvars.ContextVar(
    "tinychain_executor", default=None
)


def _headers_to_list(headers: Optional[Iterable[tuple[str, str]]]) -> list[tuple[str, str]]:
    return list(headers) if headers else []


class RequestTarget(Protocol):
    def request(
        self,
        method: str,
        path: object,
        *,
        body: object | None = None,
        headers: Optional[Iterable[tuple[str, str]]] = None,
    ) -> object:
        ...


def _is_request_target(value: object) -> bool:
    return callable(getattr(value, "request", None))


def _as_request_target(value: object) -> RequestTarget:
    from .host import Host

    if isinstance(value, str):
        return Host(value)

    if _is_request_target(value):
        return cast(RequestTarget, value)

    raise TypeError(
        "remote execution target must be a host address string or an object with a request(...) method"
    )


def _path_from_opref(op_path: object) -> str:
    path = str(op_path)
    if not path:
        raise ValueError("OpRef path must be non-empty")
    return path


@dataclass(frozen=True, slots=True)
class _RemoteRoute:
    target: RequestTarget


def _encode_json_body(value: Any) -> "object":
    import tinychain as tc

    payload = json.dumps(_encode_payload(value), separators=(",", ":")).encode("utf-8")
    try:
        return tc.StateHandle(payload)
    except ImportError:
        return payload


def _encode_payload(value: Any) -> Any:
    from .state import (
        IdRef,
        OpDef,
        OpRef as StateOpRef,
        Scalar,
        TCRef,
        While,
        autobox,
    )
    from .state.value import Value
    from .uri import URI

    if hasattr(value, "__tc_route__") and hasattr(value, "__tc_instance__"):
        route = value.__tc_route__
        instance = value.__tc_instance__
        return autobox(route.opdef(instance)).to_json()

    if isinstance(
        value,
        (
            Scalar,
            Value,
            TCRef,
            While,
            IdRef,
            OpDef,
            StateOpRef,
            URI,
        ),
    ):
        return autobox(value).to_json()

    if isinstance(value, (list, tuple)):
        return [_encode_payload(item) for item in value]

    if isinstance(value, dict):
        return {key: _encode_payload(item) for key, item in value.items()}

    if hasattr(value, "to_json"):
        return value.to_json()

    return value


@dataclass(slots=True)
class Executor:
    kernel: object | None = None
    bearer_token: Optional[str] = None
    headers: Optional[Iterable[tuple[str, str]]] = None
    mode: str = "eager"
    _token: Optional[contextvars.Token["Executor | None"]] = None

    def __enter__(self) -> "Executor":
        self._token = _current_executor.set(self)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._token is not None:
            _current_executor.reset(self._token)
            self._token = None

    def is_eager(self) -> bool:
        return self.mode == "eager"

    def _merge_headers(self, extra: Optional[Iterable[tuple[str, str]]]) -> list[tuple[str, str]]:
        merged = _headers_to_list(self.headers)
        merged.extend(_headers_to_list(extra))
        if self.bearer_token and not any(k.lower() == "authorization" for k, _ in merged):
            merged.append(("authorization", f"Bearer {self.bearer_token}"))
        return merged

    def _merge_remote_headers(
        self,
        extra: Optional[Iterable[tuple[str, str]]],
    ) -> list[tuple[str, str]]:
        merged = _headers_to_list(self.headers)
        merged.extend(_headers_to_list(extra))
        return merged

    def dispatch(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Iterable[tuple[str, str]]] = None,
        body: Any = None,
    ) -> object:
        import tinychain as tc

        if self.kernel is None:
            raise RuntimeError("no local kernel configured for dispatch; set `kernel=` in tc.backend(...)")

        if not hasattr(tc, "KernelRequest"):
            raise ImportError(
                "KernelRequest is not available; install `tinychain-local` to use the in-process executor"
            )

        request_body = None if body is None else (_encode_json_body(body) if not hasattr(body, "value") else body)
        request = tc.KernelRequest(method, path, self._merge_headers(headers), request_body)
        return self.kernel.dispatch(request)

    def execute(self, opref: "object") -> object:
        return execute(opref, executor=self)

    def _route_remote(self, path: str) -> _RemoteRoute | None:
        if "://" in path:
            parsed = URI.parse(path)
            authority = parsed.authority()
            assert authority is not None
            origin = f"{parsed.scheme}://{authority}"
            return _RemoteRoute(target=_as_request_target(origin))

        return None


def current() -> "Executor":
    executor = _current_executor.get()
    if executor is None:
        raise RuntimeError("no active TinyChain executor (use `with tc.backend(...):`)")
    return executor


def try_current() -> "Executor | None":
    return _current_executor.get()


def backend(
    kernel: object | None = None,
    *,
    token: object | None = None,
    headers: Optional[Iterable[tuple[str, str]]] = None,
    mode: str = "eager",
) -> Executor:
    if mode not in {"eager", "deferred"}:
        raise ValueError("backend mode must be 'eager' or 'deferred'")

    from .auth import bearer_token

    return Executor(
        kernel=kernel,
        bearer_token=bearer_token(token),
        headers=headers,
        mode=mode,
    )


def _is_state_handle(obj: object) -> bool:
    return hasattr(obj, "value")


def _as_headers(value: object) -> Optional[Iterable[tuple[str, str]]]:
    if value is None:
        return None
    return value  # type: ignore[return-value]


def _encode_body(body: Any) -> "object":
    if body is None or _is_state_handle(body):
        return body
    return _encode_json_body(body)


def _encode_dispatch_body(body: Any) -> "object":
    if body is None or _is_state_handle(body):
        return body

    try:
        import tinychain as tc
    except ImportError:
        return _encode_body(body)

    if isinstance(body, (bytes, bytearray)):
        try:
            return tc.StateHandle(bytes(body))
        except ImportError:
            return _encode_body(body)
    if isinstance(body, str):
        try:
            return tc.StateHandle(body)
        except ImportError:
            return _encode_body(body)

    return _encode_body(body)


def _kernel_dispatch(kernel: object, method: str, path: str, headers, body) -> object:
    import tinychain as tc

    request = tc.KernelRequest(method, path, headers, body)
    return kernel.dispatch(request)


def _default_local_kernel() -> object | None:
    try:
        import tinychain as tc
    except ImportError:
        return None

    try:
        kernel_handle = getattr(tc, "KernelHandle", None)
        local_ctor = getattr(kernel_handle, "local", None)
        if callable(local_ctor):
            return local_ctor()
    except ImportError:
        return None

    return None


def _default_executor_for_path(path: str) -> Executor:
    if "://" in path:
        return Executor()

    kernel = _default_local_kernel()
    if kernel is None:
        raise RuntimeError(
            "no active TinyChain executor and no default local TinyChain host is available; "
            "install/initialize `tinychain-local` or use `with tc.backend(...)`"
        )
    return Executor(kernel=kernel)


def execute(opref: "object", *, executor: "Executor | None" = None) -> object:
    from .opref import OpRef
    from .ref import Ref

    if hasattr(opref, "op"):
        opref = opref.op

    if not isinstance(opref, OpRef):
        raise TypeError(f"expected OpRef or Ref, got {type(opref).__name__}")

    path = _path_from_opref(opref.path)
    exec_ctx = executor or try_current()
    if exec_ctx is None:
        exec_ctx = _default_executor_for_path(path)

    headers = exec_ctx._merge_headers(opref.headers)
    remote_route = exec_ctx._route_remote(path)

    if remote_route is not None:
        remote_headers = exec_ctx._merge_remote_headers(opref.headers)
        # Headers already include merged auth and contextual metadata.
        return remote_route.target.request(
            opref.method,
            path,
            body=opref.body,
            headers=remote_headers,
        )

    if exec_ctx.kernel is None:
        raise RuntimeError(
            f"no local kernel configured and no remote route matched {path}; set `kernel=` in tc.backend(...) or declare an authority on the Library"
        )

    method = opref.method.upper()
    if not hasattr(exec_ctx.kernel, "dispatch"):
        raise NotImplementedError("kernel does not implement dispatch")

    if method == "GET":
        return _kernel_dispatch(
            exec_ctx.kernel,
            method,
            path,
            headers,
            _encode_dispatch_body(opref.body),
        )
    return _kernel_dispatch(exec_ctx.kernel, method, path, headers, _encode_body(opref.body))
