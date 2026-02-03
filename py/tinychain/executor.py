from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional


_current_executor: contextvars.ContextVar["Executor | None"] = contextvars.ContextVar(
    "tinychain_executor", default=None
)


def _headers_to_list(headers: Optional[Iterable[tuple[str, str]]]) -> list[tuple[str, str]]:
    return list(headers) if headers else []


def _encode_json_body(value: Any) -> "object":
    import tinychain as tc

    payload = json.dumps(value, separators=(",", ":"))
    return tc.StateHandle(payload)


@dataclass(slots=True)
class Executor:
    kernel: object
    bearer_token: Optional[str] = None
    headers: Optional[Iterable[tuple[str, str]]] = None
    _token: Optional[contextvars.Token["Executor | None"]] = None

    def __enter__(self) -> "Executor":
        self._token = _current_executor.set(self)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._token is not None:
            _current_executor.reset(self._token)
            self._token = None

    def _merge_headers(self, extra: Optional[Iterable[tuple[str, str]]]) -> list[tuple[str, str]]:
        merged = _headers_to_list(self.headers)
        merged.extend(_headers_to_list(extra))
        if self.bearer_token and not any(k.lower() == "authorization" for k, _ in merged):
            merged.append(("authorization", f"Bearer {self.bearer_token}"))
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

        if not hasattr(tc, "KernelRequest"):
            raise ImportError(
                "KernelRequest is not available; install `tinychain-local` to use the in-process executor"
            )

        request_body = None if body is None else (_encode_json_body(body) if not hasattr(body, "value") else body)
        request = tc.KernelRequest(method, path, self._merge_headers(headers), request_body)
        return self.kernel.dispatch(request)

    def execute(self, opref: "object") -> object:
        return execute(opref, executor=self)

    def resolve(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        bearer_token: Optional[str] = None,
    ) -> object:
        token = bearer_token or self.bearer_token

        method = method.upper()
        fn_name = f"resolve_{method.lower()}"
        fn = getattr(self.kernel, fn_name, None)
        if fn is None:
            raise NotImplementedError(f"kernel does not implement {fn_name}")

        if method in {"GET", "DELETE"}:
            encoded = _encode_body(body)
            if encoded is None:
                return fn(path, bearer_token=token)
            return fn(path, encoded, bearer_token=token)

        raise NotImplementedError(f"{fn_name} is not wired for method {method}")


def current() -> "Executor":
    executor = _current_executor.get()
    if executor is None:
        raise RuntimeError("no active TinyChain executor (use `with tc.backend(...):`)")
    return executor


def try_current() -> "Executor | None":
    return _current_executor.get()


def backend(kernel: object, *, bearer_token: Optional[str] = None) -> Executor:
    return Executor(kernel=kernel, bearer_token=bearer_token)


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


def _kernel_dispatch(kernel: object, method: str, path: str, headers, body) -> object:
    import tinychain as tc

    request = tc.KernelRequest(method, path, headers, body)
    return kernel.dispatch(request)


def _kernel_resolve(
    kernel: object, method: str, path: str, bearer_token: Optional[str], body: Any
) -> object:
    method = method.upper()
    fn_name = f"resolve_{method.lower()}"
    fn = getattr(kernel, fn_name, None)
    if fn is None:
        raise NotImplementedError(f"kernel does not implement {fn_name}")

    if method in {"GET", "DELETE"}:
        encoded = _encode_body(body)
        if encoded is None:
            return fn(path, bearer_token=bearer_token)
        return fn(path, encoded, bearer_token=bearer_token)

    raise NotImplementedError(f"{fn_name} is not wired for method {method}")


def execute(opref: "object", *, executor: "Executor | None" = None) -> object:
    from .opref import OpRef
    from .ref import Ref

    if isinstance(opref, Ref):
        opref = opref.op

    if not isinstance(opref, OpRef):
        raise TypeError(f"expected OpRef or Ref, got {type(opref).__name__}")

    exec_ctx = executor or current()
    headers = exec_ctx._merge_headers(opref.headers)

    # GETs are always resolved through the kernel's op resolver when available. This avoids
    # leaking "local vs remote" deployment details into per-method decorators and ensures
    # transaction lifetimes remain kernel-owned (resolve_get rolls back automatically).
    if opref.method.upper() == "GET" and hasattr(exec_ctx.kernel, "resolve_get"):
        return _kernel_resolve(
            exec_ctx.kernel, "GET", opref.path, exec_ctx.bearer_token, opref.body
        )

    return _kernel_dispatch(exec_ctx.kernel, opref.method, opref.path, headers, _encode_body(opref.body))
