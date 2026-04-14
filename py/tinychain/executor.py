from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, cast

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
        bearer_token: Optional[str] = None,
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


def _normalize_route_prefix(prefix: object) -> str:
    raw = str(prefix).strip()
    if not raw:
        raise ValueError("remote route prefix must be non-empty")

    if "://" in raw:
        parsed = URI.parse(raw)
    else:
        normalized = raw if raw.startswith("/") else "/" + raw.lstrip("/")
        parsed = URI.parse(normalized)

    if parsed.host is not None:
        raise ValueError("remote route prefix must be a canonical path, not an absolute URI")

    path = parsed.path or "/"
    if path == "/":
        return "/"

    return path.rstrip("/")


def _matches_route(path: str, prefix: str) -> bool:
    if prefix == "/":
        return True
    if path == prefix:
        return True
    return path.startswith(prefix + "/")


def _path_from_opref(op_path: object) -> str:
    path = str(op_path)
    if not path:
        raise ValueError("OpRef path must be non-empty")
    return path


def _path_for_prefix_matching(path: str) -> str:
    if "://" not in path:
        return path

    return URI.parse(path).path


@dataclass(frozen=True, slots=True)
class _RemoteSelector:
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class _RemoteRoute:
    target: RequestTarget


def _normalize_remote_selector(selector: object) -> _RemoteSelector:
    raw = str(selector).strip()
    if not raw:
        raise ValueError("remote selector must be non-empty")

    if "://" in raw:
        parsed = URI.parse(raw)
        authority = parsed.authority()
        if authority is None:
            raise ValueError("remote authority selector must include scheme and authority")
        return _RemoteSelector("authority", f"{parsed.scheme}://{authority}")

    return _RemoteSelector("prefix", _normalize_route_prefix(raw))


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
        IfRef,
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
            IfRef,
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
    remote: str | RequestTarget | None = None
    remotes: Mapping[object, str | RequestTarget] | None = None
    mode: str = "eager"
    _token: Optional[contextvars.Token["Executor | None"]] = None

    def __enter__(self) -> "Executor":
        self._token = _current_executor.set(self)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._token is not None:
            _current_executor.reset(self._token)
            self._token = None

    def should_auto_execute(self) -> bool:
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

    def resolve(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        bearer_token: Optional[str] = None,
    ) -> object:
        if self.kernel is None:
            raise RuntimeError("no local kernel configured for resolve; set `kernel=` in tc.backend(...)")

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

    def _route_remote(self, path: str) -> _RemoteRoute | None:
        if "://" in path:
            parsed = URI.parse(path)
            authority = parsed.authority()
            assert authority is not None
            origin = f"{parsed.scheme}://{authority}"
            if self.remotes:
                for raw_selector, target in self.remotes.items():
                    selector = _normalize_remote_selector(raw_selector)
                    if selector.kind == "authority" and selector.value == origin:
                        return _RemoteRoute(target=_as_request_target(target))
            return _RemoteRoute(target=_as_request_target(origin))

        if self.remotes:
            match_path = _path_for_prefix_matching(path)
            best: tuple[int, _RemoteRoute] | None = None
            for raw_prefix, target in self.remotes.items():
                selector = _normalize_remote_selector(raw_prefix)
                if selector.kind != "prefix":
                    continue

                if _matches_route(match_path, selector.value):
                    request_target = _as_request_target(target)
                    score = len(selector.value)
                    if best is None or score > best[0]:
                        best = (
                            score,
                            _RemoteRoute(target=request_target),
                        )

            if best is not None:
                return best[1]

        if self.remote is not None:
            return _RemoteRoute(target=_as_request_target(self.remote))

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
    bearer_token: Optional[str] = None,
    headers: Optional[Iterable[tuple[str, str]]] = None,
    remote: str | RequestTarget | None = None,
    remotes: Mapping[object, str | RequestTarget] | None = None,
    mode: str = "eager",
    auto_execute: Optional[bool] = None,
) -> Executor:
    if mode not in {"eager", "deferred"}:
        raise ValueError("backend mode must be 'eager' or 'deferred'")

    if auto_execute is not None:
        compat_mode = "eager" if auto_execute else "deferred"
        if mode != "eager" and mode != compat_mode:
            raise ValueError("conflicting backend mode and auto_execute arguments")
        mode = compat_mode

    return Executor(
        kernel=kernel,
        bearer_token=bearer_token,
        headers=headers,
        remote=remote,
        remotes=remotes,
        mode=mode,
    )


def _is_state_handle(obj: object) -> bool:
    return hasattr(obj, "value")


def _as_headers(value: object) -> Optional[Iterable[tuple[str, str]]]:
    if value is None:
        return None
    return value  # type: ignore[return-value]


def _bearer_from_headers(headers: Iterable[tuple[str, str]]) -> Optional[str]:
    for key, value in headers:
        if key.lower() != "authorization":
            continue
        parts = value.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    return None


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


def _is_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text and "not found" in text


def _is_invalid_bearer_error(exc: Exception) -> bool:
    return "invalid bearer token" in str(exc).lower()


def execute(opref: "object", *, executor: "Executor | None" = None) -> object:
    from .opref import OpRef
    from .ref import Ref

    if isinstance(opref, Ref):
        opref = opref.op

    if not isinstance(opref, OpRef):
        raise TypeError(f"expected OpRef or Ref, got {type(opref).__name__}")

    path = _path_from_opref(opref.path)
    exec_ctx = executor or try_current()
    if exec_ctx is None:
        raise RuntimeError(
            "no active TinyChain executor; use `with tc.backend(...)` to run requests"
        )

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
            bearer_token=None,
        )

    if exec_ctx.kernel is None:
        raise RuntimeError(
            f"no local kernel configured and no remote route matched {path}; set `kernel=` or `remote=`/`remotes=` in tc.backend(...)"
        )

    method = opref.method.upper()
    if hasattr(exec_ctx.kernel, "dispatch"):
        if method == "GET":
            return _kernel_dispatch(
                exec_ctx.kernel,
                method,
                path,
                headers,
                _encode_dispatch_body(opref.body),
            )
        return _kernel_dispatch(exec_ctx.kernel, method, path, headers, _encode_body(opref.body))

    if method in {"GET", "DELETE"}:
        bearer = _bearer_from_headers(headers) or exec_ctx.bearer_token
        return _kernel_resolve(exec_ctx.kernel, method, path, bearer, opref.body)

    raise NotImplementedError("kernel does not implement dispatch for non-GET/DELETE routes")
