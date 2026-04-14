#!/usr/bin/env python3
"""
Framework-native eager vs deferred execution mode.

No package-level `deferred` kwargs are needed on route methods.
"""

from __future__ import annotations

import tinychain as tc


class _Body:
    def __init__(self, payload: object):
        self._payload = payload

    def value(self):
        class _Value:
            def __init__(self, payload: object):
                self._payload = payload

            def to_json(self) -> str:
                import json

                return json.dumps(self._payload)

        return _Value(self._payload)


class _Response:
    def __init__(self, payload: object):
        self.status = 200
        self.body = _Body(payload)


class _Kernel:
    def dispatch(self, request):
        if request.path.endswith("/hello"):
            return _Response("hello")
        raise ValueError(f"unexpected path {request.path}")


class Echo(tc.Library):
    publisher = "example-devco"
    name = "echo"
    version = "0.1.0"

    @tc.get
    def hello(self) -> tc.String:
        ...


def main() -> int:
    kernel = _Kernel()
    echo = Echo()

    with tc.backend(kernel, mode="eager"):
        eager_value = echo.hello()
        print("eager:", eager_value)  # "hello"

    with tc.backend(kernel, mode="deferred"):
        plan = echo.hello()
        print("deferred plan type:", type(plan).__name__)

    with tc.backend(kernel, mode="eager"):
        resolved = tc.execute(plan)
    print("resolved:", resolved)  # "hello"

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
