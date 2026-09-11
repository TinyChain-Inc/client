from __future__ import annotations

import pathlib
import pytest

import tinychain as tc

from .support import require_tinychain_local


ACTOR_ID = "example-admin"


class A(tc.Library):
    publisher = "example-devco"
    resource_name = "a"
    version = "0.1.0"
    dependencies = ()

    @tc.get
    def auth_context(self) -> tc.Ref:
        return tc.auth.context()


def test_framework_auth_context_available_in_local_and_native_routes(tmp_path: pathlib.Path):
    _, _ = require_tinychain_local(require_library_definition=True)

    secret_key_b64 = tc.auth.generate_actor_secret(ACTOR_ID)
    a = A()
    a_root = tc.URI(a)
    host_link = "http://127.0.0.1:8702"
    token = tc.auth.mint_rjwt_token(
        host=host_link,
        actor_id=ACTOR_ID,
        libs=[a_root],
        ttl_secs=300,
        secret_key_b64=secret_key_b64,
    )

    data_dir = tmp_path / "tc-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    kernel = tc.kernel.with_library(a, data_dir=data_dir, token=token)

    with tc.backend(kernel, token=token):
        direct_ctx = tc.execute(tc.auth.context())
        native_ctx = a.auth_context()
        assert isinstance(direct_ctx, dict)
        assert isinstance(native_ctx, dict)
        assert direct_ctx["principal"].endswith(f"::{ACTOR_ID}")
        assert native_ctx["principal"] == direct_ctx["principal"]

        with tc.backend(kernel, token=token, mode="deferred"):
            deferred_ctx = a.auth_context()

        assert isinstance(deferred_ctx, tc.Ref)
        assert tc.execute(deferred_ctx)["principal"] == direct_ctx["principal"]

    with tc.backend(kernel):
        with pytest.raises(RuntimeError, match="missing authenticated request context"):
            tc.execute(tc.auth.context())
