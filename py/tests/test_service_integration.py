from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import tinychain as tc


@pytest.mark.parametrize("transport", ["local", "http"])
def test_service_install_mutate_restart(tmp_path, transport):
    token = tc.auth.mint_rjwt_token(
        host="http://127.0.0.1:8702", actor_id="example-admin",
        libs=["/service/example-devco/btree/1.0.0", "/service/example-devco/table/1.0.0"],
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", ""),
    ]))
    for phase in ("write", "read"):
        if phase == "read":
            # Default transaction TTL (3s) plus clock-skew grace (3s): startup
            # expiry must coexist with the first new requests after recovery.
            time.sleep(6.2)
        result = subprocess.run(
            [sys.executable, "-m", "tests.service_pilot", str(tmp_path), transport, phase],
            input=json.dumps(asdict(token)), text=True, capture_output=True, env=env, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
