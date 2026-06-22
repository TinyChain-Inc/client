from __future__ import annotations

import pytest
import requests

TC_SERVER_HOST = "http://localhost:8702"


@pytest.fixture(scope="session")
def tc_server_url() -> str:
    """Skip the test session if tc-server is not reachable at localhost:8702."""
    try:
        requests.get(f"{TC_SERVER_HOST}/healthz", timeout=1)
    except requests.exceptions.ConnectionError:
        pytest.skip("tc-server is not running at localhost:8702")
    return TC_SERVER_HOST
