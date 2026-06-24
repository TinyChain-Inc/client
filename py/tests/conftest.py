from __future__ import annotations

import os
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


@pytest.fixture(scope="session")
def tc_autodiff_route_root(tc_server_url: str) -> str:
    """Require installed OpDef-backed autodiff routes on the live tc-server."""
    route_root = os.environ.get("TC_AUTODIFF_ROUTE_ROOT", "/lib/std/autodiff/0.1.0")
    probe = f"{tc_server_url}{route_root.rstrip('/')}/add"
    try:
        response = requests.post(probe, data="{}", timeout=1)
    except requests.exceptions.ConnectionError:
        pytest.skip("tc-server is not running at localhost:8702")

    if response.status_code == 404:
        pytest.skip(
            f"autodiff OpDef-backed routes are not installed at {route_root}; "
            "install the autodiff tensor-op library or set TC_AUTODIFF_ROUTE_ROOT"
        )
    return route_root
