import tinychain as tc


def test_uri_library_builder():
    root = tc.URI("lib", "example-devco", "math", "1.2.3")
    assert root.path == "/lib/example-devco/math/1.2.3"

    route = tc.URI("lib", "example-devco", "math", "1.2.3", "add")
    assert route.path == "/lib/example-devco/math/1.2.3/add"


def test_uri_service_builder():
    root = tc.URI("service", "example-devco", "ml", "trainer", "0.1.0")
    assert root.path == "/service/example-devco/ml/trainer/0.1.0"


def test_uri_state_builder():
    root = tc.URI("state", "demo", "users")
    assert root.path == "/state/demo/users"

    media = tc.URI("state", "media", "images", "cats")
    assert media.path == "/state/media/images/cats"


def test_uri_healthz_builder():
    assert tc.URI("healthz").path == "/healthz"


def test_authority_and_origin_helpers():
    assert tc.authority("127.0.0.1:8702") == "127.0.0.1:8702"
    assert tc.authority("http://127.0.0.1:8702/lib/example-devco/x/0.1.0") == "127.0.0.1:8702"
    assert tc.origin("127.0.0.1:8702") == "http://127.0.0.1:8702"
    assert tc.origin("https://api.example.com:443/lib/x") == "https://api.example.com:443"
