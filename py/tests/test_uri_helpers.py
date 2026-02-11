import tinychain as tc


def test_uri_library_builder():
    root = tc.uri("lib", "example-devco", "math", "1.2.3")
    assert root.path == "/lib/example-devco/math/1.2.3"

    route = tc.uri("lib", "example-devco", "math", "1.2.3", "add")
    assert route.path == "/lib/example-devco/math/1.2.3/add"


def test_uri_service_builder():
    root = tc.uri("service", "example-devco", "ml", "trainer", "0.1.0")
    assert root.path == "/service/example-devco/ml/trainer/0.1.0"


def test_uri_state_builder():
    root = tc.uri("state", "demo", "users")
    assert root.path == "/state/demo/users"

    media = tc.uri("state", "media", "images", "cats")
    assert media.path == "/state/media/images/cats"


def test_uri_healthz_builder():
    assert tc.uri("healthz").path == "/healthz"
