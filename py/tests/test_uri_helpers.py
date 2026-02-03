import tinychain as tc


def test_uri_library_builder():
    root = tc.uri.library(publisher="example-devco", name="math", version="1.2.3")
    assert root.path == "/lib/example-devco/math/1.2.3"

    route = tc.uri.library(
        publisher="example-devco",
        name="math",
        version="1.2.3",
        path=["add"],
    )
    assert route.path == "/lib/example-devco/math/1.2.3/add"


def test_uri_service_builder():
    root = tc.uri.service(
        publisher="example-devco",
        namespace="ml",
        name="trainer",
        version="0.1.0",
    )
    assert root == "/service/example-devco/ml/trainer/0.1.0"


def test_uri_state_builder():
    root = tc.uri.state(namespace="demo", path=["users"])
    assert root == "/state/demo/users"

    media = tc.uri.media(path=["images", "cats"])
    assert media == "/state/media/images/cats"


def test_uri_healthz_builder():
    assert tc.uri.healthz() == "/healthz"
