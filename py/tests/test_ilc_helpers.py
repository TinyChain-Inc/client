import ilc


def test_ilc_library_schema_and_oprefs():
    lib = ilc.ILCLibrary(server="127.0.0.1:8700", version="0.1.0")

    schema = lib.schema()
    assert schema["id"] == ilc.ILC_CLIENT_PATH
    assert schema["version"] == "0.1.0"
    assert schema["dependencies"] == [ilc.ILC_SERVER_PATH]

    dep = lib.dependency
    assert dep.host == "127.0.0.1"
    assert dep.port == 8700

    add = lib.opref_add({"metric": [3, 5], "lhs": [1.0, 2.0], "rhs": [3.0, 4.0]})
    mul = lib.opref_mul({"metric": [3, 5], "lhs": [1.0, 2.0], "rhs": [3.0, 4.0]})
    encrypt = lib.opref_encrypt({"key": "k", "plain": [1, 2]})
    decrypt = lib.opref_decrypt({"key": "k", "cipher": [1, 2]})

    assert add.method == "POST"
    assert add.path == f"{ilc.ILC_CLIENT_PATH}/cipher/add"
    assert mul.path == f"{ilc.ILC_CLIENT_PATH}/cipher/mul"
    assert encrypt.path == f"{ilc.ILC_SERVER_PATH}/crypto/encrypt"
    assert decrypt.path == f"{ilc.ILC_SERVER_PATH}/crypto/decrypt"
