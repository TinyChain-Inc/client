# A persistent Service through local or HTTP execution

Use a rebuilt `tinychain_local` extension for local execution, or a running
authenticated TinyChain HTTP host for remote execution. In either case, `token`
is a signed bearer token authorized for `/service/example-devco/items/1.0.0`.
Use the existing authentication setup; the client never chooses transaction IDs.

```python
import tinychain as tc


class Items(tc.Service):
    publisher = "example-devco"
    resource_name = "items"
    version = "1.0.0"
    label = "items"
    data = tc.chain.Sync(tc.collection.BTree([
        ["key", "/state/scalar/value/number"],
    ]))

    @tc.get
    def count(self, key):
        return self.data.count()

    @tc.post
    def append(self, value):
        return self.data.insert(value)
```

For local execution, construct one kernel with explicit storage roots:

```python
from tinychain_local import KernelHandle

kernel = KernelHandle.local(
    data_dir="items-data", workspace="items-workspace", token=token,
)
items = Items()
tc.install(Items, kernel=kernel, token=token)

with tc.backend(kernel=kernel, token=token):
    items.append(value=[1])
    tc.execute(items.data.insert([2]))
    assert items.count(key=None) == 2
```

For HTTP execution, use the same definition and calls with an authority:

```python
authority = "http://127.0.0.1:8702"
items = Items(authority=tc.URI.parse(authority))
tc.install(Items, remote=authority, token=token)

with tc.backend(headers=[("authorization", f"Bearer {token.bearer_token}")]):
    items.append(value=[1])
    tc.execute(items.data.insert([2]))
    assert items.count(key=None) == 2
```

Run each example against fresh storage. To verify persistence, stop the local
Python process or remote host after the calls complete. Reopen with the same
storage roots and authority credentials, reconstruct `items`, and repeat the
count call without repeating the writes. The result remains `2`. Keep the
workspace's host-control records. Consume responses and collection streams before
ending a phase; response completion participates in commit.

Use deferred mode to inspect a method call without executing it:

```python
with tc.backend(mode="deferred"):
    pending = items.append(value=[3])
```

The [Service behavior document](../SERVICE_PARITY.md) describes the supported
port and debug runtime requirements. No client API selects commit or rollback.
