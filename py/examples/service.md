# A persistent Service through local or HTTP execution

Use a rebuilt `tinychain_local` extension for local execution, or a running
authenticated TinyChain HTTP host for remote execution. In either case, `token`
is a signed bearer token authorized for `Items.class_id()` below. Use the existing
authentication setup; the client never chooses transaction IDs.

```python
import tinychain as tc


class Items(tc.Service):
    publisher = "example-devco"
    resource_name = "items"
    version = "1.0.0"
    label = "items"
    data = tc.chain.Sync(tc.collection.BTree([
        ["key", str(tc.URI(tc.state.value.Number))],
    ]))

    @tc.get
    def count(self, key):
        return self.data.count()

    @tc.post
    def append(self, value):
        return self.data.insert(value)
```

`Sync` declares a Chain-owned member. Both the method call and direct member
write below reach that Chain; returning a read view does not grant write authority.
See the [selection-write examples](../README.md#mental-model) for Table and Tensor.

For local execution, construct one kernel with explicit storage roots and drain
it before leaving the context:

```python
from tinychain_local import KernelHandle

with KernelHandle.local(
    data_dir="items-data", workspace="items-workspace", token=token,
) as kernel:
    items = Items()
    tc.install(Items, kernel=kernel, token=token)

    with tc.backend(kernel=kernel, token=token):
        items.append(value=[1])
        tc.execute(items.data.insert([2]))
        assert items.count(key=None) == 2

    # Consume or drop any retained native responses and streams before this exit.
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

Run each example against fresh storage. To verify graceful reopening, leave the
local kernel context or send SIGTERM/SIGINT to the HTTP host and wait for it to
exit. Consume or drop responses and collection streams before shutdown. Reopen
with the same storage roots and authority credentials, reconstruct `items`, and
repeat the count call without repeating the writes; the result remains `2`.
Preserve the workspace's host-control records.

Execution commits before projection; abandoning a response does not undo its
transaction. SIGKILL or an expired supervisor grace period is abrupt termination,
not graceful drain. Abrupt-restart recovery has separate host tests; this example
does not establish a power-loss guarantee.

Use deferred mode to inspect a method call without executing it:

```python
with tc.backend(mode="deferred"):
    pending = items.append(value=[3])
```

The [Service behavior document](../SERVICE_PARITY.md) describes authoring limits
and validation. No client API selects commit or rollback.
