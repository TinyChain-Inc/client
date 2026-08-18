# Python Class v1 → v2 parity

| v1 behavior | v2 status | v2 contract |
| --- | --- | --- |
| Python model subclass declares an object type | **Adapted** | `tc.Class` subclasses declare immutable, versioned `/class/{publisher}/{resource}/{version}` definitions. |
| Extend the nearest native state or model | **Retained** | Base Classes inherit one native state type; ordinary Python inheritance declares one user-defined parent. |
| Public class data becomes prototype data | **Retained** | Public non-metadata values are canonical prototype members. |
| Decorated instance methods become bound methods | **Adapted** | Existing route decorators compile prototype OpDefs and bind `self`; no Class-only decorators exist. |
| Calling a model constructs an instance | **Retained** | Keyword members construct a map-backed symbolic instance; one positional value constructs an instance over a native parent. |
| Attribute access and calls resolve inherited members | **Retained** | Python descriptors and state-subject references preserve runtime lookup without exposing a context or transaction. |
| Implicit identity from the Python class name | **Removed** | Publisher, resource name, and version are explicit stable metadata. |
| Mixed positional and keyword construction | **Removed** | Construction accepts either one native parent or member keywords. |
| Method/value kind-changing overrides | **Removed** | They fail at declaration with `UnsupportedClassOverride`. |
| Client-owned transaction/replication helpers | **Removed** | Installation, authorization, transactions, routing, and replication remain framework owned. |
| Dynamic models and multiple inheritance | **Deferred** | v2 supports one bounded Class inheritance chain. |
| Live instance materialization | **Deferred upstream** | Authoring emits manifests/plans; live views depend on the state/server object codec contract. |

## Canonical example

```python
import tinychain as tc

class Message(tc.Class, tc.Map):
    publisher = "example-devco"
    resource_name = "message"
    version = "1.0.0"
    prefix = "hello"

    @tc.get
    def render(self, name: tc.String) -> tc.String:
        return tc.String("hello, {{name}}").render(name=name)

class LoudMessage(Message):
    publisher = "example-devco"
    resource_name = "loud-message"
    version = "1.0.0"

    @tc.get
    def render(self, name: tc.String) -> tc.String:
        return tc.String("HELLO, {{name}}").render(name=name)

message = LoudMessage(prefix="HELLO")
with tc.backend(mode="deferred"):
    plan = message.render(name="Ada")

# The containing Library uses the only install lifecycle:
# tc.install(MessageBundle, remote=host, token=token)
```

Missing members, invalid parents, and unsupported override kinds raise exported
`ClassError` subclasses. Backend authorization and compatibility errors retain
the typed errors produced by the ordinary host adapters.
