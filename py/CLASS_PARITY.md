# Python Class Authoring Contract

Python `tc.Class` mirrors the native Class value while retaining explicit v2
application identity.

## Stable behavior

- A concrete Class declares `publisher`, `resource_name`, and semantic
  `version`; its Python class name is not public identity.
- One Python base Class declares one user-defined parent. A native TinyChain
  state base declares the native parent. Dynamic multiple inheritance is not
  supported.
- Public immutable class values become prototype members.
- Existing route decorators compile instance methods into prototype `OpDef`s;
  there are no Class-specific verbs or decorators.
- Calling a Class with one positional native value constructs an instance over
  that parent. Calling it with keyword members constructs a map-backed instance.
- Member lookup follows instance members, the concrete prototype, inherited
  prototypes, and native-parent behavior. `$self` is the complete instance.
- Overrides which change an inherited member between incompatible value and
  method roles fail at declaration.

Classes declared by a Library are installed first through the same ordinary
`/class` PUT contract as direct Class definitions. The Library literal then
contains ordinary links to them; it does not embed a Class envelope. Instances
are ordinary State values and may be stored by collections; there is no
instance registry or Class-owned transaction lifecycle.

```python
import tinychain as tc

class Message(tc.Class, tc.Map):
    publisher = "example-devco"
    resource_name = "message"
    version = "1.0.0"

    @tc.get
    def render(self, name: tc.String) -> tc.String:
        return tc.String("hello, {{name}}").render(name=name)

message = Message(prefix="hello")
with tc.backend(mode="deferred"):
    plan = message.render(name="Ada")
```

Invalid parents, missing members, and unsupported overrides raise the exported
`ClassError` subclasses. Authorization, persistence, routing, transactions, and
replication remain host responsibilities and are not duplicated in Python.
