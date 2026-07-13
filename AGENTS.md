# Client Agent Notes

## URI construction

* Build `URI`s via concatenation/composition helpers (e.g., base + path segments).
  Do not define repetitive string-constant paths throughout the client code.

## Operation selection rule

* Default to `GetOpRef` when an operation takes exactly one `Value` argument that identifies a resource id or range of ids.
* Do not add a parallel `PostOpRef` variant for that same operation unless the operation must accept a non-`Value` `State` payload.
* Use `PostOpRef` for structured/non-`Value` argument maps, `DeleteOpRef` for deletions, and `PutOpRef` for key/value replacement semantics.
