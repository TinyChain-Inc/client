# Client Agent Notes

## Serialization boundaries

* Client symbolic and runtime code must exchange canonical typed values directly.
  Serialization is allowed only at an actual host/process boundary, a stable
  storage or sandbox ABI boundary, or explicit foreign-runtime materialization.
* Encode once in the outermost transport adapter and decode once when its response
  enters the client. Do not encode again in route wrappers, manifests, symbolic
  graph builders, local executors, hooks, caches, or UI adapters.
* Never use JSON or another wire form to implement equality, hashing, cloning,
  reference or URI construction, type inspection, validation, routing, or local
  delegation. Traverse or compare the canonical form directly and delegate leaf
  behavior to its owning type.
* Preserve lazy streams and native handles. Do not materialize a collection merely
  to serialize it, inspect it, or pass it to another colocated operation.

## URI construction

* Build `URI`s via concatenation/composition helpers (e.g., base + path segments).
  Do not define repetitive string-constant paths throughout the client code.

## Backpressure

* Preserve lazy transport and collection iteration across every client. Request
  concurrency, prefetch, and retries must be finite and explicit; surface
  structured saturation rather than buffering or retrying without a bound.
