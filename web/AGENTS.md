# Web Client Agent Notes

## Serialization boundaries

- Browser and SSR code must consume canonical TinyChain client values and lazy
  streams directly. Do not JSON-round-trip values for cloning, equality,
  caching, normalization, route construction, or component handoff.
- The outermost remote transport owns TinyChain request encoding and response
  decoding exactly once. Components, loaders, hooks, and SSR adapters must not
  duplicate that work or build a second protocol client.
- HTML-safe server-to-browser embedding is a distinct browser boundary and may
  serialize once using the shared safe serializer. Treat the embedded result as
  boundary data; never feed it back into native or symbolic TinyChain logic as a
  substitute for the canonical value.
- Preserve response streams and async iteration end to end. Do not buffer a
  collection merely to serialize, inspect, or render metadata about it.
