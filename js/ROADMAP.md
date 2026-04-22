# Node.js client roadmap

## Scope

This roadmap delivers JavaScript/TypeScript feature parity with the completed
Python client surface, while preserving Node/browser runtime constraints and
TinyChain transport invariants.

## Parity dependency

The Python client is the canonical functional target. JS milestones that claim
parity require the corresponding Python milestone to be complete and frozen.

## Parity contract

JS parity is complete only when:

1. API parity: equivalent user-facing capabilities exist in `@tinychain/js`.
2. Behavioral parity: request/response/error semantics match Python fixtures.
3. Docs parity: Python and JS examples for the same flow produce the same result.
4. Validation parity: JS passes the parity fixture set associated with Python.

## Milestone plan

### J0: Python parity matrix import

Deliverables:
- Import the finalized Python parity matrix and map each row to a JS owner and
  target module.
- Mark each row as `implemented`, `planned`, or `out_of_scope` with rationale.

Dependencies:
- Python M0 parity inventory complete.

Exit criteria:
- Every parity row has a JS disposition and milestone assignment.
- No ambiguous ownership for parity gaps.

Validation:
- Matrix review and sign-off in `client/js`.

### J1: Core API and IR parity

Deliverables:
- Match Python client equivalents for core state/collection APIs and request
  envelope behavior.
- Keep IR serialization and capability handling aligned with Python and host
  expectations.

Dependencies:
- J0 matrix mapping.
- Python core runtime parity stabilized.

Exit criteria:
- Core parity rows move to `implemented`.
- Equivalent Python/JS fixture cases pass for core operations.

Validation:
- Unit tests for API behavior and envelope shaping.
- Integration tests against TinyChain host endpoints.
- Cross-language fixture tests asserting equivalent outcomes.

### J2: Shard-aware routing and runtime ergonomics

Deliverables:
- Implement client-owned shard routing across blocks/hosts with deterministic
  behavior and clear fallback/health semantics.
- Ensure streaming/tensor APIs remain non-blocking and backpressure-friendly.

Dependencies:
- J1 core API parity.

Exit criteria:
- Shard routing rows in parity matrix marked `implemented`.
- Runtime ergonomics checks pass under stress/streaming fixtures.

Validation:
- Integration tests for routing hints, shard placement, and failure modes.
- Performance/latency sanity checks for event-loop safety.

Decision gates:
- If routing complexity creates semantics drift from Python, pause and reconcile
  the shared contract before expanding behavior.

### J3: Browser/edge parity surface

Deliverables:
- Ship browser/edge build path (ESM/WASM-enabled as applicable) with the same
  high-level API semantics as the Node package.
- Document bundler/runtime integration patterns and security expectations.

Dependencies:
- J1 core API parity.
- Transport parity for HTTP/WebSocket/WebTransport.

Exit criteria:
- Browser/edge fixture suite passes with the same behavior as Node for supported
  operations.
- Documentation includes one canonical integration path and security boundaries.

Validation:
- Browser integration tests.
- Build matrix checks for Node + browser bundles.

### J4: Advanced parity (install/runtime flows, LogChain, media readiness)

Deliverables:
- Match Python parity rows for library install/runtime helpers, LogChain helpers,
  and other advanced surfaces committed in Python.
- Add streaming/media-ready helpers where backend support exists, without
  introducing transport semantics outside TinyChain constraints.

Dependencies:
- Python advanced parity milestones complete.
- J3 browser/edge parity stabilized.

Exit criteria:
- Remaining non-out-of-scope parity rows marked `implemented`.
- Advanced JS examples match Python outcomes for equivalent workflows.

Validation:
- End-to-end parity fixtures for install/runtime and logging flows.
- Compatibility tests across Node and browser for supported features.

## Cross-cutting validation requirements

- Maintain parity fixture suites that run the same logical workflows in Python
  and JS.
- Keep unit/integration/browser tiers separate and reproducible.
- Require negative tests for auth failures, dependency policy violations, and
  transport error behavior.

## Risks and mitigations

- Risk: JS introduces convenience APIs that drift from Python semantics.
  - Mitigation: parity matrix gate and cross-language fixture checks.
- Risk: browser build diverges from Node behavior.
  - Mitigation: shared contract tests across runtime targets.
- Risk: advanced parity blocked by Python schedule.
  - Mitigation: explicit dependency labeling and staged gating (J0-J4).

## Out of scope

- Adding non-TinyChain transport stacks (e.g., WebRTC/STUN/TURN orchestration).
- Shipping UI/design-system components in `client/js` (belongs in `client/web`).
