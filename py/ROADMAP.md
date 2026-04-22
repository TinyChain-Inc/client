# Python client roadmap

## Scope

This roadmap drives full feature parity between this client and the v1 Python
client at `~/Documents/tinychain/client/py`, then hardens the v2 implementation
for long-term maintenance.

## Parity contract

Parity is complete only when all of the following are true:

1. API parity: supported v1 user-facing modules and flows have an equivalent v2
   path.
2. Behavioral parity: equivalent operations produce equivalent request shapes,
   response handling, and error semantics.
3. Documentation parity: migration guidance exists for each changed v1 surface.
4. Test parity: coverage proves equivalent behavior for the same fixture cases.

## Milestone plan

### M0: Parity inventory and rubric

Deliverables:
- Build an explicit parity matrix using `~/Documents/tinychain/client/py` as the
  reference by module and feature area.
- Label each gap as either `parity_gap` or `framework_gap`.
- Define target behavior for each parity gap and identify owner files/tests.

Dependencies:
- Stable local checkout of the v1 reference client.

Exit criteria:
- Every v1 public surface is marked `matched`, `gap`, or `explicitly_out_of_scope`.
- Each `gap` entry has a planned milestone and validation case.

Validation:
- Static comparison of module exports, docs, and key examples.
- Tracking doc committed in `client/py`.

Decision gates:
- If a v1 surface conflicts with v2 invariants, document the mismatch and
  require an explicit design decision before implementation.

### M1: Core runtime parity (transport/auth/session/collections)

Deliverables:
- Close parity gaps for host transport/auth ergonomics and eager/deferred session
  expectations.
- Align collection and scalar workflows for common operations (`Table`, `BTree`,
  `Tensor`, scalar ops) with v1 user expectations.
- Preserve the no-exposed-`txn_id` contract while matching v1 workflow outcomes.

Dependencies:
- M0 parity matrix.
- Stable kernel-side request/response envelopes.

Exit criteria:
- Core parity matrix rows move to `matched`.
- Existing docs include equivalent v1-to-v2 snippets for each major flow.

Validation:
- Unit tests for API behavior and envelope shaping.
- Integration tests against local TinyChain hosts.
- Side-by-side fixture assertions comparing v1 and v2 outcomes.

Migration notes:
- Publish one canonical migration guide for session and auth behavior changes.

### M2: Library/runtime install parity and cross-host behavior

Deliverables:
- Mirror `/lib` installer contract and dependency edge persistence.
- Verify cross-host dependency execution semantics in Python in-process and
  remote-host combinations.
- Keep telemetry/billing/auth propagation aligned with host expectations.

Dependencies:
- M1 transport/auth parity.

Exit criteria:
- Reference example demonstrates local library `A` invoking remote dependency
  `B` with correct auth and transaction behavior.
- Installer and dependency graph behavior validated end-to-end.

Validation:
- Integration tests for install, dependency calls, auth propagation.
- Negative tests for egress restrictions and missing capability failures.

Migration notes:
- Document deprecation path for legacy package-local HTTP shims.

### M3: Python-defined library compilation and opdef transforms

Deliverables:
- Implement `tc.define` compilation into standard v2 IR (`TCRef`, `OpRef`,
  `Scalar`, `LibrarySchema`).
- Implement Autograph-style AST rewrite for supported decorator flows.
- Keep unsupported syntax handling deterministic with explicit error messages.

Dependencies:
- M2 installer/runtime parity.

Exit criteria:
- Python-defined libraries install and execute through the same host dispatch path
  as WASM/HTTP/PyO3.
- AST transform acceptance tests cover supported and rejected syntax.

Validation:
- Compiler unit tests for emitted IR determinism.
- Integration tests for install + execute + rollback/error paths.

Decision gates:
- If transform complexity grows beyond the supported subset, freeze scope and
  require explicit expansion approval before adding syntax classes.

### M4: LogChain and parity hardening

Deliverables:
- Add `tc.logchain` helper surfaces for publish/subscribe/export/topic flows.
- Finalize parity matrix with remaining items moved to `matched` or documented
  `out_of_scope`.
- Tighten docs and examples to remove stale compatibility ambiguity.

Dependencies:
- M1 core runtime parity.

Exit criteria:
- LogChain helper API is documented and tested.
- Parity matrix has no unresolved `gap` entries without a decision record.

Validation:
- Integration tests for batch and streaming log flows.
- Documentation review for migration completeness and accuracy.

## Cross-cutting validation requirements

- Maintain separate test tiers: unit, integration, and migration/parity fixtures.
- Keep a runnable parity regression suite that compares representative v1/v2
  workflows.
- Include negative tests for auth, dependency egress, and transaction boundaries.

## Risks and mitigations

- Risk: parity scope expands indefinitely.
  - Mitigation: freeze parity matrix definitions in M0 and require change control
    for new parity targets.
- Risk: compiler/AST features introduce hidden semantic drift.
  - Mitigation: enforce deterministic IR snapshots and reject unsupported syntax.
- Risk: docs drift from behavior.
  - Mitigation: bind migration examples to tested fixtures.

## Deferred explorations

- Peer-assisted discovery (`tc://`, overlay hints) for partially disconnected
  deployments, after core parity milestones close.
