# TinyChain web client roadmap

This roadmap evolves `client/web` from the current Express + React Native Web
integration into a production UI toolkit for TinyChain applications.

## Scope

The roadmap covers three layers:

1. Integration layer: SSR + hydration + TinyChain JS usage across server/browser.
2. UI layer: full React Native widget library (Material-like scope).
3. App utilities: auth/session and state-management helpers for production apps.

## Core constraints

- `client/js` remains the TinyChain protocol/runtime source of truth.
- Web transport scope remains TinyChain-native (HTTP/WebSocket/WebTransport).
- No browser exposure of server-only credentials.
- One obvious path per feature; no legacy fallback branches.

## Milestone plan

### W0: Integration baseline lock (current scaffold hardening)

Deliverables:
- Keep canonical Express + RN SSR/hydration example stable.
- Maintain explicit server/browser config boundaries and startup validation.
- Preserve single TinyChain JS call path and Node 20-only runtime policy.

Dependencies:
- Stable `@tinychain/js` package contract.

Exit criteria:
- Example runs and tests pass (`unit`, `integration`, `browser`) on Node 20.
- Docs accurately describe runtime/environment/test workflows.

Validation:
- `npm run test:all` in `client/web`.
- CI workflow runs same suite.

### W1: Widget system foundation

Deliverables:
- Define design tokens (typography, spacing, color, elevation, radius, motion).
- Implement base primitives for RN web rendering:
  `Box`, `Text`, `Button`, `Input`, `Select`, `Checkbox`, `Radio`, `Switch`,
  `FormField`, `Card`, `List`, `Table/Grid`, `Modal`, `Toast`, `Tabs`, `Nav`.
- Add theming and dark/light mode support with SSR-safe defaults.

Dependencies:
- W0 baseline lock.

Exit criteria:
- Widget primitives documented with examples and accessibility notes.
- SSR and hydration produce stable markup without mismatch warnings.

Validation:
- Unit tests for primitive behavior.
- Visual/regression snapshots for core components.
- Browser functional tests for interaction and focus behavior.

Decision gates:
- Freeze naming and token model before adding large widget families.

### W2: Composite widgets and app-shell patterns

Deliverables:
- Add higher-level composites:
  data panels, filter bars, paginated lists/tables, form layouts, nav shells.
- Provide layout templates for dashboard/detail/edit flows.
- Define extension points for app-specific theming and iconography.

Dependencies:
- W1 primitive set complete.

Exit criteria:
- At least one complete app-shell example composed only from widget library APIs.
- Composite widgets include keyboard/accessibility behavior docs.

Validation:
- Integration tests for composite interactions.
- Browser smoke flows for common app-shell navigation.

### W3: TinyChain `State` widget coverage

Deliverables:
- Provide canonical renderers/editors for common TinyChain state surfaces:
  scalars, tuples/maps, typed values, `Table`, `BTree`, and `Tensor` views.
- Define a shared data-binding contract for SSR-injected + browser-refreshed
  state updates.
- Add reusable data ops utilities (sort/filter/paginate/select/edit) for state
  widgets.

Dependencies:
- W2 app-shell patterns.
- Stable JS client data access surfaces.

Exit criteria:
- State widget catalog covers prioritized TinyChain state types.
- Example app demonstrates state inspect and update loops using shared widgets.

Validation:
- Unit tests for state rendering and edit serialization.
- Integration tests for server fetch + browser refresh consistency.
- Browser tests for end-user state operations.

Migration notes:
- Document recommended replacement path for hand-rolled state displays.

### W4: Auth/session utility layer

Deliverables:
- Add auth helpers for login/logout/session establishment.
- Provide secure cookie utilities:
  HTTP-only cookie setup, same-site policy defaults, secure flag guidance.
- Add authorized request helpers for SSR and browser flows:
  cookie forwarding, session continuity, auth failure handling.

Dependencies:
- W0 runtime boundary enforcement.
- Host-side auth/token policies required by target deployments.

Exit criteria:
- Auth/session utilities documented and exercised in example app flows.
- Security boundary documentation includes cookie and credential handling rules.

Validation:
- Integration tests for login/session lifecycle.
- Browser functional tests for authenticated navigation and expiry handling.
- Negative tests for unauthorized and CSRF-sensitive scenarios.

Decision gates:
- If auth requirements vary by deployment model, provide explicit policy adapters
  rather than implicit fallback behavior.

### W5: Client state-management utilities

Deliverables:
- Ship minimal app state utilities for:
  auth/session state, request lifecycle state, optimistic update state,
  cache invalidation hooks.
- Ensure utilities compose with state widgets and TinyChain request helpers.

Dependencies:
- W3 state widget contract.
- W4 auth/session utilities.

Exit criteria:
- Example app uses shared state-management utilities instead of local ad-hoc
  state wiring.
- Utility APIs documented with anti-pattern guidance.

Validation:
- Unit tests for state transitions and error states.
- Integration tests for optimistic update and rollback behavior.

### W6: Production readiness and ecosystem packaging

Deliverables:
- Publish packaging/versioning strategy for `client/web` widget and utility
  surfaces.
- Add upgrade notes, deprecation policy, and compatibility matrix with
  `@tinychain/js`.
- Provide domain starter templates built from the widget/state/auth foundations.

Dependencies:
- W0-W5 complete.

Exit criteria:
- Release checklist exists and is reproducible.
- Documentation supports onboarding without internal context.

Validation:
- Full CI matrix for unit/integration/browser tests.
- Smoke validation of starter templates.

## Cross-cutting acceptance criteria

Across all milestones:

1. No server-secret leakage to browser bundles.
2. SSR/hydration consistency holds for supported components.
3. TinyChain JS usage remains through shared, documented contracts.
4. Test coverage includes positive and negative auth/data/update flows.
5. Docs stay aligned with shipped behavior and runtime requirements.

## Risks and mitigations

- Risk: component sprawl without consistent UX contract.
  - Mitigation: freeze token system and primitive API before composites.
- Risk: auth/session helpers become framework-specific.
  - Mitigation: keep helpers transport- and framework-agnostic; expose adapters
    only at integration boundaries.
- Risk: state widgets drift from TinyChain data semantics.
  - Mitigation: require shared serializer/binding contracts and fixture-based
    tests for all supported state types.

## Non-goals

- Building a framework-specific monolith tied to one web framework internals.
- Adding non-TinyChain transport orchestration (WebRTC/STUN/TURN).
- Duplicating TinyChain protocol logic already owned by `client/js`.
