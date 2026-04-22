# TinyChain web client roadmap

This document plans the `client/web` package that will sit alongside `py/`,
`js/`, and `rust/`.

## Goal

Deliver a minimal, production-usable web integration that demonstrates:

1. A Node.js server using Express.
2. A templated webpage flow (server-rendered HTML shell + injected app state).
3. React Native component rendering on both:
   - the server (SSR), and
   - the browser (hydrate/takeover).
4. TinyChain JS client usage on both:
   - server-side request handlers, and
   - browser-side UI interactions.

## Design constraints

- Reuse `client/js` as the protocol/runtime source of truth; `client/web` is an
  integration layer, not a second TinyChain client implementation.
- Keep the runtime single-threaded and event-loop friendly in Node.
- Keep transports aligned with TinyChain invariants (HTTP/WebSocket/WebTransport
  only; no WebRTC/STUN/TURN scope).
- Keep browser code dependency-minimal and avoid bundling server secrets.
- Keep URI/path construction in code via shared builders/helpers, not string
  concatenation in examples.

## Non-goals

- Building a general-purpose UI component library.
- Introducing new kernel verbs or adapter-specific semantics.
- Shipping a framework-specific monolith (for example, tightly coupled Next.js
  internals) in v1 of `client/web`.

## Proposed package layout

`client/web/`

- `README.md`: quickstart + architecture notes.
- `ROADMAP.md`: this plan.
- `examples/express-rn-ssr/`: canonical reference app.
- `packages/server/`: Express + SSR wiring helpers.
- `packages/browser/`: hydration/bootstrap helpers.
- `packages/shared/`: isomorphic view and TinyChain access helpers.

The `packages/*` split is optional for MVP; a single example-first layout is
acceptable as long as the server/browser/shared boundaries are explicit.

## Implementation phases

1. **Phase 0: contract and scaffolding**
   - Create `client/web` docs and skeleton.
   - Define environment contract for TinyChain endpoints/auth:
     - server env (full credentials/capability context),
     - browser env (public-safe config only).
   - Decide bundling path for dual targets (Node + browser) with one shared UI.

2. **Phase 1: Express + template shell**
   - Add an Express server with:
     - HTML template rendering,
     - route-level state injection for hydration,
     - static asset serving for browser bundle.
   - Add one example route (`/`) proving template + state wiring.

3. **Phase 2: React Native SSR and hydration**
   - Render shared React Native (via `react-native-web`) components on the server.
   - Hydrate the same component tree in the browser.
   - Verify no SSR/client markup divergence for the starter page.

4. **Phase 3: TinyChain JS isomorphic usage**
   - Server side:
     - call TinyChain via `client/js` inside Express handlers,
     - render fetched data into SSR output.
   - Browser side:
     - initialize browser-safe TinyChain client config,
     - perform a user-triggered call and update UI state.
   - Document auth boundary explicitly (never expose server secrets).

5. **Phase 4: hardening and docs**
   - Add minimal tests for:
     - SSR response shape,
     - hydration bootstrap success,
     - one server-side and one client-side TinyChain call path.
   - Add a lightweight browser functional smoke test and document prerequisites.
   - Document local run workflow and expected outputs.
   - Add explicit testing guidance (`TESTING.md`) for unit/integration/browser tiers.
   - Add planned blog draft in `docs/blog/0.17` for this surface area.

## MVP acceptance criteria

The first deliverable is complete when:

1. `client/web/examples/express-rn-ssr` runs locally and serves one page.
2. The page is server-rendered from React Native components and then hydrated in
   the browser without runtime mismatch errors.
3. Server-side TinyChain call succeeds and renders data in initial HTML.
4. Browser-side TinyChain call succeeds from a user action and updates the page.
5. Example docs explain env vars, run commands, and the server/browser security
   boundary.

## Risks and mitigations

- **Risk:** SSR/hydration mismatch from React Native Web config drift.
  - **Mitigation:** lock one canonical Babel/runtime config shared by server and
    browser bundles.
- **Risk:** accidental credential leakage to browser bundle.
  - **Mitigation:** strict split of server-only env vars vs public config, with
    explicit validation at startup.
- **Risk:** duplicated TinyChain client wrappers across server/browser.
  - **Mitigation:** keep protocol calls in shared helpers that import `client/js`
    primitives and specialize only transport/bootstrap edges.

## Follow-on milestones (post-MVP)

1. Queue-aware examples for long-running workflows that exceed synchronous
   request budgets.
2. Optional websocket streaming demo once the JS client websocket helper lands.
3. Framework adapters (for example, Vite or Next integration) after the Express
   reference path is stable.
