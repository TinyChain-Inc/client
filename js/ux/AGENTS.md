# TinyChain UX Agent Notes

- The UX kit **must not invent new protocol layers**. All data access flows through the published TinyChain JS client (Node.js) or WASM bundle; do not embed bespoke REST clients or bypass capability tokens.
- Keep environment configuration centralized: `.env` files for server/browser, secure secret handling for API tokens, and optional manifest-driven config for staging/prod switching.
- Ensure React (web) and React Native share as much code as possible via monorepo packages; avoid large platform-specific forks unless necessary.
- Testing:
  - Unit tests (Jest) for shared hooks/components.
  - Integration/e2e tests (Playwright for web, Detox/Expo for native) that hit a local TinyChain stack (can run against mocked `tc-server` if needed).
  - Snapshot tests for critical admin dashboards (rollout monitor, clearinghouse queue).
- Documentation: keep README steps executable; reference the main roadmap for milestones; highlight how to swap between server-rendered, browser-only, and hybrid deployments.
