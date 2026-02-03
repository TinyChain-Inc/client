# TinyChain UX Kit

This subproject houses the reference React.js + React Native experience layer for TinyChain. It
demonstrates how to consume TinyChain services from:

1. **Node.js server mode.** A Next.js/Express-style server that embeds the TinyChain JS client (from `js/`)
   to pre-render views or proxy API calls. Useful for SSR, dashboards, and secure environments.
2. **Browser mode.** A React SPA that bundles the TinyChain WASM/JS client and talks directly to TinyChain hosts
   (via fetch/WebSocket). Ideal for lightweight dashboards or developer tooling.
3. **Hybrid / React Native mode.** A shared component library consumed by React Native apps that call TinyChain
   over HTTPS using the same JS client.

## Getting started (outline)

> Detailed commands will land once the kit stabilizes; track the roadmap for the MVP milestone.

1. Install dependencies:
   ```bash
   npm install
   npm run bootstrap   # sets up shared packages (storybook, RN app, etc.)
   ```
2. Run the Node.js server example:
   ```bash
   npm run dev:server
   ```
3. Run the browser SPA:
   ```bash
   npm run dev:web
   ```
4. Run the React Native app (Expo/Bare workflow TBD):
   ```bash
   npm run dev:native
   ```

## Design goals

- **Zero bespoke protocols.** Reuse the TinyChain JS client / WASM bundle; no proprietary transport layers.
- **Configurable endpoints.** All environments load endpoints/auth from `.env` files or TinyChain manifests so publishers can point UX at staging vs production effortlessly.
- **Shared components.** React + React Native share as much UI state/logging code as possible (via monorepo packages) to cut duplication.
- **CI coverage.** UX tests (Jest + Playwright/Detox) run in the same pipeline as the Node client, ensuring sample flows (login, queue management, clearinghouse remediations) stay working.

## Relationship to other modules

- Depends on `js/` (Node.js client) for protocol bindings.
- Uses control-plane services such as `/service/std/rollout`, `/service/std/clearinghouse`, and `/service/std/time` for admin dashboards.
- Meant as a reference; production apps can fork components or copy patterns.
