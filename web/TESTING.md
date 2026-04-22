# Web Testing Guide

This guide defines the test surface for `client/web` and the
`examples/express-rn-ssr` reference app.

Supported runtime for all commands below: Node.js `20.x`.

## Test levels

1. Unit tests
   - Focus on small pure functions and adapters.
   - No server process required.
2. Integration tests
   - Build artifacts + SSR route/output checks.
   - TinyChain adapter behavior against a mock TinyChain HTTP server.
3. Browser functional tests
   - Headless browser flow over the real built app.
   - Verify SSR -> hydration -> in-browser refresh behavior.

## Commands

Run from `client/web`:

```bash
# requires Node.js 20.x
npm run build
npm run test:unit
npm run test:integration
```

Default test command (build + unit + integration):

```bash
npm run test
```

Browser smoke test (optional in lightweight CI):

```bash
npm run test:browser
```

Everything:

```bash
npm run test:all
```

GitHub CI runs the same full suite in
[`client-web.yml`](/home/haydn/Documents/tcv2/.github/workflows/client-web.yml).

## Browser test prerequisites

The browser test uses `@playwright/test` and requires a local browser install:

```bash
npx playwright install chromium
```

If your environment cannot run browsers (restricted CI/sandbox), skip
`test:browser` and run it in a nightly or pre-release validation job instead.

Playwright uses [`playwright.config.mjs`](/home/haydn/Documents/tcv2/client/web/playwright.config.mjs)
with a managed `webServer` command so browser tests no longer need custom
process/port orchestration scripts.

## Coverage map

- Unit:
  - `tests/safe_serialize.unit.test.mjs`
- Integration:
  - `tests/ssr_shell.test.mjs`
  - `tests/tinychain_client.integration.test.mjs`
- Browser functional:
  - `tests/browser/smoke.spec.mjs`
