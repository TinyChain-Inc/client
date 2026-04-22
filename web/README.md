# TinyChain Web Client Example

`client/web` is the TinyChain web integration scaffold: a Node server renders
an SSR HTML shell, injects initial state, and the browser hydrates the same UI
with browser-safe TinyChain config.

It consumes the local `client/js` package as `@tinychain/js` via
`file:../js` dependency wiring.

Runtime requirement: Node.js `20.x` only. Older Node releases are unsupported.

Canonical example entry: [`examples/express-rn-ssr/README.md`](/home/haydn/Documents/tcv2/client/web/examples/express-rn-ssr/README.md).
Build scripts default to that example source root; override with
`TC_WEB_APP_SOURCE_ROOT` only when testing alternate entrypoints.

## Purpose

- Keep `client/js` as the TinyChain protocol/runtime implementation.
- Show one explicit server/browser boundary for SSR + hydration.
- Provide a minimal contract for the HTML shell and injected initial state.

## Architecture Boundaries

Server responsibilities (Node):

- Render SSR HTML and the root mount node.
- Fetch TinyChain data with server-only credentials.
- Inject a browser-safe initial state payload into the shell.
- Serve static browser bundles from `dist/client`.

Browser responsibilities (hydrate client):

- Read injected initial state from the HTML shell.
- Hydrate the pre-rendered app tree.
- Perform user-triggered TinyChain calls using only public-safe config.

Shared responsibilities:

- Reuse shared view/components and one TinyChain adapter contract across
  server/browser runtimes.
- Avoid string-concatenated TinyChain paths in handlers/views.

## TinyChain Loading Model

Server side:

1. Build a TinyChain client from server-only env vars.
2. Run TinyChain calls inside request handlers.
3. Convert results into a safe initial state object.

Browser side:

1. Read injected JSON state (typically from `window.__TC_INITIAL_STATE__`).
2. Initialize browser client with `initialState.tinychain`.
3. Hydrate and issue user-driven requests with that restricted config.

`client/web` calls the TinyChain JS client through one explicit path:
`request({ method: "GET", path: "/healthz" })`. If that API is unavailable,
startup fails fast. There is no fallback path.

The minimal injected state contract used by tests is:

```json
{
  "tinychain": {
    "endpoint": "http://127.0.0.1:8702",
    "publisher": "example-devco",
    "service": "hello-web",
    "version": "0.17.0"
  },
  "view": {}
}
```

The shared key is `WINDOW_STATE_KEY = "__TC_INITIAL_STATE__"` in
`src/shared/constants.mts`.

## Security Note

Never expose server credentials to the browser bundle. Server-only vars (for
example, install/admin/service tokens) must stay in Node runtime env only and
must never be serialized into `window` globals, inline scripts, or browser
bundled code.

## Environment Variables

Use [`client/web/.env.example`](/home/haydn/Documents/tcv2/client/web/.env.example) as the baseline contract.

- Server-only: `TC_TINYCHAIN_SERVER_TOKEN` and any privileged credentials.
- Browser-safe: `TC_PUBLIC_*` values only.
- Shared metadata: host/service identity used to assemble URI builders.
- `TC_TINYCHAIN_HTTP_URL` is the only supported server endpoint variable.

## Run And Build Commands

From [`client/web/examples/express-rn-ssr/package.json`](/home/haydn/Documents/tcv2/client/web/examples/express-rn-ssr/package.json):

```bash
# from repo root
# requires Node.js 20.x
cp client/web/.env.example client/web/.env
cd client/web/examples/express-rn-ssr
npm run bootstrap
npm run dev     # watch/rebuild server + client bundles
npm run build   # emits dist/server + dist/client bundles
npm run start   # runs built server
npm run test    # build + unit + integration tests
npm run test:browser   # optional Playwright smoke test
```

## Testing

Detailed test guidance (unit/integration/browser) lives in
[`TESTING.md`](/home/haydn/Documents/tcv2/client/web/TESTING.md).

Run with plain Node tooling:

```bash
cd client/web
npm run test:unit
npm run test:integration
```

Test behavior:

- If present, tests read built SSR HTML from:
  `client/web/dist/server/ssr-shell.html` (or the first `.html` under
  `client/web/dist/server`).
- Otherwise tests use a fixture shell under `client/web/tests/fixtures/`.
- No source imports are required; tests validate rendered artifact shape only.
- Integration coverage validates that both server and browser runtime adapters
  can call TinyChain (`GET /healthz`) through `@tinychain/js`.

Optional test overrides:

- `TC_WEB_TEST_SSR_HTML`: absolute/relative path to SSR HTML to validate.
- `TC_WEB_TEST_ROOT_ID`: expected root mount id (default: `app`).
- `TC_WEB_TEST_STATE_SCRIPT_ID`: expected JSON script id (default:
  `tc-web-initial-state`, optional).
