import escapeHtml from "escape-html";
import serialize from "serialize-javascript";

import {
  APP_ROOT_ID,
  INITIAL_STATE_SCRIPT_ID,
  WINDOW_STATE_KEY,
} from "../shared/constants.mts";

export function renderDocument({ appHtml, appStyleHtml, initialState, title }) {
  const safeState = serialize(initialState, { isJSON: true });
  const safeTitle = escapeHtml(title);
  const safeWindowKey = escapeHtml(WINDOW_STATE_KEY);
  const safeStateScriptId = escapeHtml(INITIAL_STATE_SCRIPT_ID);
  const safeRootId = escapeHtml(APP_ROOT_ID);

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${safeTitle}</title>
    <style>
      :root { color-scheme: light; }
      html, body { margin: 0; padding: 0; }
      body { font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; }
    </style>
    ${appStyleHtml}
  </head>
  <body>
    <div id="${safeRootId}">${appHtml}</div>
    <script id="${safeStateScriptId}" type="application/json">${safeState}</script>
    <script>
      (function () {
        const script = document.getElementById("${safeStateScriptId}");
        window.${safeWindowKey} = script ? JSON.parse(script.textContent || "{}") : {};
      })();
    </script>
    <script type="module" src="/public/client-entry.mjs"></script>
  </body>
</html>`;
}
