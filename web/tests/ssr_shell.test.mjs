import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { load } from "cheerio";

const testFile = fileURLToPath(import.meta.url);
const testsDir = path.dirname(testFile);
const webRoot = path.resolve(testsDir, "..");
const fixturePath = path.join(testsDir, "fixtures", "ssr-shell.fixture.html");
const serverDistRoot = path.join(webRoot, "dist", "server");
const defaultBuildHtmlPath = path.join(serverDistRoot, "ssr-shell.html");

const rootId = process.env.TC_WEB_TEST_ROOT_ID ?? "app";
const stateScriptId =
  process.env.TC_WEB_TEST_STATE_SCRIPT_ID ?? "tc-web-initial-state";

const forbiddenInitialStateKeys = new Set([
  "tc_tinychain_server_token",
  "tinychain_server_token",
  "server_token",
  "server_secret",
  "install_token",
  "admin_token",
  "client_secret",
  "private_key",
]);

function resolveHtmlPath() {
  const overridePath = process.env.TC_WEB_TEST_SSR_HTML;

  if (overridePath) {
    return path.resolve(overridePath);
  }

  if (fs.existsSync(defaultBuildHtmlPath)) {
    return defaultBuildHtmlPath;
  }

  if (fs.existsSync(serverDistRoot)) {
    const htmlCandidates = fs
      .readdirSync(serverDistRoot, { withFileTypes: true })
      .filter((entry) => entry.isFile() && path.extname(entry.name) === ".html")
      .map((entry) => path.join(serverDistRoot, entry.name))
      .sort();

    if (htmlCandidates.length > 0) {
      return htmlCandidates[0];
    }
  }

  return fixturePath;
}

function readSsrHtml() {
  const htmlPath = resolveHtmlPath();
  return {
    htmlPath,
    html: fs.readFileSync(htmlPath, "utf8"),
  };
}

function parseHtml(html) {
  return load(html, {
    scriptingEnabled: false,
  });
}

function hasHtmlDoctype($) {
  return $.root()
    .contents()
    .toArray()
    .some((node) => {
      if (node.type !== "directive") {
        return false;
      }

      const directive = `${node.data ?? node.name ?? ""}`.toLowerCase();
      return directive === "!doctype html";
    });
}

function findScriptById($, scriptId) {
  return $("script")
    .filter((_, element) => $(element).attr("id") === scriptId)
    .first();
}

function extractInitialStateJson($, scriptId) {
  const scriptNode = findScriptById($, scriptId);
  assert.ok(scriptNode.length === 1, `missing initial-state script id=${scriptId}`);

  const scriptType = `${scriptNode.attr("type") ?? ""}`.trim().toLowerCase();
  assert.equal(
    scriptType,
    "application/json",
    "initial-state script must use type=\"application/json\""
  );

  const jsonText = scriptNode.text().trim();
  assert.ok(jsonText.length > 0, "initial-state script payload is empty");

  return JSON.parse(jsonText);
}

function extractInitialState(html) {
  const $ = parseHtml(html);
  return extractInitialStateJson($, stateScriptId);
}

function assertNoForbiddenKeys(value, breadcrumbs = []) {
  if (!value || typeof value !== "object") {
    return;
  }

  for (const [key, nestedValue] of Object.entries(value)) {
    const normalized = key.toLowerCase();
    assert.ok(
      !forbiddenInitialStateKeys.has(normalized),
      `initial state includes server secret key "${[
        ...breadcrumbs,
        key,
      ].join(".")}"`
    );

    assertNoForbiddenKeys(nestedValue, [...breadcrumbs, key]);
  }
}

test("SSR shell contains a document root and app mount node", () => {
  const { html, htmlPath } = readSsrHtml();
  const $ = parseHtml(html);

  assert.ok(hasHtmlDoctype($), `expected <!doctype html> in ${htmlPath}`);
  assert.ok($("html").length > 0, `expected <html> in ${htmlPath}`);
  assert.ok($("body").length > 0, `expected <body> in ${htmlPath}`);
  assert.ok(
    $("div")
      .filter((_, element) => $(element).attr("id") === rootId)
      .length > 0,
    `expected root mount id="${rootId}" in ${htmlPath}`
  );
});

test("SSR shell initial-state injection uses expected browser-safe shape", () => {
  const { html } = readSsrHtml();
  const initialState = extractInitialState(html);

  assert.equal(typeof initialState, "object");
  assert.ok(initialState && !Array.isArray(initialState));
  assert.equal(typeof initialState.tinychain, "object");
  assert.ok(initialState.tinychain && !Array.isArray(initialState.tinychain));
  assert.equal(typeof initialState.tinychain.endpoint, "string");
  assert.ok(initialState.tinychain.endpoint.length > 0);
  assert.equal(typeof initialState.view, "object");
  assert.ok(initialState.view && !Array.isArray(initialState.view));

  assertNoForbiddenKeys(initialState);
});
