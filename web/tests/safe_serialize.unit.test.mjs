import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const testFile = fileURLToPath(import.meta.url);
const testsDir = path.dirname(testFile);
const webRoot = path.resolve(testsDir, "..");
const exportsPath = path.join(webRoot, "dist", "server", "testExports.js");

async function loadTestExports() {
  const moduleUrl = pathToFileURL(exportsPath).href;
  return import(moduleUrl);
}

test("serializeForInlineScript escapes script-sensitive characters", async () => {
  const { serializeForInlineScript } = await loadTestExports();
  const payload = {
    value: "<script>\u2028&\u2029</script>"
  };

  const serialized = serializeForInlineScript(payload);

  assert.equal(typeof serialized, "string");
  assert.match(serialized, /\\u003[cC]script\\u003[eE]/);
  assert.match(serialized, /\\u003[cC]\\u002[fF]script\\u003[eE]/);
  assert.ok(serialized.includes("\\u2028"));
  assert.ok(serialized.includes("\\u2029"));
  assert.ok(serialized.includes("&"));
});

test("escapeHtml escapes HTML meta characters", async () => {
  const { escapeHtml } = await loadTestExports();

  const escaped = escapeHtml(`<a href="x&y">O'Reilly</a>`);
  assert.equal(escaped, "&lt;a href=&quot;x&amp;y&quot;&gt;O&#39;Reilly&lt;/a&gt;");
});
