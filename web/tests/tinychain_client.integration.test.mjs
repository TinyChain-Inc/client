import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { close, createMockTinyChainServer, listen } from "./helpers/mockTinyChainServer.mjs";

const testFile = fileURLToPath(import.meta.url);
const testsDir = path.dirname(testFile);
const webRoot = path.resolve(testsDir, "..");

async function importAdapterModule(distPath) {
  const moduleUrl = pathToFileURL(distPath).href;
  return import(moduleUrl);
}

test("TinyChain JS client calls succeed for server and browser runtime adapters", async () => {
  const serverAdapterPath = path.join(webRoot, "dist", "server", "index.js");
  const browserAdapterPath = path.join(webRoot, "dist", "client", "hydrate.mjs");

  const [serverAdapterModule, browserAdapterModule] = await Promise.all([
    importAdapterModule(serverAdapterPath),
    importAdapterModule(browserAdapterPath)
  ]);

  const createServerAdapter = serverAdapterModule.createTinyChainClientAdapter;
  const createBrowserAdapter = browserAdapterModule.createTinyChainClientAdapter;

  assert.equal(typeof createServerAdapter, "function");
  assert.equal(typeof createBrowserAdapter, "function");

  const mockServer = createMockTinyChainServer();
  const port = await listen(mockServer);
  const endpoint = `http://127.0.0.1:${port}`;

  try {
    const [serverClient, browserClient] = await Promise.all([
      createServerAdapter({ runtime: "server", endpoint }),
      createBrowserAdapter({ runtime: "browser", endpoint })
    ]);

    const [serverPayload, browserPayload] = await Promise.all([
      serverClient.fetchDemoData({ scope: "server-test" }),
      browserClient.fetchDemoData({ scope: "browser-test" })
    ]);

    assert.equal(serverPayload.source, "real");
    assert.equal(serverPayload.runtime, "server");
    assert.equal(serverPayload.ok, true);
    assert.equal(serverPayload.service, "mock-tinychain");

    assert.equal(browserPayload.source, "real");
    assert.equal(browserPayload.runtime, "browser");
    assert.equal(browserPayload.ok, true);
    assert.equal(browserPayload.service, "mock-tinychain");
  } finally {
    await close(mockServer);
  }
});
