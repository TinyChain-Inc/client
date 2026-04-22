const path = require("node:path");
const { spawn } = require("node:child_process");
const {
  WEB_ROOT,
  SERVER_DIST,
  CLIENT_DIST,
  collectServerEntries,
  collectClientEntries,
  collectBuiltJavaScript
} = require("./lib");

function runNodeScript(scriptPath) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [scriptPath], {
      cwd: WEB_ROOT,
      stdio: "inherit"
    });

    child.on("error", reject);
    child.on("exit", code => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`Script failed: ${path.basename(scriptPath)} (exit ${code})`));
    });
  });
}

function runNodeTest(filePattern) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--test", filePattern], {
      cwd: WEB_ROOT,
      stdio: "inherit"
    });

    child.on("error", reject);
    child.on("exit", code => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`Test execution failed for ${filePattern} (exit ${code})`));
    });
  });
}

async function assertOutputs(label, hasSources, outputDir) {
  if (!hasSources) {
    return;
  }

  const builtFiles = await collectBuiltJavaScript(outputDir);
  if (builtFiles.length === 0) {
    throw new Error(`Expected built ${label} output in ${outputDir}, but none was found.`);
  }
}

async function runUnitAndIntegrationTests() {
  await runNodeTest("tests/safe_serialize.unit.test.mjs");
  await runNodeTest("tests/ssr_shell.test.mjs");
  await runNodeTest("tests/tinychain_client.integration.test.mjs");
}

async function main() {
  const [serverEntries, clientEntries] = await Promise.all([
    collectServerEntries(),
    collectClientEntries()
  ]);

  await runNodeScript(path.join(__dirname, "build.js"));

  await Promise.all([
    assertOutputs("server", serverEntries.length > 0, SERVER_DIST),
    assertOutputs("client", clientEntries.length > 0, CLIENT_DIST)
  ]);

  await runUnitAndIntegrationTests();

  console.log("Web scaffold test passed.");
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
