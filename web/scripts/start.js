const path = require("node:path");
const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const { collectBuiltJavaScript, SERVER_DIST } = require("./lib");

const DEFAULT_SERVER_ENTRY = path.join(SERVER_DIST, "index.js");

async function resolveServerEntry() {
  if (existsSync(DEFAULT_SERVER_ENTRY)) {
    return DEFAULT_SERVER_ENTRY;
  }

  const builtFiles = await collectBuiltJavaScript(SERVER_DIST);
  if (builtFiles.length > 0) {
    return builtFiles[0];
  }

  throw new Error(
    "No built server bundle found in dist/server. Run `npm run build` first."
  );
}

async function main() {
  const serverEntry = await resolveServerEntry();
  const args = [serverEntry, ...process.argv.slice(2)];
  const child = spawn(process.execPath, args, {
    stdio: "inherit"
  });

  const forwardSignal = signal => {
    if (!child.killed) {
      child.kill(signal);
    }
  };

  process.on("SIGINT", () => forwardSignal("SIGINT"));
  process.on("SIGTERM", () => forwardSignal("SIGTERM"));

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }

    process.exit(code === null ? 1 : code);
  });
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
