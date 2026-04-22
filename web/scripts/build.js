const path = require("node:path");
const { build } = require("esbuild");
const {
  WEB_ROOT,
  APP_SOURCE_ROOT,
  SERVER_SRC,
  CLIENT_SRC,
  SERVER_DIST,
  CLIENT_DIST,
  collectServerEntries,
  collectClientEntries,
  cleanDist,
  ensureDistRoots
} = require("./lib");

async function buildServer(entryPoints) {
  if (entryPoints.length === 0) {
    console.log(`No server entries found under ${APP_SOURCE_ROOT}/server.`);
    return;
  }

  await build({
    absWorkingDir: WEB_ROOT,
    entryPoints,
    outdir: SERVER_DIST,
    outbase: SERVER_SRC,
    bundle: true,
    platform: "node",
    format: "cjs",
    target: ["node20"],
    sourcemap: true,
    logLevel: "info",
    tsconfig: path.join(WEB_ROOT, "tsconfig.server.json")
  });
}

async function buildClient(entryPoints) {
  if (entryPoints.length === 0) {
    console.log(`No client entries found under ${APP_SOURCE_ROOT}/client.`);
    return;
  }

  await build({
    absWorkingDir: WEB_ROOT,
    entryPoints,
    outdir: CLIENT_DIST,
    outbase: CLIENT_SRC,
    bundle: true,
    splitting: true,
    format: "esm",
    platform: "browser",
    target: ["es2022"],
    outExtension: {
      ".js": ".mjs"
    },
    sourcemap: true,
    logLevel: "info",
    tsconfig: path.join(WEB_ROOT, "tsconfig.client.json")
  });
}

async function main() {
  const [serverEntries, clientEntries] = await Promise.all([
    collectServerEntries(),
    collectClientEntries()
  ]);

  await cleanDist();
  await ensureDistRoots();

  await Promise.all([buildServer(serverEntries), buildClient(clientEntries)]);

  if (serverEntries.length === 0 && clientEntries.length === 0) {
    console.log("Build completed with no entrypoints.");
    return;
  }

  console.log("Build completed.");
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
