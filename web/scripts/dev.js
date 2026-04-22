const path = require("node:path");
const { context } = require("esbuild");
const {
  WEB_ROOT,
  APP_SOURCE_ROOT,
  SERVER_SRC,
  CLIENT_SRC,
  SERVER_DIST,
  CLIENT_DIST,
  collectServerEntries,
  collectClientEntries,
  ensureDistRoots
} = require("./lib");

async function createServerWatch(entryPoints) {
  if (entryPoints.length === 0) {
    console.log(`No server entries found under ${APP_SOURCE_ROOT}/server; server watcher disabled.`);
    return null;
  }

  return context({
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

async function createClientWatch(entryPoints) {
  if (entryPoints.length === 0) {
    console.log(`No client entries found under ${APP_SOURCE_ROOT}/client; client watcher disabled.`);
    return null;
  }

  return context({
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

  await ensureDistRoots();

  const [serverContext, clientContext] = await Promise.all([
    createServerWatch(serverEntries),
    createClientWatch(clientEntries)
  ]);

  const contexts = [serverContext, clientContext].filter(Boolean);

  if (contexts.length === 0) {
    console.log("No entrypoints found; dev watch has nothing to do.");
    return;
  }

  await Promise.all(contexts.map(ctx => ctx.watch()));

  console.log("Watching server/client sources. Press Ctrl+C to stop.");

  let stopping = false;
  const stop = async () => {
    if (stopping) {
      return;
    }

    stopping = true;
    await Promise.all(contexts.map(ctx => ctx.dispose()));
    process.exit(0);
  };

  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
