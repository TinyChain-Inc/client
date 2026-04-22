const fs = require("node:fs/promises");
const path = require("node:path");
const { existsSync } = require("node:fs");

const WEB_ROOT = path.resolve(__dirname, "..");
const APP_SOURCE_ROOT = process.env.TC_WEB_APP_SOURCE_ROOT
  ? path.resolve(WEB_ROOT, process.env.TC_WEB_APP_SOURCE_ROOT)
  : path.join(WEB_ROOT, "examples", "express-rn-ssr", "src");
const DIST_ROOT = path.join(WEB_ROOT, "dist");

const SERVER_SRC = path.join(APP_SOURCE_ROOT, "server");
const CLIENT_SRC = path.join(APP_SOURCE_ROOT, "client");
const SERVER_DIST = path.join(DIST_ROOT, "server");
const CLIENT_DIST = path.join(DIST_ROOT, "client");

const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".mts", ".cts"]);

function hasSourceExtension(filename) {
  if (filename.endsWith(".d.ts")) {
    return false;
  }

  return SOURCE_EXTENSIONS.has(path.extname(filename));
}

async function collectFilesRecursive(dir, includeFile) {
  if (!existsSync(dir)) {
    return [];
  }

  const files = [];
  const stack = [dir];

  while (stack.length > 0) {
    const current = stack.pop();
    const entries = await fs.readdir(current, { withFileTypes: true });

    for (const entry of entries) {
      const absolutePath = path.join(current, entry.name);

      if (entry.isDirectory()) {
        stack.push(absolutePath);
        continue;
      }

      if (entry.isFile() && includeFile(absolutePath)) {
        files.push(absolutePath);
      }
    }
  }

  files.sort();
  return files;
}

async function collectServerEntries() {
  return collectFilesRecursive(SERVER_SRC, hasSourceExtension);
}

async function collectClientEntries() {
  return collectFilesRecursive(CLIENT_SRC, hasSourceExtension);
}

async function collectBuiltJavaScript(dir) {
  return collectFilesRecursive(dir, file => {
    const extension = path.extname(file);
    return extension === ".js" || extension === ".mjs";
  });
}

async function cleanDist() {
  await fs.rm(DIST_ROOT, { recursive: true, force: true });
}

async function ensureDistRoots() {
  await Promise.all([
    fs.mkdir(SERVER_DIST, { recursive: true }),
    fs.mkdir(CLIENT_DIST, { recursive: true })
  ]);
}

module.exports = {
  WEB_ROOT,
  APP_SOURCE_ROOT,
  SERVER_SRC,
  CLIENT_SRC,
  SERVER_DIST,
  CLIENT_DIST,
  collectServerEntries,
  collectClientEntries,
  collectBuiltJavaScript,
  cleanDist,
  ensureDistRoots
};
