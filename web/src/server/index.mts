import path from "node:path";

import "dotenv/config";
import { cleanEnv, str, url } from "envalid";
import express from "express";
import { renderToStaticMarkup, renderToString } from "react-dom/server";
import { AppRegistry } from "react-native-web";

import { App } from "../shared/App.mts";
import { APP_NAME } from "../shared/constants.mts";
import { createTinyChainClientAdapter } from "../shared/tinychain/adapter.mts";
import { renderDocument } from "./renderDocument.mts";

const WEB_ROOT = path.resolve(__dirname, "../..");
const DEFAULT_PUBLIC_ROOT = path.join(WEB_ROOT, "examples", "express-rn-ssr", "public");

let isAppRegistered = false;

function readServerConfig() {
  const env = cleanEnv(process.env, {
    TC_TINYCHAIN_HTTP_URL: url({ default: "http://127.0.0.1:8702" }),
    TC_PUBLIC_TINYCHAIN_HTTP_URL: str({ default: "" }),
    TC_PUBLIC_TINYCHAIN_PUBLISHER: str({ default: "example-devco" }),
    TC_PUBLIC_TINYCHAIN_SERVICE: str({ default: "hello-web" }),
    TC_PUBLIC_TINYCHAIN_SERVICE_VERSION: str({ default: "0.17.0" }),
    TC_WEB_PUBLIC_ROOT: str({ default: "" }),
  });

  const endpoint = env.TC_TINYCHAIN_HTTP_URL;
  const publicEndpoint =
    env.TC_PUBLIC_TINYCHAIN_HTTP_URL.length > 0
      ? env.TC_PUBLIC_TINYCHAIN_HTTP_URL
      : endpoint;
  const publicRoot =
    env.TC_WEB_PUBLIC_ROOT.length > 0
      ? path.resolve(WEB_ROOT, env.TC_WEB_PUBLIC_ROOT)
      : DEFAULT_PUBLIC_ROOT;

  return {
    endpoint,
    publicEndpoint,
    publicRoot,
    publisher: env.TC_PUBLIC_TINYCHAIN_PUBLISHER,
    service: env.TC_PUBLIC_TINYCHAIN_SERVICE,
    version: env.TC_PUBLIC_TINYCHAIN_SERVICE_VERSION,
  };
}

function renderApp(initialProps) {
  if (!isAppRegistered) {
    AppRegistry.registerComponent(APP_NAME, () => App);
    isAppRegistered = true;
  }

  const { element, getStyleElement } = AppRegistry.getApplication(APP_NAME, {
    initialProps,
  });

  return {
    appHtml: renderToString(element),
    appStyleHtml: renderToStaticMarkup(getStyleElement()),
  };
}

export function createServer() {
  const config = readServerConfig();
  const app = express();

  app.use("/public", express.static(config.publicRoot));
  app.use("/client", express.static(path.join(WEB_ROOT, "dist", "client")));

  app.get("/healthz", (req, res) => {
    res.status(200).json({
      ok: true,
      service: "tinychain-web-example"
    });
  });

  app.get("/", async (req, res, next) => {
    try {
      const tinychainClient = await createTinyChainClientAdapter({
        runtime: "server",
        endpoint: config.endpoint,
      });

      const serverData = await tinychainClient.fetchDemoData({
        scope: "server-ssr",
      });

      const initialState = {
        tinychain: {
          endpoint: config.publicEndpoint,
          publisher: config.publisher,
          service: config.service,
          version: config.version,
          ...serverData,
        },
        view: {
          title: "TinyChain + React Native Web",
          message: serverData.message ?? "SSR load complete.",
        },
      };

      const { appHtml, appStyleHtml } = renderApp({
        initialData: initialState.tinychain,
        tinychainClient: null,
      });

      const html = renderDocument({
        appHtml,
        appStyleHtml,
        initialState,
        title: initialState.view.title,
      });

      res.status(200).type("html").send(html);
    } catch (error) {
      next(error);
    }
  });

  app.use((error, req, res, next) => {
    const message = error instanceof Error ? error.message : String(error);
    res.status(500).type("text").send(`SSR demo error: ${message}`);
  });

  return app;
}

export { createTinyChainClientAdapter };
