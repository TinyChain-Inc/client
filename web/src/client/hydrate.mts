import React from "react";
import { hydrateRoot } from "react-dom/client";

import { App } from "../shared/App.mts";
import { APP_ROOT_ID, WINDOW_STATE_KEY } from "../shared/constants.mts";
import { createTinyChainClientAdapter } from "../shared/tinychain/adapter.mts";

function readInitialState() {
  const value = window[WINDOW_STATE_KEY];
  if (value && typeof value === "object") {
    return value;
  }

  return {
    tinychain: {
      source: "bootstrap",
      packageName: "uninitialized",
      runtime: "browser",
      scope: "hydrate-bootstrap",
      endpoint: null,
      requestCount: 0,
      fetchedAt: new Date().toISOString(),
      message: "No initial state found in window.",
    },
  };
}

export async function hydrateTinyChainWebDemo() {
  const root = document.getElementById(APP_ROOT_ID);
  if (!root) {
    return;
  }

  const initialState = readInitialState();
  const tinychainClient = await createTinyChainClientAdapter({
    runtime: "browser",
    endpoint: initialState.tinychain?.endpoint,
  });

  hydrateRoot(
    root,
    React.createElement(App, {
      initialData: initialState.tinychain,
      tinychainClient,
    })
  );
}

export { createTinyChainClientAdapter };
