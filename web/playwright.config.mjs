import { defineConfig } from "@playwright/test";

const APP_PORT = 3110;
const TINYCHAIN_MOCK_PORT = 18702;
const TINYCHAIN_ENDPOINT = `http://127.0.0.1:${TINYCHAIN_MOCK_PORT}`;

export default defineConfig({
  testDir: "./tests/browser",
  testMatch: "**/*.spec.mjs",
  timeout: 30000,
  expect: {
    timeout: 10000
  },
  use: {
    baseURL: `http://127.0.0.1:${APP_PORT}`,
    headless: true
  },
  webServer: {
    command:
      `npm run build && ` +
      `TC_WEB_PORT=${APP_PORT} ` +
      `TC_TINYCHAIN_HTTP_URL=${TINYCHAIN_ENDPOINT} ` +
      `TC_PUBLIC_TINYCHAIN_HTTP_URL=${TINYCHAIN_ENDPOINT} ` +
      `npm run start`,
    url: `http://127.0.0.1:${APP_PORT}/healthz`,
    timeout: 120000,
    reuseExistingServer: !process.env.CI
  }
});
