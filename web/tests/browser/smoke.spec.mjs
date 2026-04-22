import { test, expect } from "@playwright/test";
import { close, createMockTinyChainServer, listen } from "../helpers/mockTinyChainServer.mjs";

const TINYCHAIN_MOCK_PORT = 18702;

let mockTinyChain;

test.beforeAll(async () => {
  mockTinyChain = createMockTinyChainServer({ cors: true });
  await listen(mockTinyChain, { port: TINYCHAIN_MOCK_PORT });
});

test.afterAll(async () => {
  if (mockTinyChain) {
    await close(mockTinyChain);
  }
});

test("hydrates and refreshes TinyChain data in the browser", async ({ page }) => {
  await page.goto("/", { waitUntil: "networkidle" });

  await expect(page.getByTestId("row-value-runtime")).toHaveText("server");
  await page.getByTestId("refresh-button").click();
  await expect(page.getByTestId("row-value-runtime")).toHaveText("browser");
});
