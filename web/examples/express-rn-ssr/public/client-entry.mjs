import { hydrateTinyChainWebDemo } from "/client/hydrate.mjs";

hydrateTinyChainWebDemo().catch((error) => {
  console.error("Hydration failed:", error);
});
