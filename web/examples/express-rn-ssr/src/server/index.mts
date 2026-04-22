import { createServer } from "../../../../src/server/index.mts";

export { createServer, createTinyChainClientAdapter } from "../../../../src/server/index.mts";

if (typeof require !== "undefined" && typeof module !== "undefined" && require.main === module) {
  const port = Number(process.env.TC_WEB_PORT ?? process.env.PORT ?? 3000);
  createServer().listen(port, () => {
    console.log(`TinyChain web demo listening on http://localhost:${port}`);
  });
}
