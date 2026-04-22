import { createTinyChainClient } from "@tinychain/js";

function hasRequestApi(rawClient) {
  return Boolean(rawClient && typeof rawClient.request === "function");
}

function toObjectPayload(value) {
  if (value && typeof value === "object") {
    return value;
  }

  return { value };
}

function createRealTinyChainClient(rawClient, { runtime, endpoint, packageName }) {
  let requestCount = 0;

  if (!hasRequestApi(rawClient)) {
    throw new Error(
      `Loaded TinyChain JS client from "${packageName}" but request(...) is unavailable`
    );
  }

  return {
    source: "real",
    packageName,
    async fetchDemoData({ scope = "unknown" } = {}) {
      requestCount += 1;
      const payload = await rawClient.request({ method: "GET", path: "/healthz" });
      return {
        ...toObjectPayload(payload),
        source: "real",
        packageName,
        runtime,
        scope,
        endpoint: endpoint ?? null,
        requestCount,
        fetchedAt: new Date().toISOString(),
        message: "Loaded real TinyChain package and executed request(GET /healthz).",
      };
    },
  };
}

export async function createTinyChainClientAdapter({ runtime, endpoint } = {}) {
  const resolvedRuntime = runtime ?? "server";
  const rawClient = await createTinyChainClient({
    endpoint
  });

  return createRealTinyChainClient(rawClient, {
    runtime: resolvedRuntime,
    endpoint,
    packageName: "@tinychain/js",
  });
}
