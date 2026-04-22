function requireEndpoint(endpoint) {
  if (typeof endpoint !== "string" || endpoint.length === 0) {
    throw new Error("TinyChain JS client requires a non-empty endpoint");
  }

  return endpoint;
}

function resolveUrl(endpoint, path) {
  if (typeof path !== "string" || path.length === 0) {
    throw new Error("TinyChain JS client requires a non-empty request path");
  }

  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  return new URL(path, endpoint).toString();
}

function normalizeHeaders(headers, hasBody) {
  const normalized = {
    accept: "application/json",
    ...headers
  };

  if (hasBody && !normalized["content-type"] && !normalized["Content-Type"]) {
    normalized["content-type"] = "application/json";
  }

  return normalized;
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

export function createTinyChainClient({ endpoint } = {}) {
  const baseEndpoint = requireEndpoint(endpoint);

  return {
    endpoint: baseEndpoint,
    async request({ method = "GET", path = "/healthz", headers, body } = {}) {
      const url = resolveUrl(baseEndpoint, path);
      const hasBody = body !== undefined;
      const response = await fetch(url, {
        method,
        headers: normalizeHeaders(headers, hasBody),
        body: hasBody ? JSON.stringify(body) : undefined
      });

      const payload = await parseResponse(response);
      if (!response.ok) {
        throw new Error(
          `TinyChain request failed (${method} ${path}): HTTP ${response.status} ${response.statusText}`
        );
      }

      return payload;
    },
    async get(path, options = {}) {
      return this.request({
        ...options,
        method: "GET",
        path
      });
    }
  };
}
