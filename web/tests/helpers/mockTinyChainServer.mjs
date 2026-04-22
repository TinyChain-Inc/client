import http from "node:http";

function createCorsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, PUT, DELETE, OPTIONS",
    "access-control-allow-headers": "content-type, authorization"
  };
}

export function createMockTinyChainServer(options = {}) {
  const { cors = false } = options;

  return http.createServer((req, res) => {
    const corsHeaders = cors ? createCorsHeaders() : {};

    if (cors && req.method === "OPTIONS") {
      res.writeHead(204, corsHeaders);
      res.end();
      return;
    }

    if (req.url === "/healthz" && req.method === "GET") {
      const payload = JSON.stringify({
        ok: true,
        service: "mock-tinychain"
      });

      res.writeHead(200, {
        ...corsHeaders,
        "content-type": "application/json",
        "content-length": Buffer.byteLength(payload)
      });
      res.end(payload);
      return;
    }

    res.writeHead(404, {
      ...corsHeaders,
      "content-type": "application/json"
    });
    res.end(JSON.stringify({ error: "not found" }));
  });
}

export function listen(server, options = {}) {
  const { port = 0, host = "127.0.0.1" } = options;

  return new Promise((resolve, reject) => {
    const onError = (error) => {
      reject(error);
    };

    server.once("error", onError);
    server.listen(port, host, () => {
      server.off("error", onError);

      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("Failed to determine mock TinyChain server address"));
        return;
      }

      resolve(address.port);
    });
  });
}

export function close(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }

      resolve();
    });
  });
}
