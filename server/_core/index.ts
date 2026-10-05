import "dotenv/config";
import express from "express";
import { createServer } from "http";
import net from "net";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { appRouter } from "../routers";
import { createContext } from "./context";
import { serveStatic, setupVite } from "./vite";
import crypto from "node:crypto";
import { reconcileAnalysisWorkerJob } from "../analysisCompletion";
import { getWorkerCallbackToken } from "../analysisWorker";
import { startKeepWarmScheduler } from "../warmupScheduler";
import { registerStorageRoutes } from "../storage";

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise(resolve => {
    const server = net.createServer();
    server.listen(port, () => {
      server.close(() => resolve(true));
    });
    server.on("error", () => resolve(false));
  });
}

async function findAvailablePort(startPort: number = 3000): Promise<number> {
  for (let port = startPort; port < startPort + 20; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found starting from ${startPort}`);
}

async function startServer() {
  // Keep one serverless GPU warm (when configured) so the first analysis of a
  // product-hour skips the multi-minute cold start.
  startKeepWarmScheduler();
  const app = express();
  const server = createServer(app);
  // Configure body parser with larger size limit for file uploads
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));
  registerStorageRoutes(app);
  app.post("/api/worker-complete/:analysisId", async (req, res) => {
    const expected = getWorkerCallbackToken(req.params.analysisId);
    const received = typeof req.query.token === "string" ? req.query.token : "";
    if (!expected || received.length !== expected.length || !crypto.timingSafeEqual(Buffer.from(received), Buffer.from(expected))) {
      res.status(401).json({ error: "Invalid worker callback token." });
      return;
    }
    try {
      const result = await reconcileAnalysisWorkerJob(req.params.analysisId);
      // A valid callback for a deleted/unknown session should not become an
      // enumeration oracle, but it has still been safely handled.
      res.status(200).json({ accepted: true, status: result.status });
    } catch (error) {
      console.error("[Worker callback] reconciliation failed", error);
      // RunPod retries a non-200 webhook. Let transient storage/API failures retry.
      res.status(503).json({ error: "Temporary completion reconciliation failure." });
    }
  });
  // tRPC API
  app.use(
    "/api/trpc",
    createExpressMiddleware({
      router: appRouter,
      createContext,
    })
  );
  // development mode uses Vite, production mode uses static files
  if (process.env.NODE_ENV === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const preferredPort = parseInt(process.env.PORT || "3000");
  const port = await findAvailablePort(preferredPort);

  if (port !== preferredPort) {
    console.log(`Port ${preferredPort} is busy, using port ${port} instead`);
  }

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
