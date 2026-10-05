import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { AddressInfo } from "node:net";
import http, { type Server } from "node:http";
import express from "express";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COOKIE_NAME } from "../shared/const";

let server: Server;
let base: string;
let root: string;

async function start() {
  vi.resetModules();
  const storage = await import("./storage");
  const { sdk } = await import("./_core/sdk");
  const db = await import("./db");
  const app = express();
  storage.registerStorageRoutes(app);
  server = await new Promise<Server>(resolve => {
    const s = app.listen(0, () => resolve(s));
  });
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const user = await db.createPasswordUser({ email: "a@example.com", name: "A", passwordHash: "scrypt$x" });
  const cookie = `${COOKIE_NAME}=${await sdk.createSessionToken(user.openId, { name: "A" })}`;
  return { storage, user, cookie };
}

describe("file storage on the app server", () => {
  beforeEach(() => {
    root = mkdtempSync(path.join(tmpdir(), "optiqen-storage-"));
    vi.stubEnv("STORAGE_DIR", root);
    vi.stubEnv("PUBLIC_APP_URL", "https://app.example.test/");
    vi.stubEnv("JWT_SECRET", "test-secret-with-enough-length-1234567890");
    vi.stubEnv("NODE_ENV", "development");
    vi.stubEnv("LOCAL_MEMORY_DB", "true");
  });
  afterEach(async () => {
    await new Promise(resolve => server?.close(resolve));
    vi.unstubAllEnvs();
    rmSync(root, { recursive: true, force: true });
  });

  it("hands the browser a same-origin upload URL and the worker an absolute download URL", async () => {
    const { storage, user } = await start();
    const upload = await storage.storageCreateUploadUrl(`analysis-sources/${user.id}/clip.mp4`, "video/mp4");
    expect(upload.uploadUrl).toBe(`/api/files/${upload.key}`);
    expect(upload.key).toMatch(new RegExp(`^analysis-sources/${user.id}/clip_[0-9a-f]{8}\\.mp4$`));
    expect(await storage.storageGetSignedUrl(upload.key)).toBe(`https://app.example.test/api/files/${upload.key}`);
  });

  it("stores a signed-in user's upload and serves it back", async () => {
    const { user, cookie } = await start();
    const key = `analysis-sources/${user.id}/clip_abc.mp4`;
    const put = await fetch(`${base}/api/files/${key}`, { method: "PUT", headers: { cookie, "content-type": "video/mp4" }, body: "video-bytes" });
    expect(put.status).toBe(200);
    const get = await fetch(`${base}/api/files/${key}`);
    expect(get.status).toBe(200);
    expect(await get.text()).toBe("video-bytes");
  });

  it("refuses uploads without a session or into another user's folder", async () => {
    const { user, cookie } = await start();
    const anonymous = await fetch(`${base}/api/files/analysis-sources/${user.id}/x.mp4`, { method: "PUT", body: "x" });
    expect(anonymous.status).toBe(401);
    const foreign = await fetch(`${base}/api/files/analysis-sources/${user.id + 1}/x.mp4`, { method: "PUT", headers: { cookie }, body: "x" });
    expect(foreign.status).toBe(403);
    const outside = await fetch(`${base}/api/files/other/x.mp4`, { method: "PUT", headers: { cookie }, body: "x" });
    expect(outside.status).toBe(403);
  });

  it("refuses uploads larger than the limit before writing them", async () => {
    const { storage, user, cookie } = await start();
    const key = `analysis-sources/${user.id}/big.mp4`;
    // fetch refuses a declared length its body does not match, so speak raw HTTP.
    const status = await new Promise<number>((resolve, reject) => {
      const request = http.request(`${base}/api/files/${key}`, {
        method: "PUT",
        headers: { cookie, "content-length": String(storage.MAX_UPLOAD_BYTES + 1) },
      });
      request.on("response", response => {
        response.resume();
        resolve(response.statusCode ?? 0);
        request.destroy();
      });
      request.on("error", reject);
      request.write("x");
    });
    expect(status).toBe(413);
    expect((await fetch(`${base}/api/files/${key}`)).status).toBe(404);
  });

  it("returns 404 for missing files and path traversal", async () => {
    await start();
    expect((await fetch(`${base}/api/files/analysis-sources/1/missing.mp4`)).status).toBe(404);
    expect((await fetch(`${base}/api/files/..%2f..%2fpackage.json`)).status).toBe(404);
  });
});
