// Storage helpers for local development and production object storage.
// Uploads via Forge Server presigned URL to S3 (PUT direct).
// Downloads return /manus-storage/{key} paths served via 307 redirect.

import { ENV } from "./_core/env";
import fs from "node:fs/promises";
import path from "node:path";

function getForgeConfig() {
  const forgeUrl = ENV.forgeApiUrl;
  const forgeKey = ENV.forgeApiKey;

  if (!forgeUrl || !forgeKey) {
    throw new Error(
      "Storage config missing: set BUILT_IN_FORGE_API_URL and BUILT_IN_FORGE_API_KEY",
    );
  }

  return { forgeUrl: forgeUrl.replace(/\/+$/, ""), forgeKey };
}

function useLocalStorage() {
  return process.env.LOCAL_STORAGE_ENABLED === "true";
}

function localStorageRoot() {
  return path.resolve(process.env.LOCAL_STORAGE_DIR ?? "uploads");
}

function localStoragePath(key: string) {
  const root = localStorageRoot();
  const resolved = path.resolve(root, normalizeKey(key));
  if (!resolved.startsWith(`${root}${path.sep}`)) throw new Error("Invalid storage key");
  return resolved;
}

export function registerLocalStorageRoutes(app: { put: Function; get: Function }) {
  if (!useLocalStorage()) return;
  app.put("/api/local-storage/*", async (req: any, res: any) => {
    try {
      const destination = localStoragePath(req.params[0] as string);
      await fs.mkdir(path.dirname(destination), { recursive: true });
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", async () => {
        await fs.writeFile(destination, Buffer.concat(chunks));
        res.status(200).json({ ok: true });
      });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : "Upload failed" });
    }
  });
  app.get("/api/local-storage/*", (req: any, res: any) => {
    try {
      res.sendFile(localStoragePath(req.params[0] as string));
    } catch {
      res.status(404).json({ error: "File not found" });
    }
  });
}

function normalizeKey(relKey: string): string {
  return relKey.replace(/^\/+/, "");
}

function appendHashSuffix(relKey: string): string {
  const hash = crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  const lastDot = relKey.lastIndexOf(".");
  if (lastDot === -1) return `${relKey}_${hash}`;
  return `${relKey.slice(0, lastDot)}_${hash}${relKey.slice(lastDot)}`;
}

export async function storageCreateUploadUrl(
  relKey: string,
  contentType = "application/octet-stream",
): Promise<{ key: string; uploadUrl: string; contentType: string }> {
  if (useLocalStorage()) {
    const key = appendHashSuffix(normalizeKey(relKey));
    const baseUrl = (process.env.LOCAL_STORAGE_BASE_URL ?? process.env.PUBLIC_APP_URL ?? "http://localhost:3000").replace(/\/$/, "");
    return { key, uploadUrl: `${baseUrl}/api/local-storage/${key}`, contentType };
  }
  const { forgeUrl, forgeKey } = getForgeConfig();
  const key = appendHashSuffix(normalizeKey(relKey));
  const presignUrl = new URL("v1/storage/presign/put", forgeUrl + "/");
  presignUrl.searchParams.set("path", key);

  const presignResp = await fetch(presignUrl, {
    headers: { Authorization: `Bearer ${forgeKey}` },
  });
  if (!presignResp.ok) {
    const msg = await presignResp.text().catch(() => presignResp.statusText);
    throw new Error(`Storage upload preparation failed (${presignResp.status}): ${msg}`);
  }

  const { url: uploadUrl } = (await presignResp.json()) as { url: string };
  if (!uploadUrl) throw new Error("Forge returned an empty upload URL");
  return { key, uploadUrl, contentType };
}

export async function storagePut(
  relKey: string,
  data: Buffer | Uint8Array | string,
  contentType = "application/octet-stream",
): Promise<{ key: string; url: string }> {
  const { forgeUrl, forgeKey } = getForgeConfig();
  const key = appendHashSuffix(normalizeKey(relKey));

  // 1. Get presigned PUT URL from Forge
  const presignUrl = new URL("v1/storage/presign/put", forgeUrl + "/");
  presignUrl.searchParams.set("path", key);

  const presignResp = await fetch(presignUrl, {
    headers: { Authorization: `Bearer ${forgeKey}` },
  });

  if (!presignResp.ok) {
    const msg = await presignResp.text().catch(() => presignResp.statusText);
    throw new Error(`Storage presign failed (${presignResp.status}): ${msg}`);
  }

  const { url: s3Url } = (await presignResp.json()) as { url: string };
  if (!s3Url) throw new Error("Forge returned empty presign URL");

  // 2. PUT file directly to S3
  const blob =
    typeof data === "string"
      ? new Blob([data], { type: contentType })
      : new Blob([data as any], { type: contentType });

  const uploadResp = await fetch(s3Url, {
    method: "PUT",
    headers: { "Content-Type": contentType },
    body: blob,
  });

  if (!uploadResp.ok) {
    throw new Error(`Storage upload to S3 failed (${uploadResp.status})`);
  }

  return { key, url: `/manus-storage/${key}` };
}

export async function storageGet(relKey: string): Promise<{ key: string; url: string }> {
  const key = normalizeKey(relKey);
  return { key, url: `/manus-storage/${key}` };
}

export async function storageGetSignedUrl(relKey: string): Promise<string> {
  if (useLocalStorage()) {
    const baseUrl = (process.env.LOCAL_STORAGE_BASE_URL ?? process.env.PUBLIC_APP_URL ?? "http://localhost:3000").replace(/\/$/, "");
    return `${baseUrl}/api/local-storage/${normalizeKey(relKey)}`;
  }
  const { forgeUrl, forgeKey } = getForgeConfig();
  const key = normalizeKey(relKey);

  const getUrl = new URL("v1/storage/presign/get", forgeUrl + "/");
  getUrl.searchParams.set("path", key);

  const resp = await fetch(getUrl, {
    headers: { Authorization: `Bearer ${forgeKey}` },
  });

  if (!resp.ok) {
    const msg = await resp.text().catch(() => resp.statusText);
    throw new Error(`Storage signed URL failed (${resp.status}): ${msg}`);
  }

  const { url } = (await resp.json()) as { url: string };
  return url;
}
