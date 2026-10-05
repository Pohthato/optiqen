// Video files live on the app server's disk (STORAGE_DIR, default ./uploads).
// Browsers upload to a same-origin URL with their session cookie; the GPU worker
// downloads from an absolute URL whose key is unguessable.
// On hosts with an ephemeral filesystem, mount a volume at STORAGE_DIR.

import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import type { Express, Request, Response } from "express";
import { sdk } from "./_core/sdk";

export const MAX_UPLOAD_BYTES = 2_000_000_000;
const ROUTE = "/api/files";
const UPLOAD_PREFIX = "analysis-sources";

function storageRoot() {
  return path.resolve(process.env.STORAGE_DIR ?? "uploads");
}

function publicBaseUrl() {
  return (process.env.PUBLIC_APP_URL ?? "http://localhost:3000").replace(/\/+$/, "");
}

function normalizeKey(relKey: string): string {
  return relKey.replace(/^\/+/, "");
}

/** Absolute path for a key; throws if the key escapes the storage root. */
function storagePath(key: string): string {
  const root = storageRoot();
  const resolved = path.resolve(root, normalizeKey(key));
  if (!resolved.startsWith(`${root}${path.sep}`)) throw new Error("Invalid storage key");
  return resolved;
}

function appendHashSuffix(relKey: string): string {
  const hash = crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  const lastDot = relKey.lastIndexOf(".");
  if (lastDot === -1) return `${relKey}_${hash}`;
  return `${relKey.slice(0, lastDot)}_${hash}${relKey.slice(lastDot)}`;
}

/** The user id that owns a key (`analysis-sources/{userId}/...`), or null. */
function ownerOfKey(key: string): number | null {
  const [prefix, owner] = normalizeKey(key).split("/");
  return prefix === UPLOAD_PREFIX && /^\d+$/.test(owner ?? "") ? Number(owner) : null;
}

export async function storageCreateUploadUrl(
  relKey: string,
  contentType = "application/octet-stream",
): Promise<{ key: string; uploadUrl: string; contentType: string }> {
  const key = appendHashSuffix(normalizeKey(relKey));
  return { key, uploadUrl: `${ROUTE}/${key}`, contentType };
}

export async function storageGetSignedUrl(relKey: string): Promise<string> {
  return `${publicBaseUrl()}${ROUTE}/${normalizeKey(relKey)}`;
}

async function receiveUpload(req: Request, res: Response) {
  let userId: number;
  try {
    userId = (await sdk.authenticateRequest(req)).id;
  } catch {
    res.status(401).json({ error: "Sign in to upload." });
    return;
  }
  const key = normalizeKey((req.params as Record<string, string>)[0] ?? "");
  if (ownerOfKey(key) !== userId) {
    res.status(403).json({ error: "You can only upload into your own folder." });
    return;
  }
  if (Number(req.headers["content-length"] ?? 0) > MAX_UPLOAD_BYTES) {
    res.status(413).json({ error: "Upload is larger than 2 GB." });
    return;
  }
  let destination: string;
  try {
    destination = storagePath(key);
  } catch {
    res.status(400).json({ error: "Invalid storage key." });
    return;
  }
  await fsp.mkdir(path.dirname(destination), { recursive: true });
  const partial = `${destination}.partial`;
  const out = fs.createWriteStream(partial);
  let received = 0;
  let failed = false;
  const fail = (status: number, message: string) => {
    if (failed) return;
    failed = true;
    req.unpipe(out);
    out.destroy();
    fs.rm(partial, { force: true }, () => undefined);
    if (!res.headersSent) res.status(status).json({ error: message });
  };
  req.on("data", (chunk: Buffer) => {
    received += chunk.length;
    if (received > MAX_UPLOAD_BYTES) fail(413, "Upload is larger than 2 GB.");
  });
  req.on("aborted", () => fail(400, "Upload was interrupted."));
  out.on("error", () => fail(500, "Upload could not be saved."));
  out.on("finish", async () => {
    if (failed) return;
    try {
      await fsp.rename(partial, destination);
      res.status(200).json({ ok: true });
    } catch {
      fail(500, "Upload could not be saved.");
    }
  });
  req.pipe(out);
}

function serveFile(req: Request, res: Response) {
  let file: string;
  try {
    file = storagePath((req.params as Record<string, string>)[0] ?? "");
  } catch {
    res.status(404).json({ error: "File not found." });
    return;
  }
  res.sendFile(file, error => {
    if (error && !res.headersSent) res.status(404).json({ error: "File not found." });
  });
}

export function registerStorageRoutes(app: Express) {
  app.put(`${ROUTE}/*`, (req, res) => void receiveUpload(req, res));
  app.get(`${ROUTE}/*`, serveFile);
}
