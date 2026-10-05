import { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";
import { ForbiddenError } from "@shared/_core/errors";
import { parse as parseCookieHeader } from "cookie";
import type { Request } from "express";
import { SignJWT, jwtVerify } from "jose";
import type { User } from "../../drizzle/schema";
import * as db from "../db";
import { ENV } from "./env";

export type SessionPayload = { openId: string; name: string };

function sessionKey(): Uint8Array {
  if (!ENV.cookieSecret) throw new Error("JWT_SECRET must be set to sign sessions");
  return new TextEncoder().encode(ENV.cookieSecret);
}

async function createSessionToken(openId: string, options: { name?: string; expiresInMs?: number } = {}): Promise<string> {
  const expiresAt = Math.floor((Date.now() + (options.expiresInMs ?? ONE_YEAR_MS)) / 1000);
  return new SignJWT({ openId, name: options.name ?? "" })
    .setProtectedHeader({ alg: "HS256", typ: "JWT" })
    .setIssuedAt()
    .setExpirationTime(expiresAt)
    .sign(sessionKey());
}

async function verifySession(token: string | undefined | null): Promise<SessionPayload | null> {
  if (!token || !ENV.cookieSecret) return null;
  try {
    const { payload } = await jwtVerify(token, sessionKey(), { algorithms: ["HS256"] });
    const { openId, name } = payload as Record<string, unknown>;
    if (typeof openId !== "string" || !openId) return null;
    return { openId, name: typeof name === "string" ? name : "" };
  } catch {
    return null;
  }
}

/** The signed-in user from the session cookie; throws ForbiddenError otherwise. */
async function authenticateRequest(req: Request): Promise<User> {
  const token = parseCookieHeader(req.headers.cookie ?? "")[COOKIE_NAME];
  const session = await verifySession(token);
  if (!session) throw ForbiddenError("Invalid session cookie");
  const user = await db.getUserByOpenId(session.openId);
  if (!user) throw ForbiddenError("Unknown user");
  return user;
}

export const sdk = { createSessionToken, verifySession, authenticateRequest };
