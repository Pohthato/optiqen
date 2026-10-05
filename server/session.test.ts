import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SignJWT } from "jose";
import { COOKIE_NAME } from "../shared/const";

async function loadSdk() {
  vi.resetModules();
  return (await import("./_core/sdk")).sdk;
}

function requestWith(headers: Record<string, string>) {
  return { headers } as never;
}

describe("sessions", () => {
  beforeEach(() => {
    vi.stubEnv("JWT_SECRET", "test-secret-with-enough-length-1234567890");
    vi.stubEnv("NODE_ENV", "development");
    vi.stubEnv("LOCAL_MEMORY_DB", "true");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("round-trips a signed session token", async () => {
    const sdk = await loadSdk();
    const token = await sdk.createSessionToken("pw_abc", { name: "Andrew" });
    expect(await sdk.verifySession(token)).toEqual({ openId: "pw_abc", name: "Andrew" });
  });

  it("rejects a tampered, foreign or expired token", async () => {
    const sdk = await loadSdk();
    const token = await sdk.createSessionToken("pw_abc", { name: "Andrew" });
    expect(await sdk.verifySession(`${token.slice(0, -2)}xx`)).toBeNull();
    const foreign = await new SignJWT({ openId: "pw_abc", name: "x" })
      .setProtectedHeader({ alg: "HS256" })
      .setExpirationTime("1h")
      .sign(new TextEncoder().encode("some-other-secret"));
    expect(await sdk.verifySession(foreign)).toBeNull();
    const expired = await sdk.createSessionToken("pw_abc", { name: "Andrew", expiresInMs: -1000 });
    expect(await sdk.verifySession(expired)).toBeNull();
    expect(await sdk.verifySession(undefined)).toBeNull();
  });

  it("authenticates a request only from the session cookie of a known user", async () => {
    const sdk = await loadSdk();
    const db = await import("./db");
    const user = await db.createPasswordUser({ email: "andrew@example.com", name: "Andrew", passwordHash: "scrypt$x" });
    const token = await sdk.createSessionToken(user.openId, { name: "Andrew" });
    const authed = await sdk.authenticateRequest(requestWith({ cookie: `${COOKIE_NAME}=${token}` }));
    expect(authed.email).toBe("andrew@example.com");
    await expect(sdk.authenticateRequest(requestWith({ authorization: `Bearer ${token}` }))).rejects.toThrow();
    const stranger = await sdk.createSessionToken("pw_nobody", { name: "x" });
    await expect(sdk.authenticateRequest(requestWith({ cookie: `${COOKIE_NAME}=${stranger}` }))).rejects.toThrow();
  });
});
