import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COOKIE_NAME } from "../shared/const";

type CookieSet = { name: string; value: string; options: Record<string, unknown> };

async function freshRouter() {
  vi.resetModules();
  return (await import("./routers")).appRouter;
}

function context(ip = "203.0.113.7") {
  const cookies: CookieSet[] = [];
  const ctx = {
    user: null,
    req: { protocol: "https", headers: { "x-forwarded-for": ip }, socket: { remoteAddress: ip } },
    res: {
      cookie: (name: string, value: string, options: Record<string, unknown>) => cookies.push({ name, value, options }),
      clearCookie: vi.fn(),
    },
  };
  return { ctx: ctx as never, cookies };
}

describe("email and password sign-in", () => {
  beforeEach(() => {
    vi.stubEnv("JWT_SECRET", "test-secret-with-enough-length-1234567890");
    vi.stubEnv("NODE_ENV", "development");
    vi.stubEnv("LOCAL_MEMORY_DB", "true");
    vi.stubEnv("ADMIN_EMAIL", "owner@example.com");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("registers an account, sets a session cookie and never returns the password hash", async () => {
    const router = await freshRouter();
    const { ctx, cookies } = context();
    const user = await router.createCaller(ctx).auth.register({ email: " Andrew@Example.com ", password: "long enough pw", name: "Andrew" });
    expect(user.email).toBe("andrew@example.com");
    expect(JSON.stringify(user)).not.toContain("scrypt");
    expect(cookies).toHaveLength(1);
    expect(cookies[0].name).toBe(COOKIE_NAME);
    expect(cookies[0].options).toMatchObject({ httpOnly: true, sameSite: "lax" });
  });

  it("refuses a second account with the same email", async () => {
    const router = await freshRouter();
    await router.createCaller(context().ctx).auth.register({ email: "a@example.com", password: "long enough pw", name: "A" });
    await expect(
      router.createCaller(context().ctx).auth.register({ email: "A@example.com", password: "another long pw", name: "B" })
    ).rejects.toMatchObject({ code: "CONFLICT" });
  });

  it("signs in with the right password and gives the same answer for a wrong password or unknown email", async () => {
    const router = await freshRouter();
    await router.createCaller(context().ctx).auth.register({ email: "a@example.com", password: "long enough pw", name: "A" });
    const { ctx, cookies } = context();
    const user = await router.createCaller(ctx).auth.login({ email: "a@example.com", password: "long enough pw" });
    expect(user.email).toBe("a@example.com");
    expect(cookies).toHaveLength(1);
    const wrong = router.createCaller(context().ctx).auth.login({ email: "a@example.com", password: "wrong password!" });
    await expect(wrong).rejects.toMatchObject({ code: "UNAUTHORIZED", message: "Email or password is incorrect." });
    const unknown = router.createCaller(context().ctx).auth.login({ email: "nobody@example.com", password: "long enough pw" });
    await expect(unknown).rejects.toMatchObject({ code: "UNAUTHORIZED", message: "Email or password is incorrect." });
  });

  it("rejects short passwords and malformed emails", async () => {
    const router = await freshRouter();
    await expect(router.createCaller(context().ctx).auth.register({ email: "a@example.com", password: "short", name: "A" })).rejects.toThrow();
    await expect(router.createCaller(context().ctx).auth.register({ email: "not-an-email", password: "long enough pw", name: "A" })).rejects.toThrow();
  });

  it("locks out repeated wrong passwords for an email and address", async () => {
    const router = await freshRouter();
    await router.createCaller(context().ctx).auth.register({ email: "a@example.com", password: "long enough pw", name: "A" });
    for (let i = 0; i < 10; i++) {
      await expect(router.createCaller(context().ctx).auth.login({ email: "a@example.com", password: "wrong password!" })).rejects.toMatchObject({ code: "UNAUTHORIZED" });
    }
    await expect(
      router.createCaller(context().ctx).auth.login({ email: "a@example.com", password: "long enough pw" })
    ).rejects.toMatchObject({ code: "TOO_MANY_REQUESTS" });
    const elsewhere = await router.createCaller(context("198.51.100.4").ctx).auth.login({ email: "a@example.com", password: "long enough pw" });
    expect(elsewhere.email).toBe("a@example.com");
  });

  it("makes the configured admin email an admin", async () => {
    const router = await freshRouter();
    const owner = await router.createCaller(context().ctx).auth.register({ email: "owner@example.com", password: "long enough pw", name: "Owner" });
    const other = await router.createCaller(context().ctx).auth.register({ email: "user@example.com", password: "long enough pw", name: "User" });
    expect(owner.role).toBe("admin");
    expect(other.role).toBe("user");
  });
});
