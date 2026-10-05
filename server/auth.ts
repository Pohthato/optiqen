import { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import type { User } from "../drizzle/schema";
import type { TrpcContext } from "./_core/context";
import { getSessionCookieOptions } from "./_core/cookies";
import { hashPassword, verifyPassword } from "./_core/passwords";
import { AttemptLimiter } from "./_core/rateLimit";
import { sdk } from "./_core/sdk";
import { publicProcedure, router } from "./_core/trpc";
import { createPasswordUser, getPasswordHash, getUserByEmail, recordSignIn } from "./db";

const MAX_FAILED_SIGN_INS = 10;
const SIGN_IN_WINDOW_MS = 15 * 60 * 1000;
const INCORRECT = "Email or password is incorrect.";

const signInLimiter = new AttemptLimiter(MAX_FAILED_SIGN_INS, SIGN_IN_WINDOW_MS);

const email = z.string().trim().toLowerCase().email().max(320);
const password = z.string().min(8, "Use at least 8 characters.").max(200);

function clientAddress(req: TrpcContext["req"]): string {
  // The right-most forwarded address is the one our hosting proxy appended.
  const forwarded = req.headers["x-forwarded-for"];
  const list = (Array.isArray(forwarded) ? forwarded.join(",") : forwarded ?? "").split(",").map(part => part.trim()).filter(Boolean);
  return list.at(-1) ?? req.socket?.remoteAddress ?? "unknown";
}

async function startSession(ctx: TrpcContext, user: User) {
  const token = await sdk.createSessionToken(user.openId, { name: user.name ?? "" });
  ctx.res.cookie(COOKIE_NAME, token, { ...getSessionCookieOptions(ctx.req), maxAge: ONE_YEAR_MS });
}

export const authRouter = router({
  me: publicProcedure.query(opts => opts.ctx.user),

  register: publicProcedure
    .input(z.object({ email, password, name: z.string().trim().min(1).max(100) }))
    .mutation(async ({ ctx, input }) => {
      if (await getUserByEmail(input.email)) {
        throw new TRPCError({ code: "CONFLICT", message: "An account with this email already exists." });
      }
      const user = await createPasswordUser({ email: input.email, name: input.name, passwordHash: await hashPassword(input.password) });
      await startSession(ctx, user);
      return user;
    }),

  login: publicProcedure.input(z.object({ email, password: z.string().min(1).max(200) })).mutation(async ({ ctx, input }) => {
    const key = `${input.email}|${clientAddress(ctx.req)}`;
    if (signInLimiter.isBlocked(key)) {
      throw new TRPCError({ code: "TOO_MANY_REQUESTS", message: "Too many attempts. Try again in 15 minutes." });
    }
    const user = await getUserByEmail(input.email);
    const stored = user ? await getPasswordHash(user.id) : undefined;
    // Hash anyway for unknown emails so response time does not reveal which emails exist.
    const valid = stored ? await verifyPassword(input.password, stored) : (await hashPassword(input.password), false);
    if (!user || !valid) {
      signInLimiter.recordFailure(key);
      throw new TRPCError({ code: "UNAUTHORIZED", message: INCORRECT });
    }
    signInLimiter.reset(key);
    await recordSignIn(user);
    await startSession(ctx, user);
    return user;
  }),

  logout: publicProcedure.mutation(({ ctx }) => {
    ctx.res.clearCookie(COOKIE_NAME, { ...getSessionCookieOptions(ctx.req), maxAge: -1 });
    return { success: true } as const;
  }),
});
