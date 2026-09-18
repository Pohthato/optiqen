import type { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import type { User } from "../../drizzle/schema";
import { sdk } from "./sdk";
import * as db from "../db";

const LOCAL_USER_OPEN_ID = "netoval-local-user";

export type TrpcContext = {
  req: CreateExpressContextOptions["req"];
  res: CreateExpressContextOptions["res"];
  user: User | null;
};

export async function createContext(
  opts: CreateExpressContextOptions
): Promise<TrpcContext> {
  let user: User | null = null;

  if (process.env.NODE_ENV === "development" && process.env.LOCAL_AUTH_ENABLED !== "false") {
    await db.upsertUser({
      openId: LOCAL_USER_OPEN_ID,
      name: "Local Netoval User",
      email: "local@netoval.test",
      loginMethod: "local",
    });
    user = (await db.getUserByOpenId(LOCAL_USER_OPEN_ID)) ?? {
      id: 1,
      openId: LOCAL_USER_OPEN_ID,
      name: "Local Netoval User",
      email: "local@netoval.test",
      loginMethod: "local",
      role: "user",
      createdAt: new Date(0),
      updatedAt: new Date(),
      lastSignedIn: new Date(),
    };
    return { req: opts.req, res: opts.res, user };
  }

  try {
    user = await sdk.authenticateRequest(opts.req);
  } catch (error) {
    // Authentication is optional for public procedures.
    user = null;
  }

  return {
    req: opts.req,
    res: opts.res,
    user,
  };
}
