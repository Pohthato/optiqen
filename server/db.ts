import { and, desc, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { nanoid } from "nanoid";
import { analysisSessions, AnalysisSession, InsertAnalysisSession, InsertUser, passwordCredentials, User, users } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;
const localUsers = new Map<string, User>();
const localSessions = new Map<string, AnalysisSession>();
const localCredentials = new Map<number, string>();
// Id 1 is the development auto-login user.
let localNextUserId = 2;

function useLocalMemory() {
  return process.env.NODE_ENV === "development" && process.env.LOCAL_MEMORY_DB === "true";
}

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  if (useLocalMemory()) {
    const existing = localUsers.get(user.openId);
    const now = new Date();
    localUsers.set(user.openId, {
      id: existing?.id ?? 1,
      openId: user.openId,
      name: user.name ?? existing?.name ?? null,
      email: user.email ?? existing?.email ?? null,
      loginMethod: user.loginMethod ?? existing?.loginMethod ?? "local",
      role: user.role ?? existing?.role ?? "user",
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
      lastSignedIn: user.lastSignedIn ?? now,
    });
    return;
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (ENV.adminEmail && user.email?.toLowerCase() === ENV.adminEmail) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  if (useLocalMemory()) return localUsers.get(openId);
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

export async function getUserByEmail(email: string): Promise<User | undefined> {
  const normalized = email.trim().toLowerCase();
  if (useLocalMemory()) return Array.from(localUsers.values()).find(user => user.email === normalized);
  const db = await getDb();
  if (!db) throw new Error("Database is not available");
  const result = await db.select().from(users).where(eq(users.email, normalized)).limit(1);
  return result[0];
}

/** Creates an email/password account; the caller checks the email is unused. */
export async function createPasswordUser(input: { email: string; name: string; passwordHash: string }): Promise<User> {
  const email = input.email.trim().toLowerCase();
  const openId = `pw_${nanoid(21)}`;
  const role = ENV.adminEmail && email === ENV.adminEmail ? "admin" : "user";
  if (useLocalMemory()) {
    const now = new Date();
    const user: User = { id: localNextUserId++, openId, name: input.name, email, loginMethod: "email", role, createdAt: now, updatedAt: now, lastSignedIn: now };
    localUsers.set(openId, user);
    localCredentials.set(user.id, input.passwordHash);
    return user;
  }
  const db = await getDb();
  if (!db) throw new Error("Database is not available");
  await db.insert(users).values({ openId, name: input.name, email, loginMethod: "email", role, lastSignedIn: new Date() });
  const [user] = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  await db.insert(passwordCredentials).values({ userId: user.id, passwordHash: input.passwordHash });
  return user;
}

export async function getPasswordHash(userId: number): Promise<string | undefined> {
  if (useLocalMemory()) return localCredentials.get(userId);
  const db = await getDb();
  if (!db) throw new Error("Database is not available");
  const result = await db.select().from(passwordCredentials).where(eq(passwordCredentials.userId, userId)).limit(1);
  return result[0]?.passwordHash;
}

export async function recordSignIn(user: User): Promise<void> {
  const now = new Date();
  if (useLocalMemory()) {
    localUsers.set(user.openId, { ...user, lastSignedIn: now });
    return;
  }
  const db = await getDb();
  if (!db) return;
  await db.update(users).set({ lastSignedIn: now }).where(eq(users.id, user.id));
}

export async function createAnalysisSession(session: InsertAnalysisSession) {
  if (useLocalMemory()) {
    const now = new Date();
    localSessions.set(session.id, {
      ...session,
      status: session.status ?? "draft",
      workerJobId: session.workerJobId ?? null,
      failureReason: session.failureReason ?? null,
      lastWorkerStatusAt: session.lastWorkerStatusAt ?? null,
      result: session.result ?? null,
      createdAt: session.createdAt ?? now,
      updatedAt: session.updatedAt ?? now,
    } as AnalysisSession);
    return session.id;
  }
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session storage");

  await db.insert(analysisSessions).values(session);
  return session.id;
}

export async function getAnalysisSessionForUser(id: string, userId: number) {
  if (useLocalMemory()) {
    const session = localSessions.get(id);
    return session?.userId === userId ? session : undefined;
  }
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session lookup");

  const rows = await db.select().from(analysisSessions).where(and(eq(analysisSessions.id, id), eq(analysisSessions.userId, userId))).limit(1);
  return rows[0];
}

/** Only server-to-server completion code may use this lookup. */
export async function getAnalysisSessionById(id: string) {
  if (useLocalMemory()) return localSessions.get(id);
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session lookup");
  const rows = await db.select().from(analysisSessions).where(eq(analysisSessions.id, id)).limit(1);
  return rows[0];
}

export async function listAnalysisSessionsForUser(userId: number) {
  if (useLocalMemory()) {
    return Array.from(localSessions.values())
      .filter(session => session.userId === userId)
      .sort((left, right) => right.createdAt.getTime() - left.createdAt.getTime())
      .slice(0, 24);
  }
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session lookup");

  return db.select().from(analysisSessions).where(eq(analysisSessions.userId, userId)).orderBy(desc(analysisSessions.createdAt)).limit(24);
}

export async function updateAnalysisSessionForUser(
  id: string,
  userId: number,
  update: Pick<InsertAnalysisSession, "status" | "workerJobId" | "result" | "failureReason" | "lastWorkerStatusAt">
) {
  if (useLocalMemory()) {
    const session = localSessions.get(id);
    if (session?.userId === userId) {
      localSessions.set(id, { ...session, ...update, updatedAt: new Date() } as AnalysisSession);
    }
    return;
  }
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session updates");

  await db.update(analysisSessions).set(update).where(and(eq(analysisSessions.id, id), eq(analysisSessions.userId, userId)));
}

export async function updateAnalysisSessionFromWorker(
  id: string,
  workerJobId: string,
  update: Pick<InsertAnalysisSession, "status" | "result" | "failureReason" | "lastWorkerStatusAt">
) {
  if (useLocalMemory()) {
    const session = localSessions.get(id);
    if (session?.workerJobId === workerJobId) {
      localSessions.set(id, { ...session, ...update, updatedAt: new Date() } as AnalysisSession);
    }
    return;
  }
  const db = await getDb();
  if (!db) throw new Error("Database is not available for analysis session updates");
  // Match the job id as well as the session. This makes stale RunPod
  // callbacks unable to overwrite a retried analysis.
  await db.update(analysisSessions).set(update).where(and(eq(analysisSessions.id, id), eq(analysisSessions.workerJobId, workerJobId)));
}
