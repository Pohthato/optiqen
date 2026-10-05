import { index, int, json, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Stable unique account id; email accounts use `pw_...`. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/** Password hashes live apart from `users` so a user row can never carry one to the client. */
export const passwordCredentials = mysqlTable("passwordCredentials", {
  userId: int("userId").primaryKey().references(() => users.id, { onDelete: "cascade" }),
  passwordHash: varchar("passwordHash", { length: 255 }).notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export const analysisSessions = mysqlTable("analysisSessions", {
  id: varchar("id", { length: 28 }).primaryKey(),
  userId: int("userId").notNull().references(() => users.id, { onDelete: "cascade" }),
  sourceName: varchar("sourceName", { length: 255 }).notNull(),
  sourceStorageKey: varchar("sourceStorageKey", { length: 768 }).notNull(),
  sourceDurationMs: int("sourceDurationMs"),
  selectedPlayer: mysqlEnum("selectedPlayer", ["near", "far"]).notNull(),
  status: mysqlEnum("status", ["draft", "queued", "processing", "completed", "failed"]).default("draft").notNull(),
  calibration: json("calibration").notNull(),
  workerJobId: varchar("workerJobId", { length: 128 }),
  failureReason: text("failureReason"),
  lastWorkerStatusAt: timestamp("lastWorkerStatusAt"),
  result: json("result"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
}, (table) => [
  index("analysisSessions_user_created_idx").on(table.userId, table.createdAt),
  index("analysisSessions_worker_job_idx").on(table.workerJobId),
]);

export type AnalysisSession = typeof analysisSessions.$inferSelect;
export type InsertAnalysisSession = typeof analysisSessions.$inferInsert;
