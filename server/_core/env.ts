// Read lazily so tests and tooling can set variables after import.
export const ENV = {
  get cookieSecret() {
    return process.env.JWT_SECRET ?? "";
  },
  get databaseUrl() {
    return process.env.DATABASE_URL ?? "";
  },
  get adminEmail() {
    return (process.env.ADMIN_EMAIL ?? "").trim().toLowerCase();
  },
  get isProduction() {
    return process.env.NODE_ENV === "production";
  },
};
