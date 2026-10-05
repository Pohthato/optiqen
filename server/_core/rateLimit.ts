/** Counts failures per key in a fixed window; a key is blocked once it reaches the limit. */
export class AttemptLimiter {
  private readonly attempts = new Map<string, { count: number; resetAt: number }>();

  constructor(private readonly limit: number, private readonly windowMs: number) {}

  isBlocked(key: string, now = Date.now()): boolean {
    const entry = this.attempts.get(key);
    if (!entry) return false;
    if (entry.resetAt <= now) {
      this.attempts.delete(key);
      return false;
    }
    return entry.count >= this.limit;
  }

  recordFailure(key: string, now = Date.now()): void {
    const entry = this.attempts.get(key);
    if (!entry || entry.resetAt <= now) {
      this.attempts.set(key, { count: 1, resetAt: now + this.windowMs });
    } else {
      entry.count += 1;
    }
    if (this.attempts.size > 10_000) {
      this.attempts.forEach((value, stale) => {
        if (value.resetAt <= now) this.attempts.delete(stale);
      });
    }
  }

  reset(key: string): void {
    this.attempts.delete(key);
  }
}
