import { randomBytes, scrypt, timingSafeEqual } from "node:crypto";

// scrypt cost parameters; stored with each hash so they can be raised later.
const COST = 16384;
const BLOCK_SIZE = 8;
const PARALLELISM = 1;
const KEY_LENGTH = 64;
const MAX_MEMORY = 64 * 1024 * 1024;

function derive(password: string, salt: Buffer, length: number, cost: number, blockSize: number, parallelism: number) {
  return new Promise<Buffer>((resolve, reject) => {
    scrypt(password, salt, length, { N: cost, r: blockSize, p: parallelism, maxmem: MAX_MEMORY }, (error, key) =>
      error ? reject(error) : resolve(key)
    );
  });
}

/** `scrypt$N$r$p$salt$key`, salt and key base64. */
export async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16);
  const key = await derive(password, salt, KEY_LENGTH, COST, BLOCK_SIZE, PARALLELISM);
  return ["scrypt", COST, BLOCK_SIZE, PARALLELISM, salt.toString("base64"), key.toString("base64")].join("$");
}

export async function verifyPassword(password: string, stored: string): Promise<boolean> {
  const parts = stored.split("$");
  if (parts.length !== 6 || parts[0] !== "scrypt") return false;
  const [cost, blockSize, parallelism] = parts.slice(1, 4).map(Number);
  const salt = Buffer.from(parts[4], "base64");
  const expected = Buffer.from(parts[5], "base64");
  if (![cost, blockSize, parallelism].every(Number.isInteger) || salt.length === 0 || expected.length === 0) return false;
  try {
    const key = await derive(password, salt, expected.length, cost, blockSize, parallelism);
    return timingSafeEqual(key, expected);
  } catch {
    return false;
  }
}
