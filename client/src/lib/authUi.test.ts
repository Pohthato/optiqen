import { describe, expect, it } from "vitest";
import { getAuthAction, getAuthButtonLabel, loginPath, safeNextPath } from "./authUi";

describe("auth control state", () => {
  it("opens a profile menu for authenticated users", () => {
    const action = getAuthAction(true);
    expect(action).toBe("profile");
    expect(getAuthButtonLabel(action)).toBe("Open profile menu");
  });

  it("starts sign-in for everyone else", () => {
    const action = getAuthAction(false);
    expect(action).toBe("sign_in");
    expect(getAuthButtonLabel(action)).toBe("Sign in to analyze");
  });
});

describe("sign-in redirects", () => {
  it("sends the visitor to the sign-in page with where they were", () => {
    expect(loginPath("/", "")).toBe("/login?next=%2F");
    expect(loginPath("/sessions/abc", "?tab=coach")).toBe("/login?next=%2Fsessions%2Fabc%3Ftab%3Dcoach");
  });

  it("only returns to paths on this site", () => {
    expect(safeNextPath("/sessions/abc?tab=coach")).toBe("/sessions/abc?tab=coach");
    expect(safeNextPath(null)).toBe("/");
    expect(safeNextPath("https://evil.example")).toBe("/");
    expect(safeNextPath("//evil.example/path")).toBe("/");
    expect(safeNextPath("/\\evil.example")).toBe("/");
    expect(safeNextPath("/login?next=%2F")).toBe("/");
  });
});
