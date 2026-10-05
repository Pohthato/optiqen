export type AuthAction = "profile" | "sign_in";

export function getAuthAction(isAuthenticated: boolean): AuthAction {
  return isAuthenticated ? "profile" : "sign_in";
}

export function getAuthButtonLabel(action: AuthAction) {
  return action === "profile" ? "Open profile menu" : "Sign in to analyze";
}

/** The sign-in page, remembering where the visitor was. */
export function loginPath(pathname: string, search: string) {
  return `/login?next=${encodeURIComponent(`${pathname}${search}`)}`;
}

/** Where to go after signing in: only a path on this site, never back to /login. */
export function safeNextPath(next: string | null) {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\") || next.startsWith("/login")) return "/";
  return next;
}
