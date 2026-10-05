import { loginPath } from "@/lib/authUi";

export { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";

/** Send the visitor to the sign-in page; they come back here afterwards. */
export const startLogin = () => {
  window.location.assign(loginPath(window.location.pathname, window.location.search));
};
