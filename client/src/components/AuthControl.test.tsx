import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { AuthControl } from "./AuthControl";

const props = {
  profileMenuOpen: false,
  onAuthClick: vi.fn(),
  onLogout: vi.fn(),
};

describe("AuthControl", () => {
  it("renders an actionable profile button and open menu for authenticated users", () => {
    const markup = renderToStaticMarkup(<AuthControl {...props} action="profile" label="Open profile menu" profileMenuOpen={true} />);
    expect(markup).toContain('aria-label="Open profile menu"');
    expect(markup).toContain("Sign out");
  });

  it("renders the sign-in action label for unauthenticated users", () => {
    const markup = renderToStaticMarkup(<AuthControl {...props} action="sign_in" label="Sign in to analyze" />);
    expect(markup).toContain('aria-label="Sign in to analyze"');
    expect(markup).toContain(">?</button>");
    expect(markup).not.toContain("Sign out");
  });
});
