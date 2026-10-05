// @vitest-environment jsdom
import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { loginMock, registerMock, invalidateMock, navigateMock } = vi.hoisted(() => ({
  loginMock: vi.fn(),
  registerMock: vi.fn(),
  invalidateMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("@/lib/trpc", () => ({
  trpc: {
    useUtils: () => ({ auth: { me: { invalidate: invalidateMock } } }),
    auth: {
      login: { useMutation: () => ({ mutateAsync: loginMock, isPending: false }) },
      register: { useMutation: () => ({ mutateAsync: registerMock, isPending: false }) },
    },
  },
}));

import Login from "./Login";

function fill(label: RegExp, value: string) {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

describe("Login page", () => {
  beforeEach(() => {
    loginMock.mockReset().mockResolvedValue({ id: 2 });
    registerMock.mockReset().mockResolvedValue({ id: 3 });
    invalidateMock.mockReset().mockResolvedValue(undefined);
    navigateMock.mockReset();
    window.history.replaceState(null, "", "/login?next=%2Fsessions%2Fabc");
  });
  afterEach(cleanup);

  it("signs in and returns to where the visitor came from", async () => {
    render(<Login navigate={navigateMock} />);
    fill(/email/i, "a@example.com");
    fill(/password/i, "long enough pw");
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(navigateMock).toHaveBeenCalledWith("/sessions/abc"));
    expect(loginMock).toHaveBeenCalledWith({ email: "a@example.com", password: "long enough pw" });
    expect(invalidateMock).toHaveBeenCalled();
  });

  it("creates an account from the second tab", async () => {
    render(<Login navigate={navigateMock} />);
    fireEvent.click(screen.getByRole("button", { name: "Create an account" }));
    fill(/name/i, "Andrew");
    fill(/email/i, "a@example.com");
    fill(/password/i, "long enough pw");
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() => expect(registerMock).toHaveBeenCalledWith({ name: "Andrew", email: "a@example.com", password: "long enough pw" }));
    expect(navigateMock).toHaveBeenCalledWith("/sessions/abc");
  });

  it("shows the server's message when sign-in fails", async () => {
    loginMock.mockRejectedValue(new Error("Email or password is incorrect."));
    render(<Login navigate={navigateMock} />);
    fill(/email/i, "a@example.com");
    fill(/password/i, "wrong password");
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "Email or password is incorrect.");
    expect(navigateMock).not.toHaveBeenCalled();
  });
});
