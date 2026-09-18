// @vitest-environment jsdom
import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { authState, startLoginMock, toastInfoMock, uploadMock, createDraftMock, submitMock } = vi.hoisted(() => ({
  authState: { isAuthenticated: true, logout: vi.fn() },
  startLoginMock: vi.fn(),
  toastInfoMock: vi.fn(),
  uploadMock: vi.fn(),
  createDraftMock: vi.fn(),
  submitMock: vi.fn(),
}));

vi.mock("@/_core/hooks/useAuth", () => ({ useAuth: () => authState }));
vi.mock("@/const", () => ({ startLogin: startLoginMock }));
vi.mock("sonner", () => ({ toast: { info: toastInfoMock, error: vi.fn(), success: vi.fn() } }));
vi.mock("@/components/AIChatBox", () => ({ AIChatBox: () => React.createElement("div", null, "DeepSeek coach") }));
vi.mock("@/lib/trpc", () => {
  const query = () => ({ data: undefined, refetch: vi.fn() });
  return {
    trpc: {
      upload: { prepareVideo: { useMutation: () => ({ mutateAsync: uploadMock, isPending: false }) } },
      analysis: {
        createDraft: { useMutation: () => ({ mutateAsync: createDraftMock, isPending: false }) },
        submit: { useMutation: () => ({ mutateAsync: submitMock, isPending: false }) },
        warmupGpu: { useMutation: () => ({ mutateAsync: vi.fn().mockResolvedValue({ accepted: true }), isPending: false }) },
        refresh: { useMutation: () => ({ mutateAsync: vi.fn(), isPending: false }) },
        get: { useQuery: query },
      },
      coach: { chat: { useMutation: () => ({ mutateAsync: vi.fn(), isPending: false }) } },
    },
  };
});

import Home from "./Home";

function mockCourtStage() {
  const stage = document.querySelector(".court-grid") as HTMLDivElement;
  Object.defineProperty(stage, "getBoundingClientRect", {
    configurable: true,
    value: () => ({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100 }),
  });
  return stage;
}

async function selectVideoAndClickThreePoints(order: Array<[number, number]>) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  const file = new File(["video-bytes"], "cropped-rally.mp4", { type: "video/mp4" });
  fireEvent.change(input, { target: { files: [file] } });
  await screen.findByText("cropped-rally.mp4");
  const stage = mockCourtStage();
  for (const [clientX, clientY] of order) fireEvent.click(stage, { clientX, clientY });
  return stage;
}

describe("Home authentication and calibration interactions", () => {
  beforeEach(() => {
    authState.isAuthenticated = true;
    authState.logout.mockReset();
    startLoginMock.mockReset();
    toastInfoMock.mockReset();
    uploadMock.mockReset();
    createDraftMock.mockReset();
    submitMock.mockReset();
    uploadMock.mockResolvedValue({ uploadUrl: "https://upload.example.test/video", contentType: "video/mp4", key: "source/video.mp4" });
    createDraftMock.mockResolvedValue({ id: "draft-1" });
    submitMock.mockResolvedValue({ status: "queued" });
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:source"), revokeObjectURL: vi.fn() });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it("opens the profile menu from the authenticated avatar", () => {
    render(<Home />);
    fireEvent.click(screen.getByRole("button", { name: "Open profile menu" }));
    expect(screen.getByText("Sign out")).toBeTruthy();
  });

  it("starts hosted sign-in from the unauthenticated avatar when OAuth is configured", () => {
    authState.isAuthenticated = false;
    vi.stubEnv("VITE_OAUTH_PORTAL_URL", "https://auth.example.test");
    vi.stubEnv("VITE_APP_ID", "netoval-local");
    render(<Home />);
    fireEvent.click(screen.getByRole("button", { name: "Sign in to analyze" }));
    expect(startLoginMock).toHaveBeenCalledTimes(1);
  });

  it("reports the local sign-in setup requirement when OAuth is unavailable", () => {
    authState.isAuthenticated = false;
    vi.stubEnv("VITE_OAUTH_PORTAL_URL", "");
    vi.stubEnv("VITE_APP_ID", "");
    render(<Home />);
    fireEvent.click(screen.getByRole("button", { name: "Configure local sign-in" }));
    expect(toastInfoMock).toHaveBeenCalledWith(expect.stringContaining("Local sign-in is not configured"));
  });

  it("keeps point labels neutral when far and near points are clicked in arbitrary order", async () => {
    render(<Home />);
    await selectVideoAndClickThreePoints([[82, 78], [18, 24], [68, 30]]);
    expect(screen.getAllByText("Observed point 1").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Observed point 2").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Observed point 3").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Estimated missing corner")).toBeTruthy();
    expect(screen.queryByText("Near left")).toBeNull();
    expect(screen.queryByText("Near right")).toBeNull();
  });

  it("submits the same resolved corner payload regardless of click order", async () => {
    render(<Home />);
    await selectVideoAndClickThreePoints([[82, 78], [18, 24], [68, 30]]);
    fireEvent.click(screen.getByRole("button", { name: /Secure upload & analyse/i }));
    await waitFor(() => expect(createDraftMock).toHaveBeenCalledTimes(1));
    const firstPayload = createDraftMock.mock.calls[0][0].corners;

    cleanup();
    createDraftMock.mockReset();
    render(<Home />);
    await selectVideoAndClickThreePoints([[68, 30], [82, 78], [18, 24]]);
    fireEvent.click(screen.getByRole("button", { name: /Secure upload & analyse/i }));
    await waitFor(() => expect(createDraftMock).toHaveBeenCalledTimes(1));
    const secondPayload = createDraftMock.mock.calls[0][0].corners;

    expect(secondPayload).toEqual(firstPayload);
  });
});
