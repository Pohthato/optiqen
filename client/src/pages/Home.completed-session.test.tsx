import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const completedSession = {
  id: "session-123",
  status: "completed",
  result: {
    calibration: { confidence: "image_space_only", supportsCourtMapping: false },
    overlays: [{ timeMs: 0, errors: { Racket: "occluded by body" }, skeleton: [{ x: 50, y: 45, confidence: 0.9 }] }],
    rally: {
      shots: [{ shot: 1, hitter: "near", hitMs: 500, path: [[500, 2, 4, 1], [900, 3, 11, 1.4]], landing: null, players: null, name: { said: "serve" } }],
      findings: [{ id: "F1", kind: "shot", shots: [1], evidence: [{ frame: 15, timeMs: 500 }], text: "Shot 1, the near player's: serve." }],
    },
  },
};

vi.mock("@/_core/hooks/useAuth", () => ({ useAuth: () => ({ isAuthenticated: true }) }));
vi.mock("@/components/AIChatBox", () => ({ AIChatBox: () => React.createElement("div", null, "DeepSeek coach") }));
vi.mock("@/lib/trpc", () => {
  const mutation = () => ({ mutateAsync: vi.fn(), isPending: false });
  return {
    trpc: {
      upload: { prepareVideo: { useMutation: mutation } },
      analysis: {
        createDraft: { useMutation: mutation },
        submit: { useMutation: mutation },
        warmupGpu: { useMutation: mutation },
        refresh: { useMutation: mutation },
        get: { useQuery: () => ({ data: completedSession, refetch: vi.fn() }) },
      },
      coach: { chat: { useMutation: mutation } },
    },
  };
});

import Home from "./Home";

describe("Home completed-session integration", () => {
  beforeEach(() => {
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:source"), revokeObjectURL: vi.fn() });
  });

  it("renders the completed annotated-review path from session result overlays", () => {
    const markup = renderToStaticMarkup(<Home />);
    expect(markup).toContain("Image-space only");
    expect(markup).toContain("POSE VERIFIED");
    expect(markup).toContain("RACKET ERROR");
    expect(markup).toContain("COURT NO DATA");
    expect(markup).toContain("DeepSeek coach");
  });

  it("draws the rally on the court with its findings when the result has the rally analysis", () => {
    const markup = renderToStaticMarkup(<Home />);
    expect(markup).toContain('aria-label="1 shot drawn on the court"');
    expect(markup).toContain("Shot 1, the near player&#x27;s: serve.");
  });
});
