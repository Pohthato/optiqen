import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalysisResult, validateCalibration } from "./analysis";
import { requestDeepSeekCoaching } from "./deepseek";

const calibration = validateCalibration([
  { label: "nearLeft", x: 12, y: 86 },
  { label: "nearRight", x: 88, y: 86 },
  { label: "farRight", x: 73, y: 18 },
  { label: "farLeft", x: 27, y: 18 },
]);

function session(shuttleTrackConfidence: number): AnalysisResult {
  return {
    processingVersion: "cv-1.0.0",
    calibration,
    quality: { usableFrameRatio: 0.88, poseTrackConfidence: 0.92, shuttleTrackConfidence },
    metrics: [{ metric: "split step timing", value: 105, unit: "ms", direction: "contextual", confidence: 0.9, evidenceFrames: [{ frame: 72, timeMs: 2400, confidence: 0.94, source: "pose" }] }],
    shots: [{ frame: 1, timeMs: 200, confidence: 0.8, label: "clear", verified: true, direction: "cross", depth: "back", rallyIndex: 0 }],
    shotDistribution: { clear: 1 },
  };
}

type SentMessage = { role: string; content: string };

async function send(result: AnalysisResult, question: string) {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
    choices: [{ message: { content: "Evidence quality: good." } }],
    usage: { prompt_cache_hit_tokens: 1200, prompt_cache_miss_tokens: 40, completion_tokens: 900 },
  }), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const feedback = await requestDeepSeekCoaching({ player: "near", calibration: result.calibration, result, question });
  const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as { messages: SentMessage[]; temperature?: number };
  return { feedback, body };
}

describe("DeepSeek coaching request", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("orders the request so repeat questions reuse DeepSeek's prefix cache", async () => {
    vi.stubEnv("DEEPSEEK_API_KEY", "deepseek-test-key");
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);

    const first = await send(session(0.72), "Where should I focus?");
    const followUp = await send(session(0.72), "How do I recover faster?");
    const otherSession = await send(session(0.41), "Where should I focus?");

    expect(first.feedback).toBe("Evidence quality: good.");
    const [rules, evidence] = first.body.messages;
    expect(rules.role).toBe("system");
    expect(rules.content).toContain("Do not diagnose injuries");
    expect(rules.content).not.toContain("split step timing");
    expect(otherSession.body.messages[0]).toEqual(rules);

    const question = "Athlete question: Where should I focus?";
    expect(evidence.role).toBe("user");
    expect(evidence.content.endsWith(question)).toBe(true);
    const sessionEvidence = evidence.content.slice(0, -question.length);
    expect(followUp.body.messages[1].content).toBe(`${sessionEvidence}Athlete question: How do I recover faster?`);

    expect(first.body.temperature).toBeUndefined();
    expect(log).toHaveBeenCalledWith(expect.stringContaining("1200 of 1240 prompt tokens cached"));
  });
});
