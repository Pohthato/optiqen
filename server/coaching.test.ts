// server/coaching.test.ts
import { describe, expect, it } from "vitest";
import { AnalysisResult } from "./analysis";
import { analysisResultSchema } from "./analysisSchema";
import { Finding, buildFindingsPrompt, checkCoaching, coachingText, findingsOf, sentencesOf } from "./coaching";
import { coachingPrompt } from "./deepseek";

const FINDINGS: Finding[] = [
  {
    id: "F1", kind: "end", about: "near", shots: [3], evidence: [{ frame: 91, timeMs: 3033.3 }, { frame: 117, timeMs: 3900 }],
    call: "out", p: 0.99, facts: { p: { in: 0.01, out: 0.99, net: 0 }, winner: "far", pHitterWon: 0.01 },
    text: "Shot 3, the near player's, came down out (99%), so the far player won the point.",
  },
  {
    id: "F2", kind: "how", about: null, shots: [2, 3], evidence: [{ frame: 61, timeMs: 2033.3 }, { frame: 91, timeMs: 3033.3 }],
    call: "too close to call", p: null, facts: { p: { winner: 0, "let in": 0, "forced error": 0.44, "unforced error": 0.56 }, spareMs: { value: 212.4, sigma: 80 } },
    text: "How the rally was won is too close to call: forced error 44%, unforced error 56%.",
  },
  {
    id: "F3", kind: "shot", about: "near", shots: [3], evidence: [{ frame: 91, timeMs: 3033.3 }],
    call: "push", p: 0.9, facts: { speedMS: { value: 30.04, sigma: 2.1 }, netClearanceM: { value: 0.351, sigma: 0.08 }, body: { elbowDeg: { value: 121.6, sigma: 9.5 } } },
    text: "Shot 3, the near player's: push, 30.0 ± 2.1 m/s off the racket, 0.35 ± 0.08 m over the net.",
  },
  {
    id: "F4", kind: "player", about: "near", shots: [1, 3], evidence: [{ frame: 31, timeMs: 1033.3 }],
    call: "right", p: 0.88, facts: { armReachM: { value: 1.389, sigma: 0.008 } },
    text: "The near player reaches 1.39 ± 0.01 m from the hips and plays right-handed (88%).",
  },
];

const reasons = (reply: string) => checkCoaching(reply, FINDINGS).dropped.map((sentence) => sentence.why);

describe("coaching from findings", () => {
  it("shows a reply whose every sentence rests on what was found", () => {
    const reply = [
      "Your push on shot 3 came down out, and that cost you the point [F1].",
      "It left the racket at 30 m/s and crossed 0.35 m over the net, so the pace was there but not the control [F1, F3].",
      "Whether your error was forced is too close to call: the analysis puts it at 56% unforced [F2].",
      "Practise soft pushes that die in the forecourt, at three-quarter pace [F3].",
    ].join(" ");
    const checked = checkCoaching(reply, FINDINGS);
    expect(checked.dropped).toEqual([]);
    expect(checked.shown.map((sentence) => sentence.cites)).toEqual([["F1"], ["F1", "F3"], ["F2"], ["F3"]]);
  });

  it("drops every sentence the findings do not support, and says why", () => {
    expect(reasons("You hit the shuttle far too hard.")).toEqual(["it names no finding"]);
    expect(reasons("Your smash reached 45 m/s [F3].")).toEqual(["45 is not in F3"]);
    expect(reasons("The push cleared the net by half a metre, 0.5 m [F3].")).toEqual(["0.5 is not in F3"]);
    expect(reasons("Your backhand clear went long [F9].")).toEqual(["it names F9, which the analysis did not find"]);
    expect(reasons("You lost the rally to an unforced error [F2].")).toEqual(["it states as fact what F2 left too close to call"]);
    expect(reasons("Rest the shoulder to avoid an injury before the next session [F3].")).toEqual(["it strays into injury or medicine"]);
    expect(reasons("Your elbow was too straight and will cause pain [F3].")).toEqual(["it strays into injury or medicine"]);
  });

  it("takes numbers as the findings hold them, rounded, as chances and in seconds", () => {
    const shown = (reply: string) => checkCoaching(reply, FINDINGS).shown.length;
    expect(shown("It crossed 0.4 m over the net [F3].")).toBe(1);
    expect(shown("It came down out, a 99% call [F1].")).toBe(1);
    expect(shown("It came down at 3.9 s [F1].")).toBe(1);
    expect(shown("You had 0.21 s to spare after shot 2, give or take, which leans unforced [F2].")).toBe(1);
    expect(shown("Your elbow was at 122 degrees at the hit [F3].")).toBe(1);
  });

  it("keeps a cite written after the full stop with its sentence", () => {
    expect(sentencesOf("Shot 3 came down out. [F1] It was a push. [F3]")).toEqual(["Shot 3 came down out. [F1]", "It was a push. [F3]"]);
    expect(sentencesOf("It crossed 0.35 m over the net [F3].\nNext line [F1].")).toEqual(["It crossed 0.35 m over the net [F3].", "Next line [F1]."]);
    expect(checkCoaching("Shot 3 came down out. [F1] Then nothing. And a claim without one.", FINDINGS).dropped.map((s) => s.text)).toEqual(["Then nothing.", "And a claim without one."]);
  });

  it("says what was left out, and falls back to the findings' own sentences when nothing passes", () => {
    const some = checkCoaching("Shot 3 came down out [F1]. You were tired.", FINDINGS);
    expect(coachingText(some, FINDINGS, "near")).toBe("Shot 3 came down out [F1].\n\n(1 sentence was left out: the analysis does not support it.)");
    const none = checkCoaching("You were tired. Rest more.", FINDINGS);
    expect(coachingText(none, FINDINGS, "near")).toBe(`${FINDINGS[0].text} [F1] ${FINDINGS[1].text} [F2] ${FINDINGS[3].text} [F4]\n\n(2 sentences were left out: the analysis does not support them.)`);
  });

  it("gives the coach the findings and the rules, and nothing else to claim from", () => {
    const prompt = buildFindingsPrompt("near", FINDINGS, "How do I stop losing points at the net?");
    for (const finding of FINDINGS) expect(prompt).toContain(`"id":"${finding.id}"`);
    expect(prompt).toContain("End it with the IDs of the findings it rests on");
    expect(prompt).toContain("Their question: How do I stop losing points at the net?");
    expect(prompt).not.toMatch(/metric|calibration/i);
  });

  it("finds the findings in a result, and none in one from before them", () => {
    expect(findingsOf({ rally: { findings: FINDINGS, shots: [] } })).toEqual(FINDINGS);
    expect(findingsOf({})).toBeNull();
    expect(findingsOf({ rally: { findings: [{ id: "X" }] } })).toBeNull();
    expect(findingsOf({ rally: { findings: [] } })).toBeNull();
  });

  it("lets a worker result carry the rally analysis and its findings", () => {
    const base = {
      processingVersion: "optiqen-badminton-worker-test",
      calibration: { corners: [{ label: "nearLeft", x: 10, y: 80 }, { label: "nearRight", x: 90, y: 80 }, { label: "farRight", x: 70, y: 20 }], confidence: "provisional", supportsCourtMapping: false, guidance: "test" },
      quality: { usableFrameRatio: 0.9, poseTrackConfidence: 0.9, shuttleTrackConfidence: 0.5 },
      metrics: [],
      shotDistribution: {},
    };
    const parsed = analysisResultSchema.safeParse({ ...base, rally: { findings: FINDINGS, shots: [{ shot: 1 }] } });
    expect(parsed.success).toBe(true);
    expect(findingsOf(parsed.data)).toEqual(FINDINGS);
    expect(analysisResultSchema.safeParse({ ...base, rally: { findings: [{ id: "F1" }] } }).success).toBe(false);
    // A job whose rally analysis could not be made says why, and is coached as before.
    const without = analysisResultSchema.safeParse({ ...base, rally: null, rallyError: "The rally analysis could not be made: no lines." });
    expect(without.success).toBe(true);
    expect(findingsOf(without.data)).toBeNull();
    // The prompt previewed is the one sent: from the findings when there are some, as before when not.
    const context = (result: unknown) => ({ player: "near" as const, calibration: (result as AnalysisResult).calibration, result: result as AnalysisResult });
    expect(coachingPrompt(context(parsed.data))).toContain("End it with the IDs of the findings it rests on");
    expect(coachingPrompt(context(base))).not.toContain("End it with the IDs of the findings it rests on");
  });
});
