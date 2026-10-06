// tools/labeller/labeller-core.test.ts
// @ts-nocheck
import { execFileSync } from "node:child_process";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

const workerDir = path.resolve(import.meta.dirname, "..", "..", "worker");
const python = process.env.PYTHON ?? "python";

function runPython(code: string, input?: string): string {
  return execFileSync(python, ["-c", code], { cwd: workerDir, input, encoding: "utf8" });
}

let core: any;

beforeAll(async () => {
  await import("./labeller-core.js");
  core = (globalThis as any).LabellerCore;
});

function freshState() {
  return core.emptyState("clip-001", 60, [1920, 1080]);
}

describe("labeller core: sources of truth", () => {
  it("uses the Python court model's keypoint names in the same order", () => {
    const names = JSON.parse(runPython("import json; from geometry.court_model import KEYPOINTS; print(json.dumps(list(KEYPOINTS)))"));
    expect(core.KEYPOINT_NAMES).toEqual(names);
  });

  it("uses the Python golden shot labels in the same order", () => {
    const labels = JSON.parse(runPython("import json; from evaluation.golden import GOLDEN_SHOT_LABELS; print(json.dumps(list(GOLDEN_SHOT_LABELS)))"));
    expect(core.SHOT_LABELS).toEqual(labels);
  });
});

describe("labeller core: contacts", () => {
  it("keeps contacts sorted, defaults the label to other, and merges taps within 20 ms", () => {
    const state = freshState();
    expect(core.addContact(state, 2000)).toBe(0);
    expect(core.addContact(state, 1000)).toBe(0);
    expect(state.contacts.map((c: any) => c.timeMs)).toEqual([1000, 2000]);
    expect(core.addContact(state, 1010)).toBe(0);
    expect(state.contacts).toHaveLength(2);
    expect(state.contacts[0].label).toBe("other");
  });

  it("rejects unknown labels and missing contacts", () => {
    const state = freshState();
    core.addContact(state, 1000);
    expect(() => core.setLabel(state, 0, "bogus")).toThrow("Unknown shot label");
    expect(() => core.setLabel(state, 5, "clear")).toThrow("No contact");
    core.setLabel(state, 0, "smash");
    expect(state.contacts[0].label).toBe("smash");
  });

  it("removes contacts and finds the nearest one within a tolerance", () => {
    const state = freshState();
    core.addContact(state, 1000);
    core.addContact(state, 2000);
    expect(core.nearestContactIndex(state, 1900, 150)).toBe(1);
    expect(core.nearestContactIndex(state, 1500, 100)).toBe(-1);
    core.removeContact(state, 0);
    expect(state.contacts.map((c: any) => c.timeMs)).toEqual([2000]);
  });
});

describe("labeller core: rallies", () => {
  it("ends the open rally after its start and applies the winner to the latest rally", () => {
    const state = freshState();
    core.startRally(state, 1000);
    expect(() => core.endRally(state, 1000)).toThrow("after it starts");
    core.endRally(state, 4000);
    core.setWinner(state, "near");
    core.startRally(state, 6000);
    core.endRally(state, 9000);
    core.setWinner(state, "far");
    expect(state.rallies).toEqual([
      { startMs: 1000, endMs: 4000, winner: "near" },
      { startMs: 6000, endMs: 9000, winner: "far" },
    ]);
  });

  it("restarting an open rally moves its start instead of adding another", () => {
    const state = freshState();
    core.startRally(state, 1000);
    core.startRally(state, 1500);
    expect(state.rallies).toHaveLength(1);
    expect(state.rallies[0].startMs).toBe(1500);
  });

  it("rejects an end with no open rally, a bad winner, and a winner with no rally", () => {
    const state = freshState();
    expect(() => core.endRally(state, 100)).toThrow("No rally is open");
    expect(() => core.setWinner(state, "near")).toThrow("No rally");
    core.startRally(state, 100);
    expect(() => core.setWinner(state, "left")).toThrow("Unknown winner");
  });

  it("leaves an open rally out of the export and says so", () => {
    const state = freshState();
    core.startRally(state, 1000);
    const { doc, warnings } = core.buildGolden(state);
    expect(doc.rallies).toEqual([]);
    expect(warnings.some((w: string) => w.includes("no end"))).toBe(true);
  });
});

describe("labeller core: court keypoints", () => {
  it("stores rounded pixel positions and replaces a re-clicked point", () => {
    const state = freshState();
    core.setKeypoint(state, "back1_sl", 612.123, 301.567);
    core.setKeypoint(state, "back1_sl", 700, 300);
    expect(state.keypoints).toEqual({ back1_sl: { x: 700, y: 300 } });
    core.removeKeypoint(state, "back1_sl");
    expect(state.keypoints).toEqual({});
  });

  it("records the frame a court point was clicked on and keeps it through export and reload", () => {
    const state = freshState();
    core.setKeypoint(state, "back1_sl", 612, 301.5, 4033.4);
    expect(state.keypoints.back1_sl).toEqual({ x: 612, y: 301.5, timeMs: 4033 });
    const { doc } = core.buildGolden(state);
    expect(doc.courtKeypoints).toEqual([{ name: "back1_sl", x: 612, y: 301.5, timeMs: 4033 }]);
    expect(core.stateFromGolden(doc)).toEqual(state);
    expect(() => core.setKeypoint(state, "back0_sl", 1, 1, NaN)).toThrow("finite");
  });

  it("rejects names outside the court model", () => {
    expect(() => core.setKeypoint(freshState(), "nope", 1, 1)).toThrow("Unknown court keypoint");
  });

  it("rejects non-finite times and pixel positions instead of storing them", () => {
    const state = freshState();
    expect(() => core.addContact(state, NaN)).toThrow("finite");
    expect(() => core.startRally(state, Infinity)).toThrow("finite");
    expect(() => core.setKeypoint(state, "back1_sl", NaN, 5)).toThrow("finite");
    expect(() => core.setKeypoint(state, "back1_sl", 5, Infinity)).toThrow("finite");
    expect(state.contacts).toEqual([]);
    expect(state.rallies).toEqual([]);
    expect(state.keypoints).toEqual({});
  });
});

describe("labeller core: video time", () => {
  it("maps video time to frame indexes whether the time is a frame start or centre", () => {
    expect(core.frameIndexAt(core.frameCentreSeconds(7, 60), 60)).toBe(7);
    expect(core.frameIndexAt(7 / 60, 60)).toBe(7);
    expect(core.frameIndexAt(7 / 60 - 0.001, 60)).toBe(7);
    expect(core.frameIndexAt(-1, 60)).toBe(0);
  });

  it("converts a frame index to milliseconds", () => {
    expect(core.frameTimeMs(60, 60)).toBe(1000);
    expect(core.frameTimeMs(1, 29.97)).toBe(33);
  });

  it("steps whole frames and clamps to the clip", () => {
    expect(core.stepSeconds(core.frameCentreSeconds(10, 60), 1, 60, 10)).toBeCloseTo(core.frameCentreSeconds(11, 60), 9);
    expect(core.stepSeconds(0.01, -5, 60, 10)).toBeCloseTo(core.frameCentreSeconds(0, 60), 9);
    expect(core.stepSeconds(9.99, 100, 60, 10)).toBeCloseTo(core.frameCentreSeconds(599, 60), 9);
  });

  it("detects fps from media timestamps even with a dropped frame", () => {
    const times = [0, 1, 2, 3, 5, 6, 7, 8].map(n => n / 60);
    expect(core.fpsFromMediaTimes(times)).toBe(60);
    expect(core.fpsFromMediaTimes([0, 1 / 30])).toBeNull();
    expect(core.fpsFromMediaTimes([])).toBeNull();
  });
});

describe("labeller core: worker import and export", () => {
  it("imports contacts, uses verified shot labels only, and never duplicates", () => {
    const state = freshState();
    core.addContact(state, 1005, "serve");
    const result = {
      events: [
        { type: "contact", timeMs: 1000 },
        { type: "contact", timeMs: 2000 },
        { type: "contact", timeMs: 3000 },
        { type: "split_step", timeMs: 1500 },
      ],
      shots: [
        { timeMs: 2010, label: "clear", verified: true },
        { timeMs: 3010, label: "smash", verified: false },
      ],
    };
    expect(core.importWorkerResult(state, result)).toBe(2);
    expect(state.contacts).toEqual([
      { timeMs: 1005, label: "serve" },
      { timeMs: 2000, label: "clear" },
      { timeMs: 3000, label: "other" },
    ]);
  });

  it("exports a golden document the Python validator accepts", () => {
    const state = freshState();
    core.setKeypoint(state, "back1_sl", 612, 301.5, 1200);
    core.setKeypoint(state, "post_left_top", 400, 200, 1200);
    core.addContact(state, 1200, "serve");
    core.addContact(state, 2100, "clear");
    core.startRally(state, 800);
    core.endRally(state, 4300);
    core.setWinner(state, "near");
    const { doc, warnings } = core.buildGolden(state);
    expect(warnings).toEqual([]);
    expect(doc.schemaVersion).toBe(1);
    expect(doc.shots).toEqual([
      { timeMs: 1200, label: "serve", landing: null },
      { timeMs: 2100, label: "clear", landing: null },
    ]);
    const errors = JSON.parse(
      runPython("import json,sys; from evaluation.golden import validate_golden; print(json.dumps(validate_golden(json.load(sys.stdin))))", JSON.stringify(doc)),
    );
    expect(errors).toEqual([]);
  });

  it("round-trips through the golden format", () => {
    const state = freshState();
    core.setKeypoint(state, "back0_sr", 1310, 298);
    core.addContact(state, 1200, "drop");
    core.addContact(state, 2100, "net");
    core.startRally(state, 800);
    core.endRally(state, 4300);
    core.setWinner(state, "far");
    const { doc } = core.buildGolden(state);
    expect(core.stateFromGolden(doc)).toEqual(state);
  });

  it("warns about contacts still labelled other", () => {
    const state = freshState();
    core.addContact(state, 1000);
    const { warnings } = core.buildGolden(state);
    expect(warnings.some((w: string) => w.includes('"other"'))).toBe(true);
  });
});

// A perspective map from court metres to pixels: far lines shrink and rise like a phone behind the baseline.
const H_BEHIND = [[120, 106.4, 400], [0, 0, 900], [0, 0.15, 1]];

function project(h: number[][], [x, y]: number[]) {
  const w = h[2][0] * x + h[2][1] * y + h[2][2];
  return [(h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w];
}

function clickCourt(state: any, h: number[][], names: string[], noise: number[][] = []) {
  names.forEach((name, i) => {
    const [x, y] = project(h, core.KEYPOINT_POSITIONS[name]);
    const [dx, dy] = noise[i] ?? [0, 0];
    core.setKeypoint(state, name, x + dx, y + dy, 500);
  });
}

describe("labeller core: court completion", () => {
  it("uses the Python court model's coordinates", () => {
    const positions = JSON.parse(runPython("import json; from geometry.court_model import KEYPOINTS; print(json.dumps({k: list(map(float, v)) for k, v in KEYPOINTS.items()}))"));
    for (const name of core.KEYPOINT_NAMES) {
      core.KEYPOINT_POSITIONS[name].forEach((value: number, axis: number) => expect(value).toBeCloseTo(positions[name][axis], 9));
    }
  });

  it("proposes every other visible floor point from four clicked corners", () => {
    const state = freshState();
    clickCourt(state, H_BEHIND, ["back0_sl", "back0_sr", "back1_sr", "back1_sl"]);
    const proposals = core.proposeCourtPoints(state);
    expect(Object.keys(proposals)).toContain("short0_c");
    expect(Object.keys(proposals)).not.toContain("back0_sl");
    for (const [name, point] of Object.entries<any>(proposals)) {
      const [x, y] = project(H_BEHIND, core.KEYPOINT_POSITIONS[name]);
      expect(point.x).toBeCloseTo(x, 1);
      expect(point.y).toBeCloseTo(y, 1);
    }
  });

  it("fits more than four slightly misplaced clicks by least squares", () => {
    const state = freshState();
    const noise = [[0.6, -0.4], [-0.5, 0.5], [0.4, 0.6], [-0.6, -0.3], [0.3, -0.6], [-0.4, 0.4]];
    clickCourt(state, H_BEHIND, ["back0_sl", "back0_sr", "back1_sr", "back1_sl", "short0_c", "short1_dl"], noise);
    const proposals = core.proposeCourtPoints(state);
    const [x, y] = project(H_BEHIND, core.KEYPOINT_POSITIONS.long0_c);
    expect(Math.hypot(proposals.long0_c.x - x, proposals.long0_c.y - y)).toBeLessThan(3);
  });

  it("proposes nothing from fewer than four points or from points on one line", () => {
    const state = freshState();
    clickCourt(state, H_BEHIND, ["back0_sl", "back0_sr", "back1_sr"]);
    expect(core.proposeCourtPoints(state)).toEqual({});
    const line = freshState();
    clickCourt(line, H_BEHIND, ["back0_dl", "back0_sl", "back0_c", "back0_sr"]);
    expect(core.proposeCourtPoints(line)).toEqual({});
  });

  it("never proposes points that would be behind the camera", () => {
    // w = 1 - 0.1 y turns negative past y = 10 m, where pixels can still land inside the frame.
    const behind = [[120, 0, -400], [0, 50, -1000], [0, -0.1, 1]];
    const state = freshState();
    clickCourt(state, behind, ["back0_sl", "back0_sr", "short0_sl", "short0_sr"]);
    const [x, y] = project(behind, core.KEYPOINT_POSITIONS.back1_sl);
    expect(x > 0 && x < 1920 && y > 0 && y < 1080).toBe(true);
    expect(core.proposeCourtPoints(state)).not.toHaveProperty("back1_sl");
  });

  it("accepts proposals as points on the frame the corners were clicked on", () => {
    const state = freshState();
    clickCourt(state, H_BEHIND, ["back0_sl", "back0_sr", "back1_sr", "back1_sl"]);
    const proposals = core.proposeCourtPoints(state);
    const added = core.acceptCourtProposals(state, proposals);
    expect(added).toBe(Object.keys(proposals).length);
    expect(state.keypoints.short0_c).toEqual({ ...proposals.short0_c, timeMs: 500, source: "proposed" });
  });

  it("lists court line segments between named points for drawing", () => {
    expect(core.COURT_LINES.length).toBe(13);
    for (const [from, to] of core.COURT_LINES) {
      expect(core.KEYPOINT_NAMES).toContain(from);
      expect(core.KEYPOINT_NAMES).toContain(to);
    }
  });
});

const AUDIO_SCRIPT = [
  "import base64, json, sys",
  "from simulation.audio import render_audio, SAMPLE_RATE",
  "samples, extra = render_audio(json.loads(sys.argv[1]), float(sys.argv[2]), distractors=int(sys.argv[3]), seed=11)",
  "print(json.dumps({'rate': SAMPLE_RATE, 'distractors': extra, 'samples': base64.b64encode(samples.astype('<f4').tobytes()).decode()}))",
].join("\n");

function syntheticAudio(contactTimes: number[], distractors: number, seconds: number) {
  const out = JSON.parse(
    execFileSync(python, ["-c", AUDIO_SCRIPT, JSON.stringify(contactTimes), String(seconds), String(distractors)], {
      cwd: workerDir,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    }),
  );
  const bytes = Buffer.from(out.samples, "base64");
  const samples = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return { rate: out.rate as number, distractors: out.distractors as number[], samples };
}

describe("labeller core: hits from audio", () => {
  it("finds every racket hit within 5 ms and ranks neighbouring-court hits below them", () => {
    const hits = [0.5, 1.4, 2.2, 3.05];
    const audio = syntheticAudio(hits, 2, 4.0);
    const onsets = core.detectOnsets(audio.samples, audio.rate);
    const matched = hits.map(t => onsets.find((o: any) => Math.abs(o.timeMs - t * 1000) <= 5));
    expect(matched.every(Boolean)).toBe(true);
    const weakestHit = Math.min(...matched.map((o: any) => o.strength));
    for (const t of audio.distractors) {
      const onset = onsets.find((o: any) => Math.abs(o.timeMs - t * 1000) <= 5);
      if (onset) expect(onset.strength).toBeLessThan(weakestHit);
    }
  });

  it("finds nothing in background noise alone", () => {
    const audio = syntheticAudio([], 0, 2.0);
    expect(core.detectOnsets(audio.samples, audio.rate)).toEqual([]);
  });

  it("reports one onset for sounds closer together than the minimum gap", () => {
    const audio = syntheticAudio([1.0, 1.06], 0, 2.0);
    expect(core.detectOnsets(audio.samples, audio.rate)).toHaveLength(1);
  });

  it("turns onsets into suggestions, skipping hits already labelled", () => {
    const state = freshState();
    core.addContact(state, 1000, "clear");
    const kept = core.setHitSuggestions(state, [
      { timeMs: 1030, strength: 3 },
      { timeMs: 2000, strength: 3 },
      { timeMs: 3000, strength: 2 },
    ]);
    expect(kept).toBe(2);
    expect(state.hitSuggestions.map((s: any) => s.timeMs)).toEqual([2000, 3000]);
    expect(core.suggestionSeekMs(state.hitSuggestions[0])).toBe(1970);
  });

  it("walks pending suggestions, accepting and skipping them", () => {
    const state = freshState();
    core.setHitSuggestions(state, [{ timeMs: 2000, strength: 3 }, { timeMs: 3000, strength: 3 }, { timeMs: 4000, strength: 3 }]);
    expect(core.nextPendingSuggestion(state, -1)).toBe(0);
    const contact = core.acceptSuggestion(state, 0, 1967, "smash");
    expect(state.contacts[contact]).toEqual({ timeMs: 1967, label: "smash" });
    core.dismissSuggestion(state, 1);
    expect(core.nextPendingSuggestion(state, 0)).toBe(2);
    expect(core.nextPendingSuggestion(state, 2)).toBe(-1);
    expect(state.hitSuggestions.map((s: any) => s.status)).toEqual(["accepted", "dismissed", "pending"]);
  });
});

describe("labeller core: rally suggestions", () => {
  it("groups hits separated by long pauses into rallies", () => {
    const contacts = [1000, 2000, 3100, 9000, 10000].map(timeMs => ({ timeMs, label: "other" }));
    expect(core.suggestRallies(contacts)).toEqual([
      { startMs: 400, endMs: 4600 },
      { startMs: 8400, endMs: 11500 },
    ]);
  });

  it("adds suggested rallies that do not overlap ones already marked, and sets winners by index", () => {
    const state = freshState();
    [1000, 2000, 9000, 10000].forEach(t => core.addContact(state, t, "clear"));
    core.startRally(state, 300);
    core.endRally(state, 3000);
    expect(core.applyRallySuggestions(state)).toBe(1);
    expect(state.rallies.map((r: any) => [r.startMs, r.endMs])).toEqual([[300, 3000], [8400, 11500]]);
    core.setRallyWinner(state, 1, "far");
    expect(state.rallies[1].winner).toBe("far");
    expect(() => core.setRallyWinner(state, 5, "far")).toThrow("No rally");
  });
});

const POSE = Array.from({ length: 17 }, (_, i) => [900 + i, 500 + i, 2]);

describe("labeller core: body poses", () => {
  it("uses the Python COCO joint order", () => {
    const joints = JSON.parse(runPython("import json; from evaluation.golden import COCO_JOINTS; print(json.dumps(list(COCO_JOINTS)))"));
    expect(core.COCO_JOINTS).toEqual(joints);
  });

  it("stores a validated pose per hit and lists hits still needing one", () => {
    const state = freshState();
    core.setSelectedPlayer(state, "near");
    core.addContact(state, 1000, "clear");
    core.addContact(state, 2000, "drop");
    core.setPose(state, 1000, "near", POSE);
    expect(core.poseTargets(state)).toEqual([2000]);
    expect(() => core.setPose(state, 2000, "near", POSE.slice(1))).toThrow("17");
    expect(() => core.setPose(state, 2000, "near", POSE.map(([x, y]) => [x, y, 5]))).toThrow("visibility");
    expect(() => core.setPose(state, 2000, "umpire", POSE)).toThrow("player");
    expect(() => core.setSelectedPlayer(state, "left")).toThrow("player");
    core.removePose(state, 1000);
    expect(core.poseTargets(state)).toEqual([1000, 2000]);
  });

  it("turns detector output into candidates and picks the selected player", () => {
    const far = { score: 0.8, keypoints: Array.from({ length: 17 }, (_, i) => ({ x: 950 + i, y: 300 + i, score: 0.9 })) };
    const near = { score: 0.7, keypoints: Array.from({ length: 17 }, (_, i) => ({ x: 600 + i, y: 700 + i * 5, score: i === 3 ? 0.1 : 0.9 })) };
    const candidates = core.candidatesFromDetections([far, near]);
    expect(candidates[1].keypoints[3][2]).toBe(0);
    expect(candidates[1].keypoints[4][2]).toBe(2);
    expect(core.pickPlayer(candidates, "near", null)).toBe(1);
    expect(core.pickPlayer(candidates, "far", null)).toBe(0);
    expect(core.pickPlayer(candidates, "near", candidates[0].keypoints)).toBe(0);
    expect(core.pickPlayer([], "near", null)).toBe(-1);
  });

  it("finds the latest confirmed pose before a time for the selected player", () => {
    const state = freshState();
    core.setSelectedPlayer(state, "near");
    core.setPose(state, 1000, "near", POSE);
    expect(core.previousPose(state, 1500)).toEqual(POSE);
    expect(core.previousPose(state, 900)).toBeNull();
  });

  it("exports poses the Python validator accepts and warns about hits without one", () => {
    const state = freshState();
    core.setSelectedPlayer(state, "near");
    core.addContact(state, 1000, "clear");
    core.addContact(state, 2000, "drop");
    core.setPose(state, 1000, "near", POSE);
    const { doc, warnings } = core.buildGolden(state);
    expect(doc.selectedPlayer).toBe("near");
    expect(doc.poses).toEqual([{ timeMs: 1000, player: "near", keypoints: POSE }]);
    expect(warnings.some((w: string) => w.includes("need a body pose"))).toBe(true);
    const errors = JSON.parse(
      runPython("import json,sys; from evaluation.golden import validate_golden; print(json.dumps(validate_golden(json.load(sys.stdin))))", JSON.stringify(doc)),
    );
    expect(errors).toEqual([]);
    expect(core.stateFromGolden(doc).poses).toEqual(state.poses);
    expect(core.stateFromGolden(doc).selectedPlayer).toBe("near");
  });

  it("upgrades autosaves from before suggestions and poses existed", () => {
    const old = { clipId: "c", fps: 60, imageSize: [1920, 1080], keypoints: {}, contacts: [], rallies: [] };
    const state = core.normalizeState(old);
    expect(state.poses).toEqual({});
    expect(state.hitSuggestions).toEqual([]);
    expect(state.selectedPlayer).toBeNull();
  });
});

// ---- Fixes from the Step 2 review ----

/** Click each name at the image position of another court point (to simulate labelling mistakes). */
function clickAs(state: any, h: number[][], labels: string[], actual: string[], timeMs = 500) {
  labels.forEach((name, i) => {
    const [x, y] = project(h, core.KEYPOINT_POSITIONS[actual[i]]);
    core.setKeypoint(state, name, x, y, timeMs, "clicked");
  });
}

const CORNERS = ["back0_sl", "back0_sr", "back1_sr", "back1_sl"];

describe("review fixes: corner mistakes never produce named proposals", () => {
  it("refuses left/right mirrored corners and says why", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, ["back0_sr", "back0_sl", "back1_sl", "back1_sr"], CORNERS);
    expect(core.proposeCourtPoints(state)).toEqual({});
    expect(core.courtProblem(state)).toMatch(/left and right/);
  });

  it("refuses near/far swapped corners and says why", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, ["back1_sl", "back1_sr", "back0_sr", "back0_sl"], CORNERS);
    expect(core.proposeCourtPoints(state)).toEqual({});
    expect(core.courtProblem(state)).toMatch(/nearest the camera/);
  });

  it("reports no problem for correctly ordered corners", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, CORNERS, CORNERS);
    expect(core.courtProblem(state)).toBeNull();
    expect(Object.keys(core.proposeCourtPoints(state)).length).toBeGreaterThan(0);
  });

  it("refuses four points when three of them lie on one line", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, ["back0_dl", "back0_sl", "back0_sr", "back1_c"], ["back0_dl", "back0_sl", "back0_sr", "back1_c"]);
    expect(core.proposeCourtPoints(state)).toEqual({});
  });
});

describe("review fixes: court points carry their frame and origin", () => {
  it("fits only the points of one frame and ignores proposals as evidence", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, CORNERS, CORNERS, 500);
    core.setKeypoint(state, "short0_c", 10, 10, 9000, "clicked"); // a stray click on another frame
    const proposals = core.proposeCourtPoints(state);
    const [x, y] = project(H_BEHIND, core.KEYPOINT_POSITIONS.long0_c);
    expect(proposals.long0_c.x).toBeCloseTo(x, 1);
    expect(proposals.long0_c.y).toBeCloseTo(y, 1);
  });

  it("marks accepted proposals as proposed and moved ones as adjusted", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, CORNERS, CORNERS);
    core.acceptCourtProposals(state, core.proposeCourtPoints(state));
    expect(state.keypoints.short0_c.source).toBe("proposed");
    core.moveKeypoint(state, "short0_c", 500, 600, 500);
    expect(state.keypoints.short0_c.source).toBe("adjusted");
    core.moveKeypoint(state, "back0_sl", 300, 990, 500);
    expect(state.keypoints.back0_sl.source).toBe("clicked");
  });

  it("re-derives untouched proposals when a clicked corner moves, but keeps adjusted ones", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, CORNERS, CORNERS);
    core.acceptCourtProposals(state, core.proposeCourtPoints(state));
    core.moveKeypoint(state, "long0_c", 700, 700, 500);
    const before = { ...state.keypoints.short0_c };
    core.moveKeypoint(state, "back0_sl", state.keypoints.back0_sl.x - 25, state.keypoints.back0_sl.y, 500);
    expect(core.rederiveProposed(state)).toBeGreaterThan(0);
    expect(state.keypoints.short0_c.x).not.toBeCloseTo(before.x, 1);
    expect(state.keypoints.long0_c).toMatchObject({ x: 700, y: 700, source: "adjusted" });
  });

  it("exports and restores where each point came from", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, CORNERS, CORNERS);
    core.acceptCourtProposals(state, core.proposeCourtPoints(state));
    const { doc } = core.buildGolden(state);
    expect(doc.courtKeypoints.find((p: any) => p.name === "short0_c").source).toBe("proposed");
    expect(core.stateFromGolden(doc).keypoints).toEqual(state.keypoints);
  });
});

describe("review fixes: hits", () => {
  it("only accepts a suggestion at a frame close to it", () => {
    const state = freshState();
    core.setHitSuggestions(state, [{ timeMs: 2000, strength: 3 }]);
    expect(() => core.acceptSuggestion(state, 0, 20000, "smash")).toThrow(/far from/);
    expect(state.contacts).toEqual([]);
    expect(state.hitSuggestions[0].status).toBe("pending");
  });

  it("deleting a hit deletes its pose", () => {
    const state = freshState();
    core.setSelectedPlayer(state, "near");
    core.addContact(state, 1000, "clear");
    core.setPose(state, 1000, "near", POSE);
    core.removeContact(state, 0);
    expect(state.poses).toEqual({});
  });

  it("warns about any pose that is not on a hit", () => {
    const state = freshState();
    core.setSelectedPlayer(state, "near");
    core.addContact(state, 1017, "clear");
    state.poses["1000"] = { player: "near", keypoints: POSE };
    expect(core.buildGolden(state).warnings.some((w: string) => w.includes("not on a hit"))).toBe(true);
  });
});

describe("review fixes: poses", () => {
  it("a pose can be confirmed only on the frame it was detected on", () => {
    expect(core.canConfirmPose({ timeMs: 1000, keypoints: POSE }, 1000)).toBe(true);
    expect(core.canConfirmPose({ timeMs: 1000, keypoints: POSE }, 1033)).toBe(false);
    expect(core.canConfirmPose({ timeMs: 1000, keypoints: null }, 1000)).toBe(false);
    expect(core.canConfirmPose(null, 1000)).toBe(false);
  });

  it("will not switch player while the other player's poses exist", () => {
    const state = freshState();
    core.setSelectedPlayer(state, "near");
    core.addContact(state, 1000, "clear");
    core.setPose(state, 1000, "near", POSE);
    expect(() => core.setSelectedPlayer(state, "far")).toThrow(/poses/);
    core.removePose(state, 1000);
    core.setSelectedPlayer(state, "far");
    expect(state.selectedPlayer).toBe("far");
  });

  it("leaves low-confidence joints unlabelled, clamps joints into the frame, and cycles visibility", () => {
    const detection = { score: 0.7, keypoints: Array.from({ length: 17 }, (_, i) => ({ x: i === 0 ? 1920 : 600 + i, y: 700, score: i === 3 ? 0.1 : 0.9 })) };
    const [candidate] = core.candidatesFromDetections([detection], 0.3, [1920, 1080]);
    expect(candidate.keypoints[3][2]).toBe(0);
    expect(candidate.keypoints[0][0]).toBeLessThan(1920);
    expect([2, 1, 0].map((v: number) => core.nextVisibility(v))).toEqual([1, 0, 2]);
  });

  it("prefers the person standing on the selected player's half of the court", () => {
    const state = freshState();
    clickAs(state, H_BEHIND, CORNERS, CORNERS);
    const fit = core.courtHomography(state);
    const standingAt = (courtX: number, courtY: number) => {
      const [ax, ay] = project(H_BEHIND, [courtX, courtY]);
      return { keypoints: Array.from({ length: 17 }, (_, i) => [ax, ay - (16 - i) * 4, 2]), score: 0.8, bottom: ay };
    };
    const nearPlayer = standingAt(2.0, 3.0);
    const farPlayer = standingAt(2.5, 10.0);
    // The previous pose is near the far player, but the selected player is "near".
    expect(core.pickPlayer([farPlayer, nearPlayer], "near", farPlayer.keypoints, fit)).toBe(1);
    expect(core.pickPlayer([farPlayer, nearPlayer], "far", null, fit)).toBe(0);
  });
});

describe("review fixes: rallies", () => {
  it("keeps suggested rallies inside the clip and out of the export until a winner is chosen", () => {
    const state = freshState();
    [1000, 2000, 9000, 9800].forEach(t => core.addContact(state, t, "clear"));
    core.applyRallySuggestions(state, 10500);
    expect(state.rallies.map((r: any) => [r.startMs, r.endMs, r.confirmed])).toEqual([[400, 3500, false], [8400, 10500, false]]);
    let exported = core.buildGolden(state);
    expect(exported.doc.rallies).toEqual([]);
    expect(exported.warnings.some((w: string) => w.includes("not checked"))).toBe(true);
    core.setRallyWinner(state, 0, "unknown");
    exported = core.buildGolden(state);
    expect(exported.doc.rallies).toEqual([{ startMs: 400, endMs: 3500, winner: "unknown" }]);
  });

  it("edits a rally's start and end", () => {
    const state = freshState();
    core.startRally(state, 1000);
    core.endRally(state, 5000);
    core.setRallyBounds(state, 0, 1200, 4800);
    expect(state.rallies[0]).toMatchObject({ startMs: 1200, endMs: 4800 });
    expect(() => core.setRallyBounds(state, 0, 4800, 1200)).toThrow(/after/);
  });

  it("finds the rally a winner key refers to", () => {
    const state = freshState();
    core.startRally(state, 1000);
    core.endRally(state, 3000);
    core.startRally(state, 8000);
    core.endRally(state, 9000);
    expect(core.rallyIndexAt(state, 2000)).toBe(0);
    expect(core.rallyIndexAt(state, 5000)).toBe(0);
    expect(core.rallyIndexAt(state, 8500)).toBe(1);
    expect(core.rallyIndexAt(state, 500)).toBe(-1);
  });
});

const AUDIO_SCRIPT_WITH_LEVELS = [
  "import base64, json, sys",
  "from simulation.audio import render_audio, SAMPLE_RATE",
  "hits, seconds, hit, noise = json.loads(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])",
  "samples, _ = render_audio(hits, seconds, seed=5, hit_amplitude=hit, noise_amplitude=noise)",
  "print(json.dumps({'rate': SAMPLE_RATE, 'samples': base64.b64encode(samples.astype('<f4').tobytes()).decode()}))",
].join("\n");

function audioAt(hits: number[], seconds: number, hit: number, noise: number) {
  const out = JSON.parse(
    execFileSync(python, ["-c", AUDIO_SCRIPT_WITH_LEVELS, JSON.stringify(hits), String(seconds), String(hit), String(noise)], {
      cwd: workerDir,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    }),
  );
  const bytes = Buffer.from(out.samples, "base64");
  return { rate: out.rate as number, samples: new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4) };
}

function foundAll(hits: number[], onsets: any[]) {
  return hits.every(t => onsets.some(o => Math.abs(o.timeMs - t * 1000) <= 5));
}

describe("review fixes: audio recall", () => {
  const hits = [0.6, 1.5, 2.4, 3.3];

  it("finds quiet hits", () => {
    const audio = audioAt(hits, 4.0, 0.1, 0.01);
    expect(foundAll(hits, core.detectOnsets(audio.samples, audio.rate))).toBe(true);
  });

  it("finds hits over loud background noise", () => {
    const audio = audioAt(hits, 4.0, 0.5, 0.05);
    expect(foundAll(hits, core.detectOnsets(audio.samples, audio.rate))).toBe(true);
  });

  it("finds exactly the hits in otherwise digital silence", () => {
    const audio = audioAt(hits, 4.0, 0.5, 0.0);
    const onsets = core.detectOnsets(audio.samples, audio.rate);
    expect(foundAll(hits, onsets)).toBe(true);
    expect(onsets).toHaveLength(hits.length);
  });
});

describe("review fixes: loading and autosave", () => {
  const fingerprint = { name: "IMG_1234.MOV", size: 1000, lastModified: 5, width: 1920, height: 1080, durationS: 12.5 };

  it("restores an autosave only for the same video", () => {
    expect(core.autosaveMatches({ fingerprint }, { ...fingerprint })).toBe(true);
    expect(core.autosaveMatches({ fingerprint }, { ...fingerprint, size: 999 })).toBe(false);
    expect(core.autosaveMatches({ fingerprint }, { ...fingerprint, durationS: 14 })).toBe(false);
    expect(core.autosaveMatches({}, fingerprint)).toBe(false);
  });

  it("validates a loaded labels file exactly as the Python validator would", () => {
    const valid = core.buildGolden(freshState()).doc;
    const docs = [
      valid,
      { ...valid, courtKeypoints: [{ name: "<img src=x onerror=alert(1)>", x: 1, y: 1 }] },
      { ...valid, contacts: [{ timeMs: "1000" }] },
      { ...valid, rallies: [{ startMs: "<b>x</b>", endMs: 5, winner: "near" }] },
      { ...valid, contacts: [{ timeMs: 1000 }], shots: [{ timeMs: 1000, label: "clear", landing: null }], poses: [{ timeMs: 1000, player: "near", keypoints: [[1, 1, 2]] }] },
      { ...valid, contacts: [{ timeMs: 1000 }], shots: [], poses: [{ timeMs: 1300, player: "near", keypoints: POSE }] },
      { ...valid, selectedPlayer: "far", contacts: [{ timeMs: 1000 }], shots: [], poses: [{ timeMs: 1000, player: "near", keypoints: POSE }] },
      "not an object",
    ];
    for (const doc of docs) {
      const pythonErrors = JSON.parse(
        runPython("import json,sys; from evaluation.golden import validate_golden; print(json.dumps(validate_golden(json.load(sys.stdin))))", JSON.stringify(doc)),
      );
      expect(core.validateGolden(doc).length > 0).toBe(pythonErrors.length > 0);
    }
  });

  it("escapes text before it is shown as HTML", () => {
    expect(core.escapeHtml(`<img src=x onerror="alert('1')">&`)).toBe("&lt;img src=x onerror=&quot;alert(&#39;1&#39;)&quot;&gt;&amp;");
  });
});

// Deterministic synthetic sounds for onset tests that need shapes the Python generator lacks.
function noiseSource(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 31 - 1;
  };
}

function sustainedAudio({ rate = 48000, seconds = 4, background = 0.01, loudFrom = 1.5, loudTo = 2.3, loud = 0.08, hits = [] as number[] }) {
  const random = noiseSource(7);
  const samples = new Float32Array(Math.round(rate * seconds));
  for (let i = 0; i < samples.length; i += 1) {
    const t = i / rate;
    samples[i] = random() * (t >= loudFrom && t < loudTo ? loud : background);
  }
  for (const hit of hits) {
    const start = Math.round(hit * rate);
    for (let i = 0; i < rate * 0.03 && start + i < samples.length; i += 1) {
      samples[start + i] += 0.5 * Math.exp(-i / (rate * 0.006)) * Math.sin((2 * Math.PI * 4000 * i) / rate);
    }
  }
  return { rate, samples };
}

describe("review fixes: audio onsets need a sharp attack", () => {
  it("a loud stretch shorter than the noise-floor window (a shout, a squeak and its echo) gives at most one onset", () => {
    const audio = sustainedAudio({});
    const onsets = core.detectOnsets(audio.samples, audio.rate);
    expect(onsets.length).toBeLessThanOrEqual(1);
  });

  it("a hit inside a loud stretch is still found", () => {
    const audio = sustainedAudio({ hits: [2.0] });
    const onsets = core.detectOnsets(audio.samples, audio.rate);
    expect(onsets.some((o: any) => Math.abs(o.timeMs - 2000) <= 5)).toBe(true);
  });
});
