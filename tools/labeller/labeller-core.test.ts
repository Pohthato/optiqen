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
