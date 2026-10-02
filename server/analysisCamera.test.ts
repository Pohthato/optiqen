// server/analysisCamera.test.ts
import { describe, expect, it } from "vitest";
import { analysisResultSchema } from "./analysisSchema";

const base = {
  processingVersion: "optiqen-badminton-worker-test",
  calibration: {
    corners: [
      { label: "nearLeft", x: 10, y: 80 },
      { label: "nearRight", x: 90, y: 80 },
      { label: "farRight", x: 70, y: 20 },
    ],
    confidence: "provisional",
    supportsCourtMapping: false,
    guidance: "test",
  },
  quality: { usableFrameRatio: 0.9, poseTrackConfidence: 0.9, shuttleTrackConfidence: 0.5 },
  metrics: [],
  shotDistribution: {},
};

const camera = {
  cameraTier: "approximate",
  rmsPx: 0,
  focalPx: 1012.4,
  k1: 0,
  looFloorCm: null,
  geometryTier: "A",
  elevationDeg: 24.4,
  visibleFraction: 1,
  reasons: [],
};

describe("analysis result camera summary", () => {
  it("keeps a worker camera summary through validation", () => {
    const parsed = analysisResultSchema.parse({ ...base, camera });
    expect(parsed.camera).toEqual(camera);
  });

  it("accepts a result with a null or missing camera summary", () => {
    expect(analysisResultSchema.parse({ ...base, camera: null }).camera).toBeNull();
    expect(analysisResultSchema.parse(base).camera).toBeUndefined();
  });

  it("rejects a geometry tier outside A, B and C", () => {
    expect(analysisResultSchema.safeParse({ ...base, camera: { ...camera, geometryTier: "D" } }).success).toBe(false);
  });
});
