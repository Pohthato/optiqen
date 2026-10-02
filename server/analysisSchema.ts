import { z } from "zod";

export const cornerSchema = z.object({
  label: z.enum(["nearLeft", "nearRight", "farRight", "farLeft"]),
  x: z.number().finite().min(0).max(100),
  y: z.number().finite().min(0).max(100),
});

const evidenceSchema = z.object({
  frame: z.number().int().nonnegative(),
  timeMs: z.number().nonnegative(),
  confidence: z.number().min(0).max(1),
  source: z.enum(["pose", "shuttle", "racket", "court"]),
});

const eventSchema = z.object({
  type: z.enum(["contact", "split_step", "recovery_complete", "racket_swing", "rally_start", "rally_end"]),
  frame: z.number().int().nonnegative(),
  timeMs: z.number().nonnegative(),
  confidence: z.number().min(0).max(1),
  source: z.enum(["pose", "shuttle", "racket", "court"]),
});

const shotSchema = z.object({
  frame: z.number().int().nonnegative(),
  timeMs: z.number().nonnegative(),
  confidence: z.number().min(0).max(1),
  label: z.string().min(1).max(32),
  verified: z.boolean(),
  direction: z.enum(["cross", "straight", "unknown"]),
  depth: z.enum(["short", "mid", "back", "unknown"]),
  landingSide: z.enum(["opponent_half", "own_half", "unclear"]).optional(),
  rallyIndex: z.number().int().nonnegative(),
});

export const analysisResultSchema = z.object({
  processingVersion: z.string().min(1).max(96),
  calibration: z.object({
    corners: z.array(cornerSchema).min(3).max(4),
    confidence: z.enum(["unverified", "provisional", "validated", "image_space_only"]),
    supportsCourtMapping: z.boolean(),
    guidance: z.string().min(1).max(1_000).default("Worker calibration guidance was not provided; treat geometry as unverified."),
    courtType: z.enum(["singles", "doubles"]).optional(),
  }),
  courtType: z.enum(["singles", "doubles"]).optional(),
  camera: z.object({
    cameraTier: z.enum(["validated", "approximate", "unavailable"]),
    rmsPx: z.number().finite().nonnegative(),
    focalPx: z.number().finite().positive(),
    k1: z.number().finite(),
    looFloorCm: z.number().finite().nonnegative().nullable(),
    geometryTier: z.enum(["A", "B", "C"]),
    elevationDeg: z.number().finite(),
    visibleFraction: z.number().min(0).max(1),
    reasons: z.array(z.string().min(1).max(300)).max(8),
  }).nullable().optional(),
  quality: z.object({
    usableFrameRatio: z.number().min(0).max(1),
    poseTrackConfidence: z.number().min(0).max(1),
    shuttleTrackConfidence: z.number().min(0).max(1),
    courtReprojectionErrorPx: z.number().nonnegative().max(10_000).optional(),
  }),
  metrics: z.array(z.object({
    metric: z.string().min(1).max(96),
    value: z.number().finite(),
    unit: z.string().min(1).max(64),
    direction: z.enum(["higher_is_better", "lower_is_better", "contextual"]),
    confidence: z.number().min(0).max(1),
    evidenceFrames: z.array(evidenceSchema).min(1).max(48),
  })).max(64),
  shotDistribution: z.record(z.string().min(1).max(64), z.number().nonnegative().finite()).refine(
    distribution => Object.values(distribution).every(Number.isInteger),
    "Shot distribution counts must be whole numbers."
  ),
  events: z.array(eventSchema).max(240).optional(),
  rallies: z.array(z.object({
    index: z.number().int().nonnegative(),
    startMs: z.number().nonnegative(),
    endMs: z.number().nonnegative(),
    contactCount: z.number().int().nonnegative(),
    verifiedShots: z.number().int().nonnegative(),
  })).max(48).optional(),
  shots: z.array(shotSchema).max(240).optional(),
  annotatedVideoStorageKey: z.string().min(1).max(768).optional(),
  overlays: z.array(z.object({
    timeMs: z.number().nonnegative(),
    errors: z.partialRecord(z.enum(["Skeleton", "Shuttle", "Racket", "Court map"]), z.string().min(1).max(300)).optional(),
    skeleton: z.array(z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) })).max(33).optional(),
    shuttle: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }).optional(),
    racket: z.object({ grip: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }), head: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }) }).optional(),
    courtPosition: z.object({ x: z.number().min(0).max(100), y: z.number().min(0).max(100), confidence: z.number().min(0).max(1) }).optional(),
  })).max(18_000).optional(),
  diagnostics: z.object({
    sampledFrames: z.number().int().nonnegative().max(1_000_000),
    sourceFps: z.number().positive().max(960),
    sampleFps: z.number().positive().max(120),
    poseSampleFps: z.number().positive().max(120).optional(),
    shuttleSampleFps: z.number().positive().max(120).optional(),
    racketSampleFps: z.number().positive().max(120).optional(),
    timingsMs: z.object({
      download: z.number().nonnegative(), decode: z.number().nonnegative(), inference: z.number().nonnegative(), render: z.number().nonnegative(), total: z.number().nonnegative(),
    }),
    modelVersions: z.record(z.string().min(1).max(64), z.string().min(1).max(128)),
    warnings: z.array(z.string().min(1).max(400)).max(32),
  }).optional(),
}).superRefine((result, ctx) => {
  if (result.calibration.supportsCourtMapping && result.calibration.confidence !== "validated") {
    ctx.addIssue({ code: "custom", path: ["calibration"], message: "Court mapping may only be enabled after worker validation." });
  }
  if (!result.calibration.supportsCourtMapping && result.metrics.some(metric => /court coverage|metres travelled|court position/i.test(metric.metric))) {
    ctx.addIssue({ code: "custom", path: ["metrics"], message: "Court-space metrics require accepted court calibration." });
  }
});

export type ValidatedAnalysisResult = z.infer<typeof analysisResultSchema>;
