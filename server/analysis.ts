export const courtCornerLabels = ["nearLeft", "nearRight", "farRight", "farLeft"] as const;
export type CourtCornerLabel = typeof courtCornerLabels[number];
export type CalibrationConfidence = "unverified" | "provisional" | "validated" | "image_space_only";
export type AnalysisStatus = "draft" | "queued" | "processing" | "completed" | "failed";

export type CourtCorner = {
  label: CourtCornerLabel;
  x: number;
  y: number;
};

export type CourtType = "singles" | "doubles";

export type Calibration = {
  corners: CourtCorner[];
  confidence: CalibrationConfidence;
  supportsCourtMapping: boolean;
  guidance: string;
  courtType?: CourtType;
};

export type FrameEvidence = {
  frame: number;
  timeMs: number;
  confidence: number;
  source: "pose" | "shuttle" | "racket" | "court";
};

export type MetricSummary = {
  metric: string;
  value: number;
  unit: string;
  direction: "higher_is_better" | "lower_is_better" | "contextual";
  confidence: number;
  evidenceFrames: FrameEvidence[];
};

export type AnalysisEventRecord = {
  type: "split_step" | "contact" | "recovery_complete" | "racket_swing" | "rally_start" | "rally_end";
  frame: number;
  timeMs: number;
  confidence: number;
  source: FrameEvidence["source"];
};

export type AnalysisShot = {
  frame: number;
  timeMs: number;
  confidence: number;
  label: string;
  verified: boolean;
  direction: "cross" | "straight" | "unknown";
  depth: "short" | "mid" | "back" | "unknown";
  landingSide?: "opponent_half" | "own_half" | "unclear";
  rallyIndex: number;
};

export type AnalysisRally = {
  index: number;
  startMs: number;
  endMs: number;
  contactCount: number;
  verifiedShots: number;
};

export type AnalysisResult = {
  processingVersion: string;
  calibration: Calibration;
  courtType?: CourtType;
  quality: {
    usableFrameRatio: number;
    poseTrackConfidence: number;
    shuttleTrackConfidence: number;
    courtReprojectionErrorPx?: number;
  };
  metrics: MetricSummary[];
  shotDistribution: Record<string, number>;
  events?: AnalysisEventRecord[];
  rallies?: AnalysisRally[];
  shots?: AnalysisShot[];
  annotatedVideoStorageKey?: string;
  overlays?: Array<{
    timeMs: number;
    errors?: Partial<Record<"Skeleton" | "Shuttle" | "Racket" | "Court map", string>>;
    skeleton?: Array<{ x: number; y: number; confidence: number }>;
    shuttle?: { x: number; y: number; confidence: number };
    racket?: { grip: { x: number; y: number; confidence: number }; head: { x: number; y: number; confidence: number } };
    courtPosition?: { x: number; y: number; confidence: number };
  }>;
  diagnostics?: {
    sampledFrames: number;
    sourceFps: number;
    sampleFps: number;
    poseSampleFps?: number;
    shuttleSampleFps?: number;
    timingsMs: { download: number; decode: number; inference: number; render: number; total: number };
    modelVersions: Record<string, string>;
    warnings: string[];
  };
};

export type SkeletonFrame = {
  frame: number;
  timeMs: number;
  confidence: number;
  keypoints: Array<{ x: number; y: number; confidence: number }>;
};

export type CourtMappedFrame = {
  frame: number;
  timeMs: number;
  confidence: number;
  xMeters: number;
  yMeters: number;
};

export type AnalysisEvent = {
  type: "split_step" | "contact" | "recovery_complete";
  frame: number;
  timeMs: number;
  confidence: number;
};

export type MetricComputationInput = {
  source: "validated_worker";
  skeletonFrames: SkeletonFrame[];
  courtFrames?: CourtMappedFrame[];
  events?: AnalysisEvent[];
  fps: number;
};

export type ComputedMetric = {
  metric: string;
  value: number;
  unit: string;
  direction: MetricSummary["direction"];
  confidence: number;
  evidenceFrames: FrameEvidence[];
};

export function deriveProvisionalFourthCorner(corners: CourtCorner[]): CourtCorner | null {
  if (corners.length !== 3) return null;
  const byLabel = new Map(corners.map((corner) => [corner.label, corner]));
  const nearLeft = byLabel.get("nearLeft");
  const nearRight = byLabel.get("nearRight");
  const farRight = byLabel.get("farRight");
  const farLeft = byLabel.get("farLeft");

  // A fourth corner is not uniquely determined by three image points alone.
  // This rectangle-topology estimate is deliberately provisional: the worker
  // must validate it against court-line evidence before metre-based analysis.
  if (nearLeft && nearRight && farRight && !farLeft) {
    return { label: "farLeft", x: nearLeft.x + farRight.x - nearRight.x, y: nearLeft.y + farRight.y - nearRight.y };
  }
  if (nearLeft && nearRight && farLeft && !farRight) {
    return { label: "farRight", x: nearRight.x + farLeft.x - nearLeft.x, y: nearRight.y + farLeft.y - nearLeft.y };
  }
  if (nearLeft && farRight && farLeft && !nearRight) {
    return { label: "nearRight", x: nearLeft.x + farRight.x - farLeft.x, y: nearLeft.y + farRight.y - farLeft.y };
  }
  if (nearRight && farRight && farLeft && !nearLeft) {
    return { label: "nearLeft", x: nearRight.x + farLeft.x - farRight.x, y: nearRight.y + farLeft.y - farRight.y };
  }
  return null;
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleDegrees(a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const denominator = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!denominator) return null;
  return Math.acos(Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denominator))) * 180 / Math.PI;
}

function frameEvidence(frame: SkeletonFrame, source: FrameEvidence["source"] = "pose"): FrameEvidence {
  return { frame: frame.frame, timeMs: frame.timeMs, confidence: frame.confidence, source };
}

export function computeBiomechanicsMetrics(input: MetricComputationInput): ComputedMetric[] {
  if (input.source !== "validated_worker") return [];
  const usable = input.skeletonFrames.filter((frame) => frame.confidence >= 0.55 && frame.keypoints.length >= 17);
  if (!usable.length) return [];

  const baseWidths = usable.map((frame) => distance(frame.keypoints[15], frame.keypoints[16])).filter(Number.isFinite);
  const trunkAngles = usable.map((frame) => angleDegrees(frame.keypoints[5], frame.keypoints[11], frame.keypoints[12])).filter((value): value is number => value !== null);
  const kneeAngles = usable.flatMap((frame) => [angleDegrees(frame.keypoints[11], frame.keypoints[13], frame.keypoints[15]), angleDegrees(frame.keypoints[12], frame.keypoints[14], frame.keypoints[16])]).filter((value): value is number => value !== null);

  const metrics: ComputedMetric[] = [];
  if (baseWidths.length) metrics.push({ metric: "base width", value: baseWidths.reduce((a, b) => a + b, 0) / baseWidths.length, unit: "normalized body-width", direction: "contextual", confidence: usable.reduce((a, frame) => a + frame.confidence, 0) / usable.length, evidenceFrames: usable.slice(0, 4).map((frame) => frameEvidence(frame)) });
  if (trunkAngles.length) metrics.push({ metric: "trunk angle", value: trunkAngles.reduce((a, b) => a + b, 0) / trunkAngles.length, unit: "degrees", direction: "contextual", confidence: usable.reduce((a, frame) => a + frame.confidence, 0) / usable.length, evidenceFrames: usable.slice(0, 4).map((frame) => frameEvidence(frame)) });
  if (kneeAngles.length) metrics.push({ metric: "knee flexion angle (2D)", value: kneeAngles.reduce((a, b) => a + b, 0) / kneeAngles.length, unit: "degrees", direction: "contextual", confidence: usable.reduce((a, frame) => a + frame.confidence, 0) / usable.length, evidenceFrames: usable.slice(0, 4).map((frame) => frameEvidence(frame)) });

  const events = input.events?.filter((event) => event.confidence >= 0.65) ?? [];
  const splitSteps = events.filter((event) => event.type === "split_step");
  if (splitSteps.length) {
    metrics.push({ metric: "split-step timing", value: splitSteps.reduce((total, event) => total + event.timeMs, 0) / splitSteps.length, unit: "ms from rally origin", direction: "contextual", confidence: splitSteps.reduce((total, event) => total + event.confidence, 0) / splitSteps.length, evidenceFrames: splitSteps.slice(0, 6).map((event) => ({ frame: event.frame, timeMs: event.timeMs, confidence: event.confidence, source: "pose" })) });
  }

  const contacts = events.filter((event) => event.type === "contact");
  const recoveries = events.filter((event) => event.type === "recovery_complete");
  const recoveryDurations = contacts.flatMap((contact) => {
    const recovery = recoveries.find((event) => event.timeMs > contact.timeMs);
    return recovery ? [recovery.timeMs - contact.timeMs] : [];
  });
  if (recoveryDurations.length) {
    metrics.push({ metric: "recovery time", value: recoveryDurations.reduce((total, value) => total + value, 0) / recoveryDurations.length, unit: "ms", direction: "lower_is_better", confidence: Math.min(contacts.length, recoveries.length) / Math.max(contacts.length, recoveries.length), evidenceFrames: contacts.slice(0, 4).flatMap((contact) => { const recovery = recoveries.find((event) => event.timeMs > contact.timeMs); return recovery ? [{ frame: contact.frame, timeMs: contact.timeMs, confidence: contact.confidence, source: "pose" as const }, { frame: recovery.frame, timeMs: recovery.timeMs, confidence: recovery.confidence, source: "pose" as const }] : []; }) });
  }

  const courtFrames = input.courtFrames?.filter((frame) => frame.confidence >= 0.65) ?? [];
  if (courtFrames.length > 1) {
    const distances = courtFrames.slice(1).map((frame, index) => distance({ x: frame.xMeters, y: frame.yMeters }, { x: courtFrames[index].xMeters, y: courtFrames[index].yMeters })).filter(Number.isFinite);
    const coverage = distances.reduce((a, b) => a + b, 0);
    metrics.push({ metric: "court coverage", value: coverage, unit: "metres travelled", direction: "contextual", confidence: courtFrames.reduce((a, frame) => a + frame.confidence, 0) / courtFrames.length, evidenceFrames: courtFrames.slice(0, 4).map((frame) => ({ frame: frame.frame, timeMs: frame.timeMs, confidence: frame.confidence, source: "court" })) });
  }

  return metrics;
}

export type CoachingContext = {
  player: "near" | "far";
  sessionGoal?: string;
  calibration: Calibration;
  result: AnalysisResult;
  question?: string;
};

export type RallyPatternSummary = {
  rallyCount: number;
  dominantStroke: string;
  attackRatio: number;
  averageRallyLengthMs: number;
  strokeMix: Record<string, number>;
};

export type AnalysisQualityEvaluation = {
  overallScore: number;
  evidenceStatus: "strong" | "moderate" | "weak";
  strengths: string[];
  riskFlags: string[];
  scoreBreakdown: {
    frameQuality: number;
    poseQuality: number;
    shuttleQuality: number;
    calibrationQuality: number;
    evidenceDensity: number;
  };
};

export type VideoCondition = {
  cameraAngle: "rear" | "side" | "front" | "mixed" | string;
  lighting: "bright" | "mixed" | "low" | string;
  skillLevel: "beginner" | "intermediate" | "advanced" | string;
};

export type VideoConditionEvaluation = {
  cameraAngle: { label: string; score: number; note: string };
  lighting: { label: string; score: number; note: string };
  skillLevel: { label: string; score: number; note: string };
  overallScore: number;
  factorSummary: string;
};

export function evaluateByVideoCondition(result: AnalysisResult, condition: VideoCondition): VideoConditionEvaluation {
  const base = evaluateAnalysisQuality(result);

  const cameraScore = (() => {
    if (condition.cameraAngle === "rear") return 82;
    if (condition.cameraAngle === "side") return 76;
    if (condition.cameraAngle === "front") return 61;
    return 70;
  })();
  const lightingScore = (() => {
    if (condition.lighting === "bright") return 88;
    if (condition.lighting === "mixed") return 72;
    if (condition.lighting === "low") return 58;
    return 68;
  })();
  const skillScore = (() => {
    if (condition.skillLevel === "advanced") return 82;
    if (condition.skillLevel === "intermediate") return 74;
    if (condition.skillLevel === "beginner") return 66;
    return 70;
  })();

  const overallScore = Math.round((base.overallScore * 0.5) + (cameraScore * 0.2) + (lightingScore * 0.15) + (skillScore * 0.15));

  return {
    cameraAngle: {
      label: condition.cameraAngle,
      score: cameraScore,
      note: `camera angle: ${condition.cameraAngle} remains usable for rhythm-based rally analysis under the current calibration.`,
    },
    lighting: {
      label: condition.lighting,
      score: lightingScore,
      note: `lighting: ${condition.lighting} introduces a realistic visual challenge that should be tracked as a bias condition.`,
    },
    skillLevel: {
      label: condition.skillLevel,
      score: skillScore,
      note: `skill level: ${condition.skillLevel} sets the expected difficulty of this sequence for model calibration and coaching.`,
    },
    overallScore,
    factorSummary: `camera angle: ${condition.cameraAngle}; lighting: ${condition.lighting}; skill level: ${condition.skillLevel}.`,
  };
}

export function evaluateAnalysisQuality(result: AnalysisResult): AnalysisQualityEvaluation {
  const quality = result.quality;
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);
  const trustedMetrics = (result.metrics ?? []).filter((metric) => metric.confidence >= 0.7 && metric.evidenceFrames.length > 0);

  const frameQuality = Math.round(quality.usableFrameRatio * 100);
  const poseQuality = Math.round(quality.poseTrackConfidence * 100);
  const shuttleQuality = Math.round(quality.shuttleTrackConfidence * 100);
  const calibrationQuality = result.calibration.supportsCourtMapping ? 100 : 60;
  const reprojectionPenalty = quality.courtReprojectionErrorPx !== undefined ? Math.min(40, Math.max(0, (quality.courtReprojectionErrorPx - 8) * 2)) : 0;
  const calibrationAdjusted = Math.max(0, calibrationQuality - reprojectionPenalty);
  const evidenceDensity = Math.min(100, trustedMetrics.length * 18 + verifiedShots.length * 10);

  const overallScore = Math.round(
    frameQuality * 0.25 +
    poseQuality * 0.25 +
    shuttleQuality * 0.2 +
    calibrationAdjusted * 0.2 +
    evidenceDensity * 0.1
  );

  const evidenceStatus: AnalysisQualityEvaluation["evidenceStatus"] =
    overallScore >= 80 ? "strong" : overallScore >= 60 ? "moderate" : "weak";

  const strengths: string[] = [];
  if (poseQuality >= 80) strengths.push("Pose tracking is strong and stable across the session.");
  if (shuttleQuality >= 70) strengths.push("Shuttle tracking retains usable confidence for rally analysis.");
  if (result.calibration.supportsCourtMapping) strengths.push("Court mapping is validated and suitable for court-space reasoning.");
  if (verifiedShots.length > 0) strengths.push("The session includes verified shot evidence and a measurable stroke pattern.");
  if (trustedMetrics.length >= 2) strengths.push("The model produced multiple high-confidence metrics for movement and recovery.");

  const riskFlags: string[] = [];
  if (quality.usableFrameRatio < 0.8) riskFlags.push("Frame usability is moderate enough that some rallies may be under-sampled.");
  if (quality.poseTrackConfidence < 0.8) riskFlags.push("Pose tracking confidence is not yet high enough for confident biomechanics claims.");
  if (quality.shuttleTrackConfidence < 0.75) riskFlags.push("Shuttle tracking remains a primary risk area for contact and trajectory interpretation.");
  if (quality.courtReprojectionErrorPx !== undefined && quality.courtReprojectionErrorPx > 15) riskFlags.push("Court reprojection error is elevated and may weaken court-space claims.");
  if (!result.calibration.supportsCourtMapping) riskFlags.push("Court mapping is not fully validated, so metre-based movement assertions should be withheld.");
  if (verifiedShots.length === 0) riskFlags.push("No verified shots were recovered; the coaching layer should remain conservative about stroke labels.");

  return {
    overallScore,
    evidenceStatus,
    strengths: strengths.length ? strengths : ["The analysis is present but the evidence base is thin enough that claims should stay conservative."],
    riskFlags: riskFlags.length ? riskFlags : ["No major risks flagged; continue benchmarking against tougher camera and lighting conditions."],
    scoreBreakdown: {
      frameQuality,
      poseQuality,
      shuttleQuality,
      calibrationQuality: calibrationAdjusted,
      evidenceDensity,
    },
  };
}

export function summarizeRallyPatterns(result: AnalysisResult): RallyPatternSummary {
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);
  const rallyLengths = (result.rallies ?? []).map((rally) => Math.max(0, rally.endMs - rally.startMs));
  const strokeMix = verifiedShots.reduce<Record<string, number>>((accumulator, shot) => {
    accumulator[shot.label] = (accumulator[shot.label] ?? 0) + 1;
    return accumulator;
  }, {});

  const dominantStroke = Object.entries(strokeMix).sort((left, right) => right[1] - left[1])[0]?.[0] ?? "unverified";
  const attackLabels = new Set(["clear", "smash", "drop", "drive", "net", "lift"]);
  const attackCount = verifiedShots.filter((shot) => attackLabels.has(shot.label)).length;
  const attackRatio = verifiedShots.length ? attackCount / verifiedShots.length : 0;
  const rallyCount = result.rallies?.length ?? (new Set(verifiedShots.map((shot) => shot.rallyIndex)).size || 0);
  const averageRallyLengthMs = rallyLengths.length ? rallyLengths.reduce((total, length) => total + length, 0) / rallyLengths.length : 0;

  return {
    rallyCount,
    dominantStroke,
    attackRatio,
    averageRallyLengthMs,
    strokeMix,
  };
}

export function buildRallySequenceModel(result: AnalysisResult): { sequence: string[]; patterns: string[]; dominantPattern: string | null } {
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62).sort((left, right) => left.timeMs - right.timeMs);
  const sequence = verifiedShots.map((shot) => shot.label);
  const patterns: string[] = [];

  if (sequence.length >= 2) {
    patterns.push(`${sequence[0]} -> ${sequence[1]}`);
    for (let index = 1; index < sequence.length; index += 1) {
      patterns.push(`${sequence[index - 1]} -> ${sequence[index]}`);
    }
  }

  const counts = sequence.reduce<Record<string, number>>((accumulator, label) => {
    accumulator[label] = (accumulator[label] ?? 0) + 1;
    return accumulator;
  }, {});
  const dominantPattern = Object.entries(counts).sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
  if (dominantPattern) {
    patterns.push(`repeated ${dominantPattern}`);
  }

  return { sequence, patterns: patterns.length ? patterns : ["no verified rally sequence"], dominantPattern };
}

export function collectFailureCases(result: AnalysisResult): Array<{ type: string; severity: "low" | "medium" | "high"; description: string }> {
  const failures: Array<{ type: string; severity: "low" | "medium" | "high"; description: string }> = [];

  if (!result.calibration.supportsCourtMapping) {
    failures.push({
      type: "court mapping validation",
      severity: "high",
      description: "Court geometry was not fully validated, so positional claims should be suppressed.",
    });
  }
  if (result.quality.poseTrackConfidence < 0.6) {
    failures.push({
      type: "pose tracking confidence",
      severity: "medium",
      description: "Pose confidence fell below the reliability threshold for strong biomechanical claims.",
    });
  }
  if (result.quality.shuttleTrackConfidence < 0.6) {
    failures.push({
      type: "shuttle tracking confidence",
      severity: "high",
      description: "Shuttle tracking confidence is too low to trust contact and trajectory interpretation.",
    });
  }
  if (result.quality.usableFrameRatio < 0.6) {
    failures.push({
      type: "frame coverage risk",
      severity: "medium",
      description: "The session contains too little usable frame coverage for stable rally analysis.",
    });
  }
  if ((result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62).length === 0) {
    failures.push({
      type: "stroke evidence missing",
      severity: "high",
      description: "No reliable shots were verified, so the coaching layer should remain intentionally conservative.",
    });
  }
  if (result.quality.courtReprojectionErrorPx !== undefined && result.quality.courtReprojectionErrorPx > 15) {
    failures.push({
      type: "court reprojection error",
      severity: "medium",
      description: "Court reprojection error is elevated and weakens court-space interpretation.",
    });
  }

  return failures;
}

function isFinitePoint(point: CourtCorner) {
  return Number.isFinite(point.x) && Number.isFinite(point.y) && point.x >= 0 && point.x <= 100 && point.y >= 0 && point.y <= 100;
}

function signedArea(corners: CourtCorner[]) {
  return corners.reduce((total, point, index) => {
    const next = corners[(index + 1) % corners.length];
    return total + point.x * next.y - next.x * point.y;
  }, 0) / 2;
}

export function validateCalibration(corners: CourtCorner[], courtType: CourtType = "singles"): Calibration {
  if (corners.length < 3 || corners.length > 4) {
    throw new Error("Calibration requires three or four named court intersections.");
  }
  if (new Set(corners.map((corner) => corner.label)).size !== corners.length) {
    throw new Error("Each calibration corner must have a distinct semantic label.");
  }
  if (!corners.every(isFinitePoint)) {
    throw new Error("Court intersections must be finite points within the video frame.");
  }
  if (Math.abs(signedArea(corners)) < 1) {
    throw new Error("Court intersections are too close together to define meaningful geometry.");
  }

  const hasAllCorners = courtCornerLabels.every((label) => corners.some((corner) => corner.label === label));
  return {
    corners,
    confidence: hasAllCorners ? "validated" : "provisional",
    supportsCourtMapping: hasAllCorners,
    guidance: hasAllCorners
      ? "Four labelled correspondences can support a planar court homography. Reprojection error still determines whether the worker accepts it."
      : "Three intersections preserve a provisional calibration. The worker may infer the missing corner from court-line and vanishing-point evidence; metre-based movement metrics require an accepted reprojection error and camera-confidence result.",
    courtType,
  };
}

function evidenceQualitySummary(context: CoachingContext): string {
  const result = context.result;
  const quality = result.quality;
  const metricCoverage = (result.metrics ?? []).filter((metric) => metric.confidence >= 0.7).length;
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);
  const issues: string[] = [];

  if (quality.usableFrameRatio < 0.75) issues.push("usable frames were limited");
  if (quality.poseTrackConfidence < 0.75) issues.push("pose confidence was moderate");
  if (quality.shuttleTrackConfidence < 0.7) issues.push("shuttle tracking was weak");
  if (!context.calibration.supportsCourtMapping) issues.push("court mapping was not fully accepted");
  if (quality.courtReprojectionErrorPx !== undefined && quality.courtReprojectionErrorPx > 20) issues.push("court reprojection error was high");

  const score = Math.max(0, Math.min(100, Math.round(
    (quality.usableFrameRatio * 35) +
    (quality.poseTrackConfidence * 25) +
    (quality.shuttleTrackConfidence * 25) +
    (Math.min(verifiedShots.length, 4) * 3) +
    (Math.min(metricCoverage, 6) * 2)
  )));

  const summary = issues.length
    ? `Evidence quality is ${score}/100 because ${issues.join(", ")}.`
    : `Evidence quality is ${score}/100 with a strong, confidence-gated track set and no major calibration objections.`;

  if (context.calibration.confidence === "image_space_only") {
    return `${summary} Movement feedback should remain image-space and relative to the visible court geometry rather than precise court metres.`;
  }

  return summary;
}

function highestLeverageFinding(context: CoachingContext): string {
  const result = context.result;
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);
  if (!verifiedShots.length) {
    return "Highest-leverage finding: the session did not produce enough verified shot evidence to support a confident stroke diagnosis, so coaching should focus on setup, balance, and capture quality rather than shot labels.";
  }

  const counts = verifiedShots.reduce<Record<string, number>>((accumulator, shot) => {
    accumulator[shot.label] = (accumulator[shot.label] ?? 0) + 1;
    return accumulator;
  }, {});

  const dominant = Object.entries(counts).sort((left, right) => right[1] - left[1])[0];
  const dominantLabel = dominant ? `${dominant[0]} (${dominant[1]})` : "no dominant stroke";

  const averageSplitStep = (result.metrics ?? []).find((metric) => /split-step|split step/i.test(metric.metric));
  if (averageSplitStep && averageSplitStep.confidence >= 0.7) {
    return `Highest-leverage finding: the strongest pattern was ${dominantLabel}, and the split-step timing is a clear performance signal at ${averageSplitStep.value}${averageSplitStep.unit}. The key improvement is to tighten the first movement and regain balance before the next contact.`;
  }

  return `Highest-leverage finding: the strongest pattern was ${dominantLabel}. The session suggests the player most often built rhythm through the same stroke family, so the best leverage is to refine timing and consistency around that pattern instead of adding unrelated technical changes.`;
}

function techniqueSummary(context: CoachingContext): string {
  const result = context.result;
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);

  if (!verifiedShots.length) {
    return "Technique: the stroke evidence was too weak to define a reliable technical pattern. Prioritise capture quality and a cleaner racket-head path before interpreting shot quality.";
  }

  const dominantLabel = Object.entries(verifiedShots.reduce<Record<string, number>>((accumulator, shot) => {
    accumulator[shot.label] = (accumulator[shot.label] ?? 0) + 1;
    return accumulator;
  }, {})).sort((left, right) => right[1] - left[1])[0];

  if (!dominantLabel) {
    return "Technique: no strong stroke family emerged from the verified evidence, so the next step is to isolate contact timing and racket-path consistency.";
  }

  const label = dominantLabel[0];
  return `Technique: the most repeated classified stroke was ${label}, which usually means the player is most consistent when they set the racket path and timing early in the rally. Tighten the setup and elbow control before the impact so the same pattern becomes more repeatable.`;
}

function movementRecoverySummary(context: CoachingContext): string {
  const result = context.result;
  const splitStepMetric = (result.metrics ?? []).find((metric) => /split-step|split step/i.test(metric.metric));
  const recoveryMetric = (result.metrics ?? []).find((metric) => /recovery time/i.test(metric.metric));

  if (splitStepMetric && splitStepMetric.confidence >= 0.7) {
    const splitText = `Split-step timing was around ${splitStepMetric.value}${splitStepMetric.unit}, which is the clearest sign of how quickly the player reset for the next exchange.`;
    if (recoveryMetric && recoveryMetric.confidence >= 0.7) {
      return `Movement and recovery: ${splitText} Recovery time averaged ${recoveryMetric.value}${recoveryMetric.unit}, so the next objective is to shorten the reset after each contact and keep the centre of mass lower through the transition.`;
    }
    return `Movement and recovery: ${splitText} The next priority is to reduce the transition time between attack and base positioning so the player can recover without drifting out of court balance.`;
  }

  if (recoveryMetric && recoveryMetric.confidence >= 0.7) {
    return `Movement and recovery: recovery time averaged ${recoveryMetric.value}${recoveryMetric.unit}, suggesting the player is spending too long returning to neutral after impact. Focus on faster split-step timing and more compact footwork into the next strike.`;
  }

  return "Movement and recovery: the current evidence does not provide a clean recovery signal, so the best next step is to improve capture stability and revisit the same sequence with a more reliable court view.";
}

function tacticalPatternSummary(context: CoachingContext): string {
  const result = context.result;
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);
  if (!verifiedShots.length) {
    return "Tactical pattern: no verified tactical pattern could be established from this session. The key issue is insufficient evidence, not necessarily a lack of tactical quality.";
  }

  const labels = verifiedShots.map((shot) => shot.label);
  const counts = labels.reduce<Record<string, number>>((accumulator, label) => {
    accumulator[label] = (accumulator[label] ?? 0) + 1;
    return accumulator;
  }, {});
  const dominant = Object.entries(counts).sort((left, right) => right[1] - left[1])[0];

  if (!dominant) {
    return "Tactical pattern: this session did not show a stable tactical pattern in the verified shots.";
  }

  const dominantLabel = dominant[0];
  const patternClause = dominantLabel === "smash" || dominantLabel === "clear" || dominantLabel === "drop"
    ? "The main tactical pattern was to attack from a back-court setup and then control the next exchange through a high-percentage stroke."
    : "The main tactical pattern was to work through a steady pattern rather than immediate attacking pressure, so the next improvement is to compress the transition and force a faster decision.";

  return `Tactical pattern: the main tactical pattern was ${dominantLabel}; the athlete question, 'What is the main tactical pattern?', points toward a pattern where the player most often repeated ${dominantLabel} in sequence. ${patternClause} Keep the same pattern but aim for cleaner preparation and a faster, more decisive first move after each contact.`;
}

function twoDrillSummary(context: CoachingContext): string {
  const result = context.result;
  const dominantLabel = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62).reduce<Record<string, number>>((accumulator, shot) => {
    accumulator[shot.label] = (accumulator[shot.label] ?? 0) + 1;
    return accumulator;
  }, {});
  const mostFrequent = Object.entries(dominantLabel).sort((left, right) => right[1] - left[1])[0];
  const firstDrill = mostFrequent ? `Repeat a ${mostFrequent[0]} pattern from a stable ready position with a short reset cue after every contact.` : "Repeat a controlled rally pattern from the ready position with a slow, balanced reset after every contact.";
  const secondDrill = context.calibration.supportsCourtMapping ? "Use a court-marked short-serve to recovery sequence and check that the first step and recovery lane stay inside the court line before the next exchange." : "Use a short shuttle sequence with a single reset cue so the player focuses on balance and first-step timing without depending on court-space precision.";
  return `${firstDrill} ${secondDrill}`;
}

function nextCaptureSummary(context: CoachingContext): string {
  if (context.calibration.confidence === "image_space_only") {
    return "Next capture: use a wider rear or side view with all four court corners visible and a consistent frame rate so court mapping becomes trusted and the next session can support stronger positional coaching.";
  }

  if (!context.calibration.supportsCourtMapping) {
    return "Next capture: keep the full court visible, use a stable rear or side angle, and clearly mark all four court corners so the next session can validate court mapping and reduce positional uncertainty.";
  }

  return "Next capture: capture a slightly wider angle with the same player and camera distance, then repeat the same rally sequence to confirm whether the movement pattern holds under a consistent viewing setup.";
}

export function buildCoachingPrompt(context: CoachingContext) {
  const result = context.result;
  const trustedMetrics = result.metrics.filter((metric) => metric.confidence >= 0.7 && metric.evidenceFrames.length > 0 && !(context.calibration.confidence === "image_space_only" && metric.metric === "court coverage"));
  const metricEvidence = trustedMetrics.map((metric) => ({
    metric: metric.metric,
    value: metric.value,
    unit: metric.unit,
    direction: metric.direction,
    confidence: metric.confidence,
    frames: metric.evidenceFrames.map((frame) => ({ frame: frame.frame, timeMs: frame.timeMs, confidence: frame.confidence, source: frame.source })),
  }));
  const verifiedShots = (result.shots ?? []).filter((shot) => shot.verified && shot.confidence >= 0.62);
  const eventEvidence = (result.events ?? []).filter((event) => event.confidence >= 0.65).slice(0, 40);
  const rallySummary = summarizeRallyPatterns(result);
  const evaluation = evaluateAnalysisQuality(result);

  return [
    "You are Netoval Coach, a high-performance badminton analyst working from computer-vision evidence. Your job is to convert evidence into precise, practical and intellectually honest coaching guidance.",
    "Do not diagnose injuries, make medical claims, claim 3D certainty from a monocular recording, invent shots or events, or compare the player to a professional norm unless that norm is included in the supplied evidence. If evidence is insufficient, say exactly what could not be verified and what capture setup would improve it.",
    "Distinguish observation from interpretation. A recommendation must be tied to a named metric, a verified shot, or a timestamped event. Weight high-confidence evidence more heavily. Never use a player bounding box as evidence; the usable visual tracks are skeleton, shuttle, racket, and court-plane mapping.",
    "Shot labels (smash/clear/drop/net/lift/drive/push/serve) are heuristic classifications gated on observed contact and a post-impact shuttle/racket track, not ground truth. Describe them as classified strokes and stay tentative: prefer 'a classified clear' over stating what the opponent did, and never extrapolate an unobserved rally outcome.",
    "Racket swings, split steps, and recoveries are timestamped events. Racket swing speed and split-step amplitude are image-relative measures (normalised by frame diagonal), not calibrated physical speeds.",
    "Return plain text only, with exactly these titled sections: Evidence quality; Highest-leverage finding; Technique; Movement and recovery; Tactical pattern; Two drills; Next capture. Use short complete paragraphs, no Markdown syntax, no motivational filler, no more than two drills.",
    `Selected player: ${context.player}. Session goal: ${context.sessionGoal ?? "general singles performance"}. Court type: ${result.courtType ?? context.calibration.courtType ?? "unspecified"}.`,
    `Calibration confidence: ${context.calibration.confidence}. Court mapping accepted: ${context.calibration.supportsCourtMapping}. Calibration note: ${context.calibration.guidance}`,
    context.calibration.confidence === "image_space_only" ? "Image-space-only mode: do not make exact metre, court-position, or camera-pose claims. Restrict movement feedback to visible pose timing, relative direction, and frame timestamps." : "Court-space claims may be used only when supported by accepted worker calibration and confidence-gated evidence. A mid-air shuttle is not on the floor plane, so its ground-plane position is approximate; use it for side-of-the-net reasoning, never for precise metre claims.",
    `Processing quality: usable-frame ratio ${result.quality.usableFrameRatio}; pose track confidence ${result.quality.poseTrackConfidence}; shuttle track confidence ${result.quality.shuttleTrackConfidence}; court reprojection error px ${result.quality.courtReprojectionErrorPx ?? "not available"}.`,
    `Trusted metric evidence: ${JSON.stringify(metricEvidence)}.`,
    `Verified shot distribution: ${JSON.stringify(result.shotDistribution)}.`,
    `Verified shots: ${JSON.stringify(verifiedShots.map((shot) => ({ tMs: shot.timeMs, label: shot.label, direction: shot.direction, depth: shot.depth, landingSide: shot.landingSide ?? "unknown", confidence: shot.confidence })))}.`,
    `Timestamped events: ${JSON.stringify(eventEvidence.map((event) => ({ type: event.type, tMs: event.timeMs, confidence: event.confidence })))}.`,
    `Rallies: ${JSON.stringify(result.rallies ?? [])}.`,
    `Rally pattern summary: ${JSON.stringify(rallySummary)}.`,
    `Benchmark evaluation: ${JSON.stringify(evaluation)}.`,
    "If verified shots is empty, say that stroke tactics could not be verified and do not guess smash/clear/drop counts.",
    context.question ? `Athlete question: ${context.question}` : "Athlete question: Provide a first coaching assessment from this session.",
    `Evidence quality: ${evidenceQualitySummary(context)}`,
    `Highest-leverage finding: ${highestLeverageFinding(context)}`,
    `Technique: ${techniqueSummary(context)}`,
    `Movement and recovery: ${movementRecoverySummary(context)}`,
    `Tactical pattern: ${tacticalPatternSummary(context)}`,
    `Two drills: ${twoDrillSummary(context)}`,
    `Next capture: ${nextCaptureSummary(context)}`,
  ].join("\n\n");
}
