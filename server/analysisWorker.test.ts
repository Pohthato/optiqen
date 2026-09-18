import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getAnalysisWorkerJob,
  resetWorkerWarmupCooldown,
  submitAnalysisWorkerJob,
  submitWorkerWarmup,
  validateWorkerResult,
  workerWarmupCooldownMs,
} from "./analysisWorker";

const request = {
  analysisId: "analysis-1",
  videoStorageKey: "analysis-sources/1/video.mp4",
  sourceVideoUrl: "https://storage.example.test/source.mp4?signature=short-lived",
  selectedPlayer: "near" as const,
  calibration: {
    corners: [],
    confidence: "provisional" as const,
    supportsCourtMapping: false,
    guidance: "Worker validation required.",
  },
  requestedLayers: ["skeleton" as const],
};

describe("analysis worker handoff", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("explains a 404 as a worker route or CV_WORKER_URL problem", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://worker.example.test");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 })
        )
    );

    await expect(submitAnalysisWorkerJob(request)).rejects.toThrow(
      "Check that CV_WORKER_URL points to the worker root and that POST /v1/analysis-jobs exists"
    );
  });

  it("submits RunPod jobs with the required input envelope and returns its job id", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubEnv("CV_WORKER_TOKEN", "runpod-test-key");
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ id: "runpod-job-1", status: "IN_QUEUE" }),
          { status: 200 }
        )
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(submitAnalysisWorkerJob(request)).resolves.toEqual({
      jobId: "runpod-job-1",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.runpod.ai/v2/endpoint-1/run",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining('"input"'),
      })
    );
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({ input: request, policy: { executionTimeout: 900_000, ttl: 3_600_000 } });
  });

  it("maps RunPod status responses to the Netoval lifecycle", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ id: "runpod-job-1", status: "IN_QUEUE" }),
          { status: 200 }
        )
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getAnalysisWorkerJob("runpod-job-1")).resolves.toEqual({
      status: "queued",
      result: undefined,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.runpod.ai/v2/endpoint-1/status/runpod-job-1",
      expect.anything()
    );
  });

  it("rejects a malformed RunPod submission response", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ status: "IN_QUEUE" }), { status: 200 })
        )
    );

    await expect(submitAnalysisWorkerJob(request)).rejects.toThrow(
      "invalid asynchronous job response"
    );
  });

  it("rejects an unknown RunPod status", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ status: "MYSTERY" }), { status: 200 })
        )
    );

    await expect(getAnalysisWorkerJob("runpod-job-1")).rejects.toThrow(
      "invalid job status"
    );
  });

  it("maps RunPod timeout to a failed Netoval job", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ status: "TIMED_OUT" }), { status: 200 })
        )
    );

    await expect(getAnalysisWorkerJob("runpod-job-1")).resolves.toEqual({
      status: "failed",
      error: "RunPod analysis job failed.",
    });
  });

  it("returns completed RunPod output that passes the existing result gate", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    const result = {
      processingVersion: "runpod-v1",
      metrics: [],
      calibration: { corners: [{ label: "nearLeft", x: 10, y: 80 }, { label: "nearRight", x: 90, y: 80 }, { label: "farRight", x: 70, y: 20 }, { label: "farLeft", x: 30, y: 20 }], confidence: "validated", supportsCourtMapping: true, guidance: "accepted" },
      shotDistribution: {},
      quality: { usableFrameRatio: 1, poseTrackConfidence: 1, shuttleTrackConfidence: 1 },
    } as any;
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ status: "COMPLETED", output: result }),
            { status: 200 }
          )
        )
    );

    const response = await getAnalysisWorkerJob("runpod-job-1");
    expect(response).toEqual({ status: "completed", result });
    expect(() => validateWorkerResult(response.result!)).not.toThrow();
  });

  it("preserves a completed payload with calibration, overlays, and annotated video metadata", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    const result = {
      processingVersion: "runpod-badminton-v1",
      calibration: {
        corners: [
          { label: "nearLeft", x: 10, y: 80 },
          { label: "nearRight", x: 90, y: 80 },
          { label: "farRight", x: 70, y: 20 },
          { label: "farLeft", x: 30, y: 20 },
        ],
        confidence: "validated",
        supportsCourtMapping: true,
        guidance: "accepted",
      },
      quality: {
        usableFrameRatio: 0.94,
        poseTrackConfidence: 0.91,
        shuttleTrackConfidence: 0.83,
        courtReprojectionErrorPx: 2.4,
      },
      metrics: [],
      shotDistribution: { clear: 3 },
      annotatedVideoStorageKey: "analysis-results/1/annotated.mp4",
      overlays: [
        {
          timeMs: 1000,
          skeleton: [{ x: 50, y: 50, confidence: 0.9 }],
          shuttle: { x: 60, y: 40, confidence: 0.8 },
          racket: {
            grip: { x: 51, y: 52, confidence: 0.88 },
            head: { x: 65, y: 35, confidence: 0.84 },
          },
          courtPosition: { x: 3.2, y: 5.1, confidence: 0.9 },
        },
      ],
    } as any;
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ status: "COMPLETED", output: result }),
            { status: 200 }
          )
        )
    );

    const response = await getAnalysisWorkerJob("runpod-job-1");
    expect(response.status).toBe("completed");
    expect(response.result?.annotatedVideoStorageKey).toBe(
      "analysis-results/1/annotated.mp4"
    );
    expect(response.result?.calibration.supportsCourtMapping).toBe(true);
    expect(response.result?.overlays?.[0]).toMatchObject({
      shuttle: expect.any(Object),
      racket: expect.any(Object),
      courtPosition: expect.any(Object),
    });
    expect(() => validateWorkerResult(response.result!)).not.toThrow();
  });

  it("maps a RunPod failure without fabricating a completed result", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ status: "FAILED", error: "worker crashed" }),
            { status: 200 }
          )
        )
    );

    await expect(getAnalysisWorkerJob("runpod-job-1")).resolves.toEqual({
      status: "failed",
      error: "worker crashed",
    });
  });

  it("accepts a valid queued worker job response", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://worker.example.test");
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ jobId: "job-123" }), { status: 202 })
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(submitAnalysisWorkerJob(request)).resolves.toEqual({
      jobId: "job-123",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "https://worker.example.test/v1/analysis-jobs",
      expect.objectContaining({ method: "POST" })
    );
  });
});

describe("worker warmup throttle", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    resetWorkerWarmupCooldown();
  });

  it("starts with an accepted RunPod warmup job", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubEnv("CV_WORKER_TOKEN", "runpod-test-key");
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ id: "warmup-1", status: "IN_QUEUE" }), { status: 200 })
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(submitWorkerWarmup()).resolves.toEqual({
      jobId: "warmup-1",
      accepted: true,
      throttled: false,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.runpod.ai/v2/endpoint-1/run",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("throttles repeated warmup jobs inside the cooldown window", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubEnv("CV_WORKER_TOKEN", "runpod-test-key");
    vi.stubEnv("CV_WORKER_WARMUP_COOLDOWN_MS", "60000");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ id: "warmup-1", status: "IN_QUEUE" }), { status: 200 })
        )
    );

    const first = await submitWorkerWarmup();
    expect(first.accepted).toBe(true);

    const second = await submitWorkerWarmup();
    expect(second.accepted).toBe(false);
    expect(second.throttled).toBe(true);
    expect(second.retryInMs).toBeGreaterThan(0);
  });

  it("force bypasses the throttle so the keep-warm scheduler is never blocked", async () => {
    vi.stubEnv("CV_WORKER_URL", "https://api.runpod.ai/v2/endpoint-1");
    vi.stubEnv("CV_WORKER_TOKEN", "runpod-test-key");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() =>
        new Response(JSON.stringify({ id: "warmup-1", status: "IN_QUEUE" }), { status: 200 })
      )
    );

    await submitWorkerWarmup();
    const forced = await submitWorkerWarmup({ force: true });
    expect(forced.accepted).toBe(true);
    expect(forced.throttled).toBe(false);
  });

  it("exposes the cooldown duration from configuration", () => {
    vi.stubEnv("CV_WORKER_WARMUP_COOLDOWN_MS", "45000");
    expect(workerWarmupCooldownMs()).toBe(45_000);
  });
});
