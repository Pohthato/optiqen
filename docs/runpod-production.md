# RunPod production configuration

Netoval uses a queue-based endpoint (`/run`) because a video analysis can
take longer than an HTTP request. The web application sends a short-lived
read-only source URL, never storage credentials. RunPod returns the job ID;
the app polls `/status/{jobId}` and also registers a signed completion webhook.
The webhook does not trust its body: it re-reads the RunPod job with the server
credential and validates the result before it is persisted.

When a player selects a video, the app fires a **warmup job** (`input.warmup =
true`). That job only loads models. Calibration then happens on an already
waking GPU.

## Before deploying

1. Put evaluated private weight files in `worker/weights/` before building:
   `badminton_pose.pt`, `shuttlecock_yolov8n.pt`, and, when available,
   `badminton_racket.pt`. The worker refuses to pretend an unavailable layer was
   detected.
2. Build and push from the repository root:

   ```bash
  docker build -f worker/Dockerfile -t pohthato/netoval-worker:2.0.0 .
  docker push pohthato/netoval-worker:2.0.0
   ```

3. Create a **queue-based** RunPod Serverless endpoint using that immutable
   image. Attach **one** 24 GB GPU (L4 / A5000 / 3090-class) per worker. Do not
   give a single analysis job three GPUs.
4. Set the following application secrets. Do not place any in the browser or
   source tree.

   ```text
   CV_WORKER_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
   CV_WORKER_TOKEN=RUNPOD_API_KEY
  PUBLIC_APP_URL=https://your-netoval-domain
   WORKER_CALLBACK_SECRET=a-long-random-secret
   CV_WORKER_EXECUTION_TIMEOUT_MS=900000
   CV_WORKER_TTL_MS=3600000
   ```

5. Apply database migration `0001_worker_observability.sql` before deploying
   the application that includes this code.

## Warm pool plus burst (one GPU ready, many users still cheap)

RunPod workers process **one video at a time by default**. The Netoval worker
now runs **two analysis processes per worker** (`HANDLER_CONCURRENCY=2`), so one
warm GPU can serve two concurrent video analyses. Concurrent users beyond that
are handled by a short queue on the warm workers, then extra workers only if the
queue would wait.

Recommended endpoint settings:

| Setting | Launch value | Why |
| --- | --- | --- |
| Active (min) workers | **1** during product hours, **0** overnight | First analysis skips cold start |
| Max workers | **2–3** | Caps spend when several people upload at once |
| GPUs per worker | **1** | Extra GPUs only slow boot |
| Idle timeout | **10–15 minutes** | Covers calibration + a second upload |
| FlashBoot | **On** | Extra workers boot in seconds, not minutes |
| Model cache / baked weights | **On** | First job must not download `.pt` files |
| Worker concurrency | **2** (in image) | One GPU absorbs two simultaneous analyses |

Cost behaviour:

- One person: only the warm GPU runs.
- Two or three people at the same time: job 2 runs in the second process on the
  same warm GPU (concurrency), job 3 waits a few seconds or FlashBoot starts a
  burst worker.
- After idle timeout, burst workers stop. The min worker stays up only if
  min workers is 1.

Do not set min workers to 3 “for speed”. That triples idle GPU cost. Keep one
warm, let max workers + intra-worker concurrency absorb spikes, and keep clips
to one rally (the worker limit is 120 seconds) so the warm GPU turns around
quickly. Increase `HANDLER_CONCURRENCY` (2→3) only after measuring VRAM headroom
on your chosen 24 GB card; do not exceed 3 processes per 24 GB GPU.

## Keep-warm scheduler (the "always-warm GPU without paying for idle-farm")

RunPod charges while a worker is booted, but a cold start takes minutes. The app
ships a **keep-warm scheduler** (`server/warmupScheduler.ts`) that POSTs a tiny
`input.warmup = true` job (model load only, ~seconds) every 12 seconds by
default. Because the interval is far shorter than the 10–15 minute idle timeout,
the warm worker **never times out**, so:

- The first user of the hour lands on a GPU whose models are already loaded.
- Real analysis starts within ~2 seconds instead of 5–15 minutes.
- Idle cost is exactly **one** min-worker GPU, not a farm.

Controls:

- `CV_WORKER_KEEP_WARM_ENABLED=false` → disables the scheduler (e.g. overnight
  or on very low traffic) and lets the worker idle out to $0.
- `CV_WORKER_KEEP_WARM_INTERVAL_MS` → how often the keep-alive fires (default
  12000 ms).
- The client also fires a predictive warmup when a signed-in user selects a
  video, throttled to one accepted job per `CV_WORKER_WARMUP_COOLDOWN_MS`
  (default 60 s) so repeated selections never spam the endpoint.

Keep the interval comfortably below the endpoint idle timeout (12 s vs
10–15 min is fine); each keep-alive costs one cheap job on an already-loaded
worker.

Enable FlashBoot. Prefer RunPod model caching or bake private weights into the
image; do not download weights inside `handler()`. Keep multiple compatible GPU
types enabled for availability, not because one job needs them.

## Required dashboards and alerts

Track p50/p95 queue delay, cold-start duration, model-load time, execution
seconds per video-minute, failure rate, GPU seconds per completed analysis,
and the fraction of sessions whose pose/shuttle/court quality clears the
coaching threshold. Alert on worker initialization over five minutes, jobs
queued over two minutes in warm mode, completion failures, and a sudden drop in
shuttle confidence.

The worker returns stage timings and model versions in `result.diagnostics`.
Those are customer-visible analysis notes, not a replacement for endpoint
metrics. Keep RunPod's metrics tab as the source of truth for infrastructure
queue/cold-start measurements.
