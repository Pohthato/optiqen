# Netoval — Badminton AI Analyzer

## Features
- **Player Detection & Pose**: YOLOv8-Pose badminton action model with confident
  skeleton-only overlays (no persistent person boxes).
- **Court Mapping**: Auto + manual homography with honesty states
  (provisional → validated → image-space-only), BWF singles/doubles dimensions,
  and court coverage in metres.
- **Shuttle & Racket Tracking**: Tiny-object detectors plus Kalman tracking.
  Contact events require a direction reversal or speed collapse plus racket/wrist
  proximity. Shot labels (smash/clear/drop/net/lift/drive/push/serve) are
  confidence-gated heuristics, never ground truth.
- **Badminton Event Layer**: contacts, rallies, split steps, recoveries, racket
  swings, split-step amplitude, racket swing speed, shot landing side.
- **AI Coaching**: DeepSeek API fed only high-confidence, timestamped evidence,
  with explicit instructions about calibration limits and uncertainty.
- **Serverless GPU Worker**: RunPod queue-based endpoint, worker concurrency
  (one GPU runs two analyses), keep-warm scheduler so the first user skips the
  cold start, and an authenticated signed-webhook completion flow.

## Architecture

- `server/` — Express + tRPC API, analysis sessions, storage, worker client.
- `worker/` — Python/RunPod serverless CV worker (YOLO pose/shuttle/racket,
  event layer, metric calculators).
- `client/` — React workspace with calibration canvas and overlay review.
- `docs/` — worker contract, evaluation protocol, RunPod deployment guide,
  analysis architecture.

## Quick Start (local app)
```bash
pnpm install
pnpm dev
```
Set `DEEPSEEK_API_KEY`, `DATABASE_URL`, storage, and `CV_WORKER_*` in `.env`
(see `.env.example`) to enable analysis submission.

## Worker (RunPod)
See `docs/runpod-production.md`. Must be built with licensed weight files in
`worker/weights/`:
`badminton_pose.pt`, `shuttlecock_yolov8n.pt`, and, when available,
`badminton_racket.pt`.

## Environment Variables
Copy `.env.example` to `.env` and set:
- `DEEPSEEK_API_KEY` — DeepSeek API key (coaching chat)
- `DATABASE_URL` — app database
- `CV_WORKER_URL` / `CV_WORKER_TOKEN` — RunPod endpoint root + API key
- `CV_WORKER_KEEP_WARM_ENABLED` — keep one GPU warm (product hours)
- `JWT_SECRET` — long random string that signs sign-in cookies (required)
- `ADMIN_EMAIL` — account that becomes admin when it registers
- `STORAGE_DIR` — where uploaded videos are stored (mount a persistent volume in production)

## Video Limits
- Max duration: 120 seconds (worker limit, configurable)
- Max file size: 1.5 GB
- Guidance: keep the whole court visible, stable rear/side view, 60+ fps

## Evaluation
Capabilities must not launch merely because an interface can draw them — see
`docs/evaluation-protocol.md`. Metrics: keypoint OKS, shuttle recall/precision,
contact timing error vs labelled hits, stroke-label macro F1, court reprojection
error, blinded coach agreement.
