# Netoval Computer-Vision Worker Contract

The web application accepts an uploaded source video, persists a calibrated session, and submits a worker request to `POST /v1/analysis-jobs`. The GPU worker is the only component permitted to claim that an observation was detected in a frame. It returns results through `GET /v1/analysis-jobs/{jobId}` or an authenticated callback that invokes the app’s result-acceptance procedure.

## Job request

```json
{
  "analysisId": "private-session-id",
  "videoStorageKey": "analysis-sources/{user}/{video}",
  "selectedPlayer": "near",
  "calibration": {
    "corners": [{ "label": "nearLeft", "x": 12.5, "y": 86.2 }],
    "confidence": "provisional",
    "supportsCourtMapping": false,
    "guidance": "..."
  },
  "requestedLayers": ["skeleton", "shuttle", "racket", "courtMap"]
}
```

## Job status and result

The worker returns `queued`, `processing`, `completed`, or `failed`. A completed response must include a processing version, quality metrics, calibration provenance, shot distribution, metric summaries, and either a fully rendered annotated-video artifact or normalized overlay packets. Overlay coordinates use percentage video coordinates from 0 to 100. Every point includes a confidence score, which allows the web client to draw only qualified evidence rather than interpolate missing detections.

| Stage | Required worker output | Product behavior |
| --- | --- | --- |
| Court registration | Labelled lines/keypoints, homography, reprojection error, camera-pose status | Court-plane metrics require accepted registration; partial calibration cannot be shown as exact metres. |
| Pose inference | Temporal 17-keypoint skeletons, per-keypoint confidence, player identity track | Only skeleton strokes are rendered; persistent person boxes are prohibited in the review layer. |
| Shuttle and racket tracking | Detector output plus temporal filter state and confidence | Gaps, occlusions, and uncertainty remain visible to the user rather than being silently filled. |
| Events | `contact`, `split_step`, `recovery_complete`, `racket_swing` with frame, timeMs, confidence, source | Coaching prompts receive confidence-gated events and source timestamps only. |
| Biomechanics | Frame evidence for every aggregate metric: base width, knee flexion angle, split-step timing, recovery, court coverage, split-step amplitude, racket swing speed | Split-step amplitude and racket swing speed are image-relative (normalised by frame diagonal), not calibrated physical speeds. |
| Shots | Verified shot only when contact plus a post-impact track clears the gate; each shot carries direction, depth, `landingSide` (`own_half`/`opponent_half`/`unclear`) | Landing side is derived from the observed post-impact shuttle cloud beyond the net centre-line, never an interpolated bounce. |
| Artifact rendering | Annotated MP4 and/or normalized overlays | The review canvas switches to verified overlays only when the worker returns a completed result. |

## Deployment boundary

The worker requires CUDA-capable infrastructure, a queue, object-store access, and enough memory for decoding and batch inference. The current web application intentionally does not emulate those components in the browser or on the managed web server. A complete worker deployment should use a GPU-capable runtime for pose, court-line/keypoint, shuttle, and racket models; use an asynchronous job queue for multi-minute videos; and send the web application a versioned result conforming to this contract.

The court plane alone supports player-foot and movement coverage mapping. It does not establish true shuttle height or full 3D joint depth from a single view. The worker must label camera-pose confidence and restrict 3D claims to clips with sufficient camera calibration evidence.
