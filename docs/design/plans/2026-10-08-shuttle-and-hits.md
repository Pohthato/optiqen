# Shuttle and Hit Detection (Phase 2) — Plan

**Goal:** For every clip, the shuttle's position in each frame and the time of every racket hit. These feed Phase 3, which fits each flight in 3D through the per-frame cameras from Phase 1b.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`.
- **L2:** a TrackNet-style heatmap detector.
- **L4:** contacts fuse the audio onset with trajectory kinks. Audio is a gate only, because the neighbouring court makes hits too.

**Gate:** hits found within ±33 ms (spec), scored as F1.
- **Synthetic clips:** with sound delay and neighbouring-court hits.
- **Golden clips:** as they are labelled.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 2a | **Audio hits in the worker.** A Python twin of the labeller's onset detector (2–9 kHz band, running noise floor, sharp attack). Audio is read from the video with ffmpeg (already in the worker image). | `worker/audio/onsets.py`, `worker/audio/extract.py` | Same onsets as the labeller's JavaScript on the same audio (the labeller's test suite runs both). Every synthetic hit is found within 5 ms of when it is heard. Neighbouring-court hits rank below the rally's. A video without an audio track gives no onsets, not an error. |
| 2b | **Shuttle detector.** TrackNetV3 inference in the worker. It is MIT-licensed, code and checkpoints both, and reports 98.6 % F1 on the Shuttlecock Trajectory Dataset. It sees a run of frames plus the clip's background, so a blurred or tiny shuttle is still found. | `worker/shuttle/` | Detections with confidence per frame; runs on CPU for tests and GPU in the worker. Needs your OK first: download the published checkpoints (`TrackNetV3_ckpts.zip`, Google Drive), and decide how to measure it on real footage. |
| 2c | **Hits from both.** A hit is an audio onset that the shuttle track agrees with: its direction changes at that moment. The sound's travel time to the phone (15–45 ms on a court) is taken off using the hitter's floor position through the per-frame camera. | `worker/hits.py`, evaluation | F1 at ±33 ms on synthetic clips with neighbouring-court hits; golden clips once labelled. |

## Decisions

- **Audio alone is not a hit.** A neighbouring court's hits are as sharp as the rally's, and quieter only by distance. They pass only when the shuttle track agrees.
- **One onset detector.** The labeller (JavaScript) and the worker (Python) must find the same onsets, so suggestions in the labeller match what the analysis will use. The labeller's test suite runs both on the same audio at 48 and 44.1 kHz.
- **Sound travel.** At 343 m/s a hit 10 m from the phone is heard 29 ms late, nearly the whole ±33 ms budget. So hit times are corrected using the camera, not taken from the audio as heard.
