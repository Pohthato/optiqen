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
| 2b | **Shuttle detector.** TrackNetV3 inference in the worker. It is MIT-licensed, code and checkpoints both, and reports 98.6 % F1 on the Shuttlecock Trajectory Dataset. It sees a run of frames plus the clip's background, so a blurred or tiny shuttle is still found. | `worker/shuttle/tracknet.py`, `worker/detect_shuttle.py` | A detection or a miss for every frame, with a score; runs on CPU for tests and GPU in the worker. On synthetic flight, nearly every in-view shuttle is found within 2.5 px (median). |
| 2b′ | **Shuttle clicks in the labeller (your choice).** A step where you confirm or move the detector's shuttle on frames around each labelled hit. Detections are imported from `detect_shuttle.py`, and the golden schema gains shuttle points. | `tools/labeller/`, `worker/evaluation/` | Detector recall and pixel error on your clips, from your clicks. |
| 2c | **Hits from both.** A hit is an audio onset that the shuttle track agrees with: its direction changes at that moment. The sound's travel time to the phone (15–45 ms on a court) is taken off using the hitter's floor position through the per-frame camera. | `worker/hits.py`, evaluation | F1 at ±33 ms on synthetic clips with neighbouring-court hits; golden clips once labelled. |

## Decisions (2b)

- **Weights.** You approved downloading the published checkpoints (`TrackNetV3_ckpts.zip`, 132 MB, from the authors' Google Drive link). The zip holds only the two `.pt` files. They load with PyTorch's `weights_only` loader: tensors and settings, no code. They live in `worker/weights/`, which git ignores, and the licence is in `THIRD_PARTY_NOTICES.md`.
- **Measuring on real footage.** By your shuttle clicks in the labeller (2b′). The public dataset has no stated licence and is TV broadcast footage, not phone footage.
- **Gaps are filled only between sightings, and only up to 8 frames.** The authors fill everything before the first sighting, which suits clips cut at the serve. On the Hendry clip that invented 88 frames (3 s) of shuttle before the first real sighting.

## Results so far (2b)

- **Synthetic clip (1280×720, 30 fps):** 61 of 61 in-view shuttles found; median error 1.0 px, 90th percentile 2.7 px.
- **Hendry clip (766 frames, fast mode):** 540 frames seen and 59 short gaps filled. The 167 frames with no shuttle are between rallies and in longer losses. The overlay shows clean flight arcs. Accuracy needs your clicks (2b′).
- **Speed:** on this laptop's CPU, about 0.2 s per frame in fast mode, and about 8× that with overlapping windows. The worker's GPU is where it runs for real; the authors report 25 fps.

## Decisions

- **Audio alone is not a hit.** A neighbouring court's hits are as sharp as the rally's, and quieter only by distance. They pass only when the shuttle track agrees.
- **One onset detector.** The labeller (JavaScript) and the worker (Python) must find the same onsets, so suggestions in the labeller match what the analysis will use. The labeller's test suite runs both on the same audio at 48 and 44.1 kHz.
- **Sound travel.** At 343 m/s a hit 10 m from the phone is heard 29 ms late, nearly the whole ±33 ms budget. So hit times are corrected using the camera, not taken from the audio as heard.
