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

## Shuttle clicks (2b′)

- The labeller has a fifth step, **Shuttle**: `N` goes to the next frame to label (130 ms before each hit, then 70, 200 and 400 ms after), a click places the shuttle, `H` marks it as not visible, and either moves on. The detector's guesses are not shown, so they cannot anchor the clicks.
- The golden schema has optional `shuttlePoints`; the Python and JavaScript validators agree (cross-checked).
- `python -m evaluation.shuttle` scores `detect_shuttle.py` output against the clicks: found within the TrackNet tolerance, wrong place, missed, false alarm, and pixel error.
- Checked in the browser on the Hendry clip: one hit gave four frames to label; a click and `H` were both saved and moved on.

## Decisions

- **Audio alone is not a hit.** A neighbouring court's hits are as sharp as the rally's, and quieter only by distance. They pass only when the shuttle track agrees.
- **One onset detector.** The labeller (JavaScript) and the worker (Python) must find the same onsets, so suggestions in the labeller match what the analysis will use. The labeller's test suite runs both on the same audio at 48 and 44.1 kHz.
- **Sound travel.** At 343 m/s a hit 10 m from the phone is heard 29 ms late, nearly the whole ±33 ms budget. So hit times are corrected using the camera, not taken from the audio as heard.

## Hits from sound and flight (2c)

**How it works** (`worker/hits.py`, `worker/find_hits.py`):

- **Turns in the flight.** For every gap between two detections, two smooth curves (five detections either side) are compared with one curve. The *gain* is how much better two curves explain the path, relative to the shuttle's speed. Where they help, the contact is timed where the two paths meet, between frames. The *jump* is how sharply the velocity changes there. A real hit reverses or swings the shuttle (jump ≥ 1); the curvature at the top of a flight, or the slowdown after a smash, does not.
- **Sound leads.** A hit is heard 0–80 ms after it happens: travel to the phone plus any audio/video offset in the file. Each sound picks the best-fitting split in that window with a real turn there (gain ≥ 0.07, jump ≥ 0.8). Two sounds cannot claim one turn: the better fit wins, so a neighbouring court's sound just before a real hit cannot take its run-up.
- **Serves and re-entries.** A sound with no turn but with the shuttle appearing right after it (a serve the detector did not see in the hand) is a hit, timed by the sound less the delay measured on this clip's own turn hits on the same half of the court. A track starting at the frame's edge is the shuttle coming back into view, not a hit.
- **No sound.** Without a sound track, clear turns (gain ≥ 0.11, jump ≥ 1.0) are hits. With a sound track, a hit nobody heard must be clearer still (gain ≥ 0.2, jump ≥ 1.2).
- **Missed frames at the hit.** Where the detector missed two or more frames right at a split, the hit is timed by its sound less the measured delay rather than by extrapolated paths.

**Results** (synthetic rallies at 30 fps: a steadily held phone, the detector modelled as the true position with 1 px noise and 10 % of frames missed, sound delay, 3 neighbouring-court hits per rally):

| | Tuned on (10 rallies) | Unseen (5 rallies) |
| --- | --- | --- |
| F1 at ±33 ms, with sound | 0.978 | 0.985 |
| Median / 90th percentile / worst timing error | 3.0 / 11.7 / 23.8 ms | 2.0 / 12.3 / 16.0 ms |
| Neighbouring-court sounds taken as hits | 0 | 0 |
| F1 without sound (serves excluded) | — | 0.78 |

**Gate passed:** F1 ≥ 0.95 at ±33 ms on synthetic rallies, with neighbouring-court hits. The thresholds were set by looking at true and false cases on the first ten rallies. The unseen rallies were only scored afterwards.

**Real footage:** `find_hits.py` runs end to end on the Hendry clip (9 hits from the flight alone, since this laptop has no ffmpeg; the worker image has it). Real accuracy will come from your labelled hits: `find_hits.py` writes contact events in the worker's result format, so `python -m evaluation.evaluate` scores them against the golden contacts.

**Deferred:** wiring into the worker pipeline (with Phase 3, alongside the per-frame cameras); a wrist-speed cue from pose; frame rates above 30 fps are supported but not yet tuned on.
