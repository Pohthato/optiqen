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

- **Synthetic clip (1280×720, 30 fps):** 61 of 61 in-view shuttles found; median error 1.0 px, 90th percentile 2.7 px. That was a one-off measurement. The tests check a 640×360 clip: median error under 2.5 px, few misses, and false alarms (none with overlapping windows; fast mode shows a faint still blob for a few frames before the serve).
- **Hendry clip (766 frames, fast mode):** 540 frames detected and 59 short gaps filled. The 167 frames with no shuttle are between rallies and in longer losses. The overlay shows clean flight arcs. Accuracy needs your clicks (2b′).
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

**Results, first claimed:** F1 0.985 at ±33 ms on "unseen" rallies. The review (2026-10-09) showed those were the same canned rally with different noise, at 1 px detector jitter and 30 fps, with neighbouring-court sounds never closer than 150 ms to a hit and no false detections. **That claim is withdrawn.**

**Results on the hard benchmark** (`python -m evaluation.hits_benchmark --scenarios 60`):
- random rallies, filmed from behind, the corner and the side;
- 30 and 60 fps;
- 1–3 px detector jitter, 5–20 % missed frames and false detections;
- neighbouring-court sounds at uniformly random times;
- 2 s without play after each rally.

| Scenarios | F1 at ±33 ms | Recall | Precision | Neighbouring-court sounds taken as hits |
| --- | --- | --- | --- | --- |
| All (60) | 0.77 | 0.73 | 0.82 | 23 of 225 |
| With sound (50) | 0.82 | 0.81 | 0.83 | 23 of 193 |
| Side view (20) | 0.66 | 0.52 | 0.88 | 0 of 68 |
| 60 fps (30) | 0.70 | 0.65 | 0.75 | 11 of 105 |
| No sound (10) | 0.46 | 0.33 | 0.75 | — |

Misses by shot: serve 25, drop 17, clear 14, smash 13, lift 11, block 6, net 1. Found hits are timed well (median 3.9 ms, 90th percentile 14.6 ms).

**Gate not met.** The cause is the method. Bends in the *image* path are not the game: perspective makes a shuttle flying away from the phone look as if it brakes, a far-court smash barely bends on screen, and the turn score is relative to speed rather than to the detector's noise. Phase 3 replaces it. It fits each flight as a 3D shuttle under gravity and drag through the per-frame cameras, and puts hits where one physically consistent flight ends and the next begins, timed by the sound. It must beat this baseline on the same benchmark.

**Review fixes made now (2026-10-09):**
- **Timestamps:** hits are timed from each frame's timestamp, so variable frame rate and missing fps metadata are handled.
- **Onset spacing:** onsets for hit finding are kept 40 ms apart (the labeller keeps 120 ms), so a neighbouring court's sound can no longer hide a hit that follows it closely.
- **Serves:** a start must open a moving, smooth flight of six detections, so a lone false detection or a still object cannot become a serve. Each start is claimed by one sound, the one that fits the clip's measured delay.
- **Filled gaps:** these are not used as sightings for hits.
- **First-sighting points:** a serve's reported point is labelled as the first sighting.
- **Detector output:** it reports detected and filled frames separately.
- **Background memory:** frames are shrunk as they are sampled, so 4K video no longer holds about 3 GB.
- **Tests:** these now count false alarms and run the default overlapping mode.

**Real footage:** `find_hits.py` runs end to end on the Hendry clip (9 hits from the flight alone, since this laptop has no ffmpeg; the worker image has it). Real accuracy will come from your labelled hits: `find_hits.py` writes contact events in the worker's result format, so `python -m evaluation.evaluate` scores them against the golden contacts.

**Deferred:** wiring into the worker pipeline (with Phase 3, alongside the per-frame cameras); a wrist-speed cue from pose; frame rates above 30 fps are supported but not yet tuned on.
