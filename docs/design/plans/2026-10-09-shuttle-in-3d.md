# The Shuttle in 3D (Phase 3) — Plan

**Goal:** Understand each shot physically: where it was hit from, how fast and at what angle, how high it went, by how much it cleared the net, where it landed and whether that was in. Every number comes with its uncertainty, and the system says "too close to call" rather than guess.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`, L2. Each flight between hits is a ballistic trajectory with quadratic drag (terminal velocity about 6.7 m/s): 6 unknowns against 2N pixel observations, projected through the per-frame camera. The spec's gate is a median landing error under 30 cm.

**Why this also fixes hits.** Phase 2 looked for hits as bends in the *image* path, and on the hard benchmark it scores F1 0.77 (see the Phase 2 record). A shuttle can only fly one way: under gravity and drag. A hit is where one physically possible flight ends and the next begins. Perspective, which made a shuttle flying away from the phone look as if it braked, is part of the model, so it no longer fools the detector. A far-court smash that barely bends on screen is a violent change in 3D.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 3a | **One flight in 3D.** Fit launch point and velocity to a flight's detections through the per-frame cameras. Robust to outliers, started from several depths along the first ray (one camera cannot see depth directly; gravity and drag fix it). It gives a covariance, so every derived number has an uncertainty. | `worker/geometry/flight_fit.py` | Synthetic flights from all three views, 30 and 60 fps, 2 px jitter: median landing error < 30 cm; launch speed and apex height errors reported; the stated uncertainty covers the true value about as often as it claims to. |
| 3b | **A rally as a chain of flights.** Candidate hit times come from sounds, image turns and track starts. The chosen hits are the ones that let physically possible flights explain the detections: a split must pay for itself against the clip's measured noise. Neighbouring flights share their contact point, the last flight ends on the floor, and a hit is timed by its sound less the travel time from the fitted 3D contact point to the phone, less the clip's measured audio/video offset. | `worker/hits.py` (replaced) | The Phase 2 hard benchmark: F1 at ±33 ms ≥ 0.95 with sound (baseline 0.82), ≥ 0.85 without (baseline 0.46), at most 1 % of neighbouring-court sounds taken as hits (baseline 12 %), on every view and both frame rates. |
| 3c | **Shot facts.** Neighbouring flights refitted together so they share their contact point, and the last flight ending on the floor (carried from 3b). Per shot: contact point and height, hitter's half, launch speed and angle, apex, net crossing height and clearance, flight time, landing point, and in, out or too close to call. In/out uses the rule boundaries (lines are in) for singles or doubles, and for serves the service court. | `worker/shots.py` | Synthetic: median landing error < 30 cm; in/out correct on ≥ 98 % of the shots it calls, with the "too close to call" rate reported; net clearance and speed errors reported. |
| 3d | **One command, one result.** Video → camera track → shuttle → hits → flights → shot facts, plus an overlay of the fitted 3D flights and a top-down court diagram. | `worker/analyse_rally.py` | Runs end to end on the Hendry clip; the result validates; real-footage accuracy comes from your labels. |

## Decisions

- **Physics before learning.** Every number traces back to detections and a physical model whose assumptions are written down. A learned model can later propose; physics checks.
- **Uncertainty is part of the answer.** A landing 3 cm from the line with a 20 cm uncertainty is "too close to call", not "in".
- **Synthetic first, real second.** The synthetic generator has exact 3D truth, so it can measure everything. Real footage measures what labels allow: hits and shuttle clicks now, landing clicks next.

## 3a results (one flight in 3D, hit times given)

Synthetic flights from random rallies (`worker/test_flight_fit.py` setup): a steadily held phone from behind, the corner and the side; 30 and 60 fps; 2 px detector jitter; 10 % of frames missed. The error is at the flight's end point: the receive point, or the landing for a rally's last shot.

| View, fps | Flights | End point, median / 90th pct | Apex height, median | Launch speed, median |
| --- | --- | --- | --- | --- |
| behind, 30 | 42 | 8.6 / 47.9 cm | 1.0 cm | 1.13 m/s |
| behind, 60 | 42 | 6.8 / 24.3 cm | 0.7 cm | 0.72 m/s |
| corner, 30 | 42 | 9.4 / 29.1 cm | 0.7 cm | 0.71 m/s |
| corner, 60 | 42 | 8.0 / 15.6 cm | 0.4 cm | 0.47 m/s |
| side, 30 | 27 | 3.9 / 15.9 cm | 0.8 cm | 0.31 m/s |
| side, 60 | 32 | 3.7 / 59.3 cm | 0.7 cm | 0.23 m/s |
| **all** | **227** | **7.2 / 28.7 cm** | | |

**Gate passed:** median under 30 cm. Heights are the best-determined quantity, since gravity fixes them. Depth along the line of sight is the weakest, which is why the 90th percentile from behind the baseline is the largest; 3b's shared contact points between flights will tighten it.

- **Uncertainty:** the fit's covariance is scaled ×1.5 in variance so the stated 95 % region holds the truth about 95 % of the time. Measured on 192 flights: 89.6 % unscaled, 96.4 % scaled. Real footage will need recalibration from labels.
- **Robustness:** start seeds come from the first three sightings, so one stray detection at the start of a flight cannot mislead the depth; the loss is robust.
- **Speed:** about 1.1 s per flight on this laptop's CPU (vectorised RK4 at 20 ms steps, within 2 mm of a fine step). 3b will need it faster.

## 3b results (a rally as a chain of flights)

**How it works (`worker/hits.py`).**

1. **Candidate boundaries.** Detections are linked into tracklets. Candidate boundaries are placed at every sound, every tracklet start and every bend in the image path.
2. **Fit and merge.** Each piece between candidates is fitted as one flight in 3D. Neighbouring pieces are merged, weakest boundary first, while one flight explains both nearly as well as two. A split must explain 150 noise units, or 40 when a sound backs it.
3. **Slide.** Each surviving boundary slides, up to 12 detections, to where its two flights fit best.
4. **Six or more detections no flight explains hold a hit.** Scraps of detections are only joined while they are too few to fit.

**Timing a hit.** A hit is timed where its two fitted flights are at the same place in the image (the "projected meet"). This uses the flights' drag, and leaves out their depth, which is the poorly seen part of a fit. On boundaries near true hits:

| Timing method | Median error | p90 error |
| --- | --- | --- |
| Projected meet | 0.9 ms | 4.7 ms |
| 3D closest approach | 5.7 ms | 20.8 ms |
| Image quadratics | 2.5 ms | 13.3 ms |

**Sound.**

- **Attaching sounds.** Sounds are attached to the surviving boundaries, one each, nearest first, after removing each sound's travel from the fitted contact point to the phone.
- **The phone's audio/video offset.** One offset is measured for the clip, from the value that at least two hits agree on within 10 ms. A test with a 60 ms offset recovers it to within 8 ms.
- **When the sound and the flights disagree.** When they disagree by more than 20 ms, the sound times the hit if it is within 2.5 frames of the flights' time. Further away, the sound is kept only if the flights still fit with the boundary moved to it. Otherwise it is someone else's sound.

**Serves, unseen hits and impossible hits.**

- **Serves.** A serve is a flight after at least 1.5 s with no shuttle, starting below the service law's 1.15 m. The check allows 1.6 m at first sighting, with two standard deviations of the fit's height uncertainty.
- **Hits whose incoming flight was unseen.** Such a hit is claimed only with a sound. When traced back to the sound's time, the shuttle must be where a racket can be. The flight before the gap, carried on out of view, must not also explain the shuttle that comes back.
- **No impossible hits.** Hits closer than 150 ms are one hit, the better-explained one kept. A contact point must be somewhere a racket can be.

**The benchmark changed.** In Phase 2's grid, the silent clips were all side view at 60 fps, so the gate "on every view and both frame rates" could not be checked without sound. The grid now crosses every view and frame rate with sound and without (one clip in three silent). `--first-seed` gives rallies the method was never tuned on. The Phase 2 2D finder, rerun on this grid, is the baseline.

F1 at ±33 ms over all true hits, including those the camera never saw:

| | 2D baseline, tuned / held-out | **3D, tuned** (seeds 1000–1059) | **3D, held-out** (5000–5059) | Held-out recall of seen hits | Held-out precision | Share of hits seen |
| --- | --- | --- | --- | --- | --- | --- |
| behind, sound | 0.81 / 0.84 | 0.985 | **0.995** | 0.989 | 1.000 | 1.00 |
| corner, sound | 0.82 / 0.81 | 0.993 | **0.988** | 0.988 | 0.988 | 1.00 |
| side, sound | 0.84 / 0.70 | 0.934 | **0.859** | 0.892 | 1.000 | 0.73 |
| 30 fps, sound | 0.83 / 0.76 | 0.968 | **0.932** | 0.956 | 0.991 | 0.86 |
| 60 fps, sound | 0.82 / 0.82 | 0.970 | **0.969** | 0.969 | 1.000 | 0.96 |
| behind, silent | 0.15 / 0.38 | 1.000 | **0.851** | 0.769 | 0.952 | 1.00 |
| corner, silent | 0.51 / 0.47 | 0.935 | **0.955** | 0.914 | 1.000 | 1.00 |
| side, silent | 0.47 / 0.33 | 0.750 | **0.755** | 0.905 | 1.000 | 0.64 |
| 30 fps, silent | 0.49 / 0.45 | 0.911 | **0.876** | 0.905 | 1.000 | 0.84 |
| 60 fps, silent | 0.34 / 0.35 | 0.897 | **0.846** | 0.825 | 0.971 | 0.91 |
| **all with sound** | 0.82 / 0.79 | 0.969 | **0.951** | 0.963 | 0.996 | 0.91 |
| **all silent** | 0.41 / 0.40 | 0.904 | **0.862** | 0.866 | 0.986 | 0.87 |
| **all** | 0.70 / 0.70 | 0.949 | **0.929** | 0.938 | 0.994 | 0.90 |

Neighbouring-court sounds taken as hits: 0 of 225 (tuned) and 0 of 209 (held-out), against 17 and 19 for the 2D finder. Median timing error of matched hits: 1.0 ms with sound, 1.6 ms silent.

**Gate: met with sound and silent, and for neighbouring sounds, but not on every view and frame rate.** The misses are concentrated as follows:

- **The side view.** The phone in the side view stands 4.5 m from the near sideline. From there it does not see the near baseline corners, or shots played high near either baseline: 27 % of hits with sound and 36 % silent were never on screen. Those hits are beyond a method that only has this video. On the hits the side view did see, precision is 1.0 and recall 0.89–0.91. This is a capture problem, so calibration should warn when the court's corners are out of view (Phase 5 capture guidance).
- **Silent behind and 60 fps.** Without a sound, a hit whose incoming flight went unseen cannot be timed; it is placed between the sightings either side.
- **Serves seen for fewer than 6 detections** before leaving the frame are missed. Six is the fewest a flight is fitted from.

**Known gaps, carried forward.**

- **Shared contact points.** The plan had neighbouring flights share their contact point; they do not yet. The projected meet already times hits to about 1 ms, so a joint fit would mainly tighten contact positions and landings; it moves to 3c.
- **The last flight ends on the floor.** This also moves to 3c, as part of landing.
- **One weakness seen in the tests.** Without sound, a near-court smash seen from behind was split once more, 174 ms into its flight, and the block that followed was missed.
- **The benchmark keeps the audio/video offset at 0.** It is tested at 60 ms (`test_hits.py`), and with fewer than two agreeing hits the offset falls back to 0, which is what phone recorders aim for.

**Speed.** The flight simulator now steps coordinates as (3, B) rows, which is 1.4–2× faster with identical results. An 8 s rally at 30 fps takes about 8.5 s on one CPU core. Rallies are independent, so a match runs in parallel. A compiled kernel (for example numba) is the next step if this becomes the bottleneck.
