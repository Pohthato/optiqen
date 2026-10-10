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
2. **Fit and merge.** Each piece between candidates is fitted as one flight in 3D. Neighbouring pieces are merged, weakest boundary first, while one flight explains both nearly as well as two. A split must explain 150 units of the detector's noise, or 40 when a sound backs it.
3. **Measured noise.** The noise is the clip's own: the median misfit of a quadratic to every stretch of 7 detections. On 24 benchmark clips it came within 0.87–1.59× of the true jitter (median 0.99×).
4. **Slide.** Each surviving boundary slides, up to 12 detections, to where its two flights fit best.
5. **Six or more detections no flight explains hold a hit.** Scraps of detections are only joined while they are too few to fit.
6. **Long videos.** Stretches with nothing seen for 3 s are read on their own, so a whole match costs the sum of its rallies.

**Timing a hit.** A hit is timed where its two fitted flights are at the same place in the image (the "projected meet"). This uses the flights' drag, and leaves out their depth, which is the poorly seen part of a fit. On the 212 boundaries near true hits in the tuned set (`python -m evaluation.hit_timing`):

| Timing method | Median error | p90 error |
| --- | --- | --- |
| Projected meet (used) | 0.8 ms | 4.1 ms |
| Image quadratics (used when one side is too short for a flight) | 1.9 ms | 8.7 ms |
| 3D closest approach | 5.4 ms | 20.0 ms |
| Sound less its travel from the true contact | 1.3 ms | 2.3 ms |

**Sound.**

- **Attaching sounds.** Sounds are attached to the surviving boundaries, one each, nearest first, after removing each sound's travel to the phone. The travel is measured from the fitted contact point, kept to where a racket can be, so a poorly fitted point cannot make it absurd.
- **The phone's audio/video offset.** One offset is measured for the clip, from the value that at least two hits agree on within 10 ms. When it is over 15 ms, the clip is read again with the offset taken off every sound, so every window that assumes a hit is heard soon after it still holds. Tested at 100 ms (`test_hits.py`): measured to within 8 ms, every hit timed within 33 ms, and the serve keeps its sound.
- **When the sound and the flights disagree.** When they disagree by more than 20 ms, the sound times the hit if it is within 2.5 frames of the flights' time. Further away, the sound is kept only if the flights still fit with the boundary moved to it. Otherwise it is someone else's sound. With no offset measured, a sound more than 60 ms from the flights' time is not the hit's.

**Serves, unseen hits and impossible hits.**

- **Serves.** A serve is a flight after at least 1.5 s with no shuttle, starting below the service law's 1.15 m, heard or not. The check allows 1.6 m at first sighting, with two standard deviations of the fit's height uncertainty.
- **Hits whose incoming flight was unseen.** Such a hit is claimed only with a sound. When traced back to the sound's time, the shuttle must be where a racket can be. The flight before the gap, carried on out of view, must not also explain the shuttle that comes back.
- **No impossible hits.** Hits closer than 150 ms are one hit, the better-explained one kept. A contact point must be somewhere a racket can be.

**The benchmark changed (`worker/evaluation/hits_benchmark.py`).**

- **A balanced grid.** In Phase 2's grid the silent clips were all side view at 60 fps, so "on every view and both frame rates" could not be checked without sound. The grid now crosses every view and frame rate with sound and without. Each set of 60 clips has, per view, 14 clips with sound and 6 silent; per frame rate, 21 and 9. A cell of 6–9 clips moves by several points on one clip.
- **Held-out rallies.** `--first-seed 5000` gives rallies the method was never tuned on.
- **The baseline.** It is the Phase 2 2D finder (`worker/hits.py` at commit 4a86248), rerun on this grid.
- **Stress.** The plain benchmark shares the method's own models: exact cameras, the same drag, the same sound travel, offset 0, equally loud hits and still false detections. `--stress` breaks each of these from a separate random stream:
  - the shuttle's terminal velocity is up to ±10 % off the 6.7 m/s the fit assumes;
  - the audio runs −20 to +80 ms off the video;
  - one hit in five is 12 dB quieter;
  - false detections drift;
  - the method is given miscalibrated cameras (about 0.2°, 3 cm and 1 % focal length for the clip, plus 0.03° frame to frame; about 5 px on the floor).

F1 at ±33 ms over all true hits, including those the camera never saw:

| | 2D baseline, tuned / held-out | **3D, tuned** (seeds 1000–1059) | **3D, held-out** (5000–5059) | Held-out recall of seen hits | Held-out precision | Share of hits seen | **3D, held-out, stressed** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| behind, sound | 0.81 / 0.84 | 0.985 | **0.989** | 0.989 | 0.989 | 1.00 | 0.973 |
| corner, sound | 0.82 / 0.81 | 0.993 | **0.988** | 0.988 | 0.988 | 1.00 | 0.926 |
| side, sound | 0.84 / 0.70 | 0.934 | **0.859** | 0.892 | 1.000 | 0.73 | 0.866 |
| 30 fps, sound | 0.83 / 0.76 | 0.968 | **0.932** | 0.956 | 0.991 | 0.86 | 0.897 |
| 60 fps, sound | 0.82 / 0.82 | 0.970 | **0.966** | 0.969 | 0.992 | 0.96 | 0.951 |
| behind, silent | 0.15 / 0.38 | 0.958 | **0.941** | 0.923 | 0.960 | 1.00 | 0.784 |
| corner, silent | 0.51 / 0.47 | 0.957 | **0.955** | 0.914 | 1.000 | 1.00 | 0.800 |
| side, silent | 0.47 / 0.33 | 0.750 | **0.731** | 0.905 | 1.000 | 0.64 | 0.691 |
| 30 fps, silent | 0.49 / 0.45 | 0.891 | **0.889** | 0.952 | 1.000 | 0.84 | 0.792 |
| 60 fps, silent | 0.34 / 0.35 | 0.921 | **0.875** | 0.875 | 0.972 | 0.91 | 0.725 |
| **all with sound** | 0.82 / 0.79 | 0.969 | **0.949** | 0.963 | 0.992 | 0.91 | 0.924 |
| **all silent** | 0.41 / 0.40 | 0.905 | **0.882** | 0.915 | 0.987 | 0.87 | 0.761 |
| **all** | 0.70 / 0.70 | 0.949 | **0.932** | 0.951 | 0.991 | 0.90 | 0.883 |

- **Neighbouring-court sounds taken as hits:**
  - plain: 0 of 225 (tuned) and 1 of 209 (held-out);
  - stressed: 8 of 218 (tuned) and 3 of 200 (held-out);
  - the 2D finder: 17 and 19.
- **Median timing error of matched hits:** 0.9 ms (tuned) and 1.2 ms (held-out); 1.3–1.9 ms stressed.

**Gate.**

- **Met, plain:** with sound overall (0.949 against 0.95, within a clip of it), silent overall (0.882), and neighbouring sounds (0.5 %).
- **Not met on every view and frame rate,** and not under stress. The misses are concentrated as follows:
  - **The side view.** The phone in the side view stands 4.5 m from the near sideline. From there it does not see the near baseline corners, or shots played high near either baseline: 27 % of hits with sound and 36 % silent were never on screen. Those hits are beyond a method that only has this video. On the hits the side view did see, precision is 1.0 and recall 0.89–0.91. This is a capture problem, so calibration should warn when the court's corners are out of view (Phase 5 capture guidance).
  - **Silent clips.** Without a sound, a hit whose incoming flight went unseen cannot be timed; it is placed between the sightings either side.
  - **Serves seen for fewer than 6 detections** before leaving the frame are missed. Six is the fewest a flight is fitted from.
  - **Stress.** When the world differs from the models, precision falls (0.935 with sound, held-out) and silent recall falls further. Measured one factor at a time on the held-out set (plain F1 0.932, precision 0.991):

    | Factor alone | F1 | What it costs |
    | --- | --- | --- |
    | Drifting false detections | 0.898 | precision 0.931: a white object that moves can pass for a short flight |
    | Terminal velocity ±10 % | 0.924 | silent F1 0.882 → 0.835: a flight the model cannot follow is split |
    | Audio offset | 0.921 | with-sound F1 0.949 → 0.933 |
    | Miscalibrated cameras | 0.933 | nothing measurable |
    | Quiet hits | 0.932 | nothing measurable |

    The next robustness steps follow from this: judging whether a short track moves like a shuttle at all, and estimating the drag (terminal velocity) per clip from its well-seen flights. Real shuttles differ: feather against nylon, speed grades and temperature.

**Known gaps, carried forward.**

- **Shared contact points.** The plan had neighbouring flights share their contact point; they do not yet. The projected meet already times hits to about 1 ms, so a joint fit would mainly tighten contact positions and landings; it moves to 3c.
- **The last flight ends on the floor.** This also moves to 3c, as part of landing.
- **One weakness seen in the tests.** Without sound, a near-court smash seen from behind was split once more, 174 ms into its flight, and the block that followed was missed.

**Speed.**

- **The simulator.** The flight simulator steps coordinates as (3, B) rows, 1.4–2× faster than before with results identical to the previous version (within 1e-14). Against an exact 1 ms RK4 it is within 0.5 mm up to 60 m/s and within 1 cm up to smash speeds (`test_flight_fit.py`).
- **A rally.** An 8 s rally at 30 fps takes about 13–15 s on one core of this laptop, and slower when it is busy.
- **A whole match.** Stretches with nothing seen for 3 s are read on their own, so a match costs the sum of its rallies. Running them in parallel processes is for the pipeline step (3d).
