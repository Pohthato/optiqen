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
| 3c | **Shot facts.** Per shot: contact point and height, hitter's half, launch speed and angle, apex, net crossing height and clearance, flight time, landing point, and in, out or too close to call. In/out uses the rule boundaries (lines are in) for singles or doubles, and for serves the service court. | `worker/shots.py` | Synthetic: median landing error < 30 cm; in/out correct on ≥ 98 % of the shots it calls, with the "too close to call" rate reported; net clearance and speed errors reported. |
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
