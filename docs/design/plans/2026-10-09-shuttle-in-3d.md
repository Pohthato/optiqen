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
