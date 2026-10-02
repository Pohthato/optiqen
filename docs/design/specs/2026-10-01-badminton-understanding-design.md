# Badminton Understanding System — Design

Status: approved in conversation 2026-10-01; this document is the written spec.
Scope of v1: **singles only**; one phone, **any placement, tripod or handheld**.

## 1. Goal

Make Optiqen explain *why* a shot or rally was good or bad and *how* to improve
stroke and shot selection from one or two rallies, with every claim traceable to
measured evidence. No existing AI does this reliably; the product's edge is an
explicit, auditable understanding of the game, validated by a human coach.

## 2. Principle

Models do **perception only**. Understanding is explicit:

```
video ─► camera pose per frame ─► 3D shuttle + 3D player ─► physical features
      ─► badminton knowledge engine ─► structured findings ─► LLM phrases findings
```

An LLM may only phrase findings that carry a finding ID and evidence frames; a
validator rejects any sentence without one. Rejected alternative: an end-to-end
video model (needs a large labelled set, opaque, unfixable).

## 3. Diagnosis of the current system

| Area | Today | Defect |
| --- | --- | --- |
| Court | 4 UI corners (percent coords, all must be inside the frame) → `cv2.getPerspectiveTransform`; "validated" by Canny edge overlap (`worker/handler.py` `calibration_homography`) | Planar only, no camera model, no reprojection error; near corners are often out of frame so the UI cannot express the real situation |
| Shuttle position | Airborne shuttle projected through the floor homography (`worker/badminton.py` `_shot_geometry`) | Biased; landing, depth, net crossing are wrong in 3D |
| Shot labels | Hand-tuned thresholds on pixel speed (`classify_shot`) | Camera-dependent, not a model |
| Contacts | Image-space speed reversal at 30 fps sampling | Missed and misplaced hits |
| Coaching | LLM given metrics | Does not know who won a rally or why |

## 4. Architecture (layers)

**L1 Court and camera.** BWF singles court model with ≥30 named 3D keypoints
(floor intersections, net posts at 1.55 m, net tape 1.524 m). Intrinsics (focal
length, one radial term) are estimated once per clip; extrinsics per frame.
Per-frame pose = court-keypoint detection + line snapping (render the court
through the current pose, align to detected lines) + frame-to-frame background
tracking with players masked, fused by a sliding-window optimiser with a
smooth-handheld-motion prior and re-anchored whenever the court is clearly
visible. Floor error is reported in cm and gates every downstream claim.

**L2 3D shuttle.** TrackNet-style heatmap detector. Each flight segment between
hits is fit as a ballistic trajectory with quadratic drag (terminal velocity
≈ 6.7 m/s): 6 unknowns against 2N pixel observations projected through the
per-frame camera, anchored at the hitter's racket and the floor or next racket.
Output: 3D path with uncertainty → landing point, in/out, net clearance, apex,
speed.

**L3 Player.** Fine-tuned 2D pose (RTMPose/ViTPose class) lifted to 3D with the
calibrated camera (feet on the floor fix scale); centre of mass from segment
mass fractions and its offset from the base of support. Identity: court half +
re-identification + the user's one-time tap; ends swap between games.

**L4 Events and shots.** Contacts fuse audio onset (gate only — adjacent courts
cause false hits), 3D trajectory kinks and wrist swing peaks. Shots are
classified from *physical* features (contact height/position, outbound speed and
elevation, apex, landing, net clearance, flight time, stroke side) so the
classifier is camera-invariant and trainable on small data plus public sets
(e.g. TrackNet, ShuttleSet — licences to be verified).

**L5 Rally reasoning.** Winner and terminal event (out, net, unreturnable,
forced error). Singles score consistency: server's service court follows score
parity and the rally winner serves next, so serve positions give an independent
check on winner calls. Trace backward to the pivot shot where positional
advantage flipped (opponent time available, own recovery, depth/height).
Coach-agreed tactical rules over real geometry first; a learned rally-value
model later.

**L6 Biomechanics.** Contact height vs reach, contact distance in front of the
body, arm extension, trunk rotation, split-step timing vs the opponent's hit,
lunge depth, recovery time, jump height — compared against reference
distributions computed by the same pipeline over pro footage.

**L7 Coaching output.** Each finding: claim, evidence frames, magnitude,
reference range, outcome impact, confidence, drill. With 1–2 rallies report
per-shot mechanics (measurable on one shot), not statistical tendencies; flag
small samples. 3D replay viewer with shuttle path, skeleton, COM, ghost shot.

## 5. Capability tiers (honesty by geometry)

Scored after upload from camera pose and court visibility:

| Tier | View | Claims |
| --- | --- | --- |
| A | Elevated/behind, court mostly visible | Full 3D: landing, in/out, net clearance, biomechanics, rally cause |
| B | Low/side-on, partial court | Shot type, approximate landing with wide uncertainty, movement, technique |
| C | Court barely visible | Pose-only technique, no court claims |

Initial thresholds (provisional, tuned on the golden set): A needs camera
elevation ≥ 12° over court centre and ≥ 85 % of the singles court in frame; B
needs ≥ 5° and ≥ 40 %. An in-app framing guide moves users toward tier A.

## 6. Data and evaluation

Evaluation comes first (`docs/evaluation-protocol.md` already forbids launching
unmeasured capabilities). A golden set of ≈20 hand-labelled singles rallies —
including handheld and varied placements — with labelled court keypoints,
contacts, shot types, landing points and rally winners. A two-phone triangulation
session supplies 3D ground truth for the shuttle fit. User calibration taps and
corrections become training labels. Court-keypoint training uses synthetic
rendered courts with handheld shake/blur/rolling-shutter augmentation.
A human coach is required for labelling and blind review (open item: access to
one).

## 7. Roadmap and exit gates

| Phase | Deliverable | Gate |
| --- | --- | --- |
| 0 | Evaluation harness, golden set, baseline numbers for today's pipeline | Baseline report committed |
| 1a | Static camera calibration solver + capability tier | Median floor error < 10 cm on held-out frames |
| 1b | Handheld per-frame camera tracking | Per-frame error budget set from Phase 0 data |
| 2 | TrackNet shuttle, pose, audio contacts | Contact timing within ±33 ms |
| 3 | Physics 3D shuttle and landing | Median landing error < 30 cm vs triangulated truth |
| 4 | 3D pose, COM, biomechanics | Joint error and jitter within agreed limits |
| 5 | Physical-feature shot classifier | Macro-F1 target set after Phase 0 |
| 6 | Rally outcome, cause tracing, rule tactics | Blind coach agreement |
| 7 | Learned tactical value model | Offline lift over the rules |
| 8 | Evidence-validated LLM coaching + 3D viewer | Blind coach review, no harmful advice |
| 9 | Deploy to the owned domain: auth, privacy, cost | Production checklist |

Each phase gets its own plan before execution. **Plan 1 (this repo, now):**
Phase 0 harness + Phase 1a solver + worker integration —
`docs/design/plans/2026-10-01-foundation-eval-and-calibration.md`.

## 8. Constraints and risks

- Monocular 3D at low fps / long range is uncertain: require ≥ 60 fps for 3D
  claims, publish uncertainty, withhold claims when too wide.
- Camera placement limits what is measurable (side-on is poor for depth) —
  the tier system makes that explicit rather than hidden.
- Rolling shutter and motion blur on handheld fast swings: augment in training,
  flag sub-60 fps clips.
- Audio contacts in shared halls: gate only.
- Existing contracts stay: RunPod worker, job queue, DeepSeek call; the result
  schema gains additive optional fields only.
- Worker pins `numpy==1.26.4`; new code must run on numpy 1.26 and 2.x.
