# Perception Programme — Design (synthetic generator, assisted labeller, Phases 1b–3)

Status: approved in conversation 2026-10-04. Extends
`docs/design/specs/2026-10-01-badminton-understanding-design.md` (§4 L1–L4, §7 Phases 1b, 2, 3).

## Capture assumption (user clarification, binding)

"Handheld" means a **landscape** video held **steadily**: small, smooth camera motion, no
fast pans. v1 tracking is designed for small inter-frame motion; rolling-shutter modelling
is out of scope. Tripod footage is the zero-motion case. Portrait is supported only where
it costs nothing (the single-frame solver already handles it).

## Order

| Step | Deliverable | Needs from the user |
| --- | --- | --- |
| 1 | Synthetic video generator | nothing |
| 2 | Assisted labeller | nothing; then labelling |
| 3 | Phase 1b per-frame camera tracking | a few landscape 60 fps phone clips |
| 4 | Phase 2 shuttle detection + audio contacts | nothing (TrackNet licence checked by us) |
| 5 | Phase 3 physics 3D shuttle | one two-phone recording session for real validation |

Each step gets its own implementation plan; Steps 1–2 come first because every later phase
is tested on synthetic truth, and the labeller unblocks real-footage gates.

## Step 1 — Synthetic video generator

Renders badminton clips with exact ground truth so Phases 1b–3 have automatic accuracy
tests before any real labels exist.

- **Shuttle physics** (production code, reused by Phase 3): projectile with quadratic drag,
  `dv/dt = −g e_z − (g / v_t²)|v|v`, `v_t ≈ 6.7 m/s` (feather); RK4 integration; launch
  velocity solved so a shot lands on a target after a chosen flight time; time at which
  the descending shuttle passes a given height (the receiver's interception).
- **Rally builder**: a list of shots (kind, landing target, flight time, receive height)
  becomes contacts (time, 3D position, hitter, kind) and flights; every flight must clear
  the net (≥ 1.55 m at the net plane) or the builder refuses it. A canned realistic rally
  (serve, clear, drop, lift, smash, block, net) is provided.
- **Camera paths**: tripod (constant) and stable handheld (smooth multi-sine shake bounded by
  ≈0.3° rotation and ≈1.5 cm translation per axis).
- **Renderer**: court floor, 40 mm white lines (drawn centred on the model lines), net with
  tape and posts, player cylinders that occlude lines, a motion-blurred shuttle, wall
  background, sensor noise; painter's order by depth.
- **Audio**: low background noise, a sharp band-limited decaying "thwack" at each contact,
  optional quieter distractor hits from a neighbouring court, optional audio/video offset.
- **Output**: frames, per-frame cameras, per-frame shuttle 3D position (null outside
  flight), contacts, landing, audio, distractor times; written as `video.mp4` (OpenCV mp4v),
  `audio.wav`, `truth.json`. No ffmpeg dependency (none is available locally); muxing audio
  into a browser-playable video is out of scope.

## Step 2 — Assisted labeller (option 1 body keypoints)

Guided steps **Court → Hits → Rallies → Body**, each with one instruction line and only its
own controls; big labelled shot buttons (keys still work); a timeline with hit markers and
rally bands; suggestions grey until confirmed; a zoom loupe for point placement.
Assistance: audio-onset hit suggestions; 4-click court completion (homography from four
corners, all other points proposed for nudging); rally suggestions from hit gaps; body
keypoints for the **selected player at each hit frame** (17 COCO joints) pre-filled by an
in-browser pose model and corrected by dragging. Golden schema gains optional pose fields.
The audio onset algorithm has a Python twin (Phase 2) and both pass the same synthetic tests.

## Step 3 — Phase 1b: per-frame camera tracking

Court initialisation from detected line segments searched against the court model (taps are
an optional hint); shared intrinsics from several good frames; per-frame model-based edge
tracking (render court through the previous pose, snap to edges, solve the small pose
update) with masked background features as support; sliding-window smoothing with a smooth
motion prior; re-detection when confidence drops. Output per frame: camera, error, state
(`tracked` / `reanchored` / `lost`). Gate: synthetic stable-handheld clips median floor error
< 5 cm and recovery after 2 s occlusion; real gate on labelled clips.

## Step 4 — Phase 2: shuttle and hit detection

TrackNetV3-class 3-frame heatmap network (pretrained weights if the licence allows, else
trained on the public shuttle-trajectory dataset), with previous frames warped by the 1b
camera track for handheld input. Audio onsets + automatic audio/video offset + visual
evidence (trajectory direction change or wrist-speed peak); audio alone never confirms a
contact. Gate: contacts within ±33 ms on synthetic clips, then on the golden set.

## Step 5 — Phase 3: physics 3D shuttle

Per flight, fit start position and velocity (6 unknowns) plus one drag constant per clip to
all 2D detections through the per-frame cameras, anchored at the hitter's racket height and
ending at the floor or next contact. Covariance gives a landing ellipse; in/out is reported
only when the ellipse clears the line, otherwise "too close to call". Gate: synthetic 60 fps
median landing error < 30 cm; real validation by two-phone triangulation.

## Constraints

- Worker pins numpy 1.26.4 / OpenCV 4.10 / SciPy 1.13.1; all code runs on those and numpy 2.x.
- `worker/simulation/` is test and tooling code (not copied into the worker image);
  `worker/geometry/shuttle_physics.py` is production code.
- 3D claims are withheld below 60 fps.
