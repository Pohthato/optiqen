# Assisted Labeller (Step 2) — Plan and Record

**Goal:** Make labelling a golden clip several times faster and clearer: guided steps, suggestions from geometry, audio and a pose model, and confirmation by the person labelling.

**Spec:** `docs/design/specs/2026-10-04-perception-program-design.md` (Step 2), option 1 for body keypoints (the selected player at each hit frame).

**Approach note:** this step is mostly interface, so the plan records tasks, interfaces and acceptance checks instead of reproducing every line of code; the shared logic is still written test-first and cross-checked against the Python sources of truth.

## Tasks

| # | Task | Files | Acceptance |
| --- | --- | --- | --- |
| 1 | Golden schema: optional `selectedPlayer` and `poses` (17 COCO joints as `[x, y, visibility]`) | `worker/evaluation/golden.py`, `worker/test_golden.py` | Valid docs pass; bad player, joint count, visibility, out-of-image joint, negative time are reported |
| 2 | Shared logic: court completion (least-squares homography from ≥ 4 floor points, no proposals behind the camera), audio hit onsets, hit-suggestion workflow, rally suggestions, pose storage, detector candidates, player picking, export/import of poses, autosave upgrade | `tools/labeller/labeller-core.js`, `tools/labeller/labeller-core.test.ts` | Court coordinates and COCO order equal the Python modules; every synthetic hit (from `worker/simulation/audio.py`) found within 5 ms with neighbouring-court hits ranked below; exports with poses pass `validate_golden` |
| 3 | Interface: step bar (Court → Hits → Rallies → Body) with progress, timeline, shot buttons, loupe, drag-to-fix points and joints, MoveNet MultiPose pre-fill loaded on demand from jsDelivr, manual joint placement fallback | `tools/labeller/index.html` | Driven in a real browser on `uploads/Hendry_Clip.mp4`: 4 clicks → 28 proposals; 44 audio hits in 1.3 s; next/label/skip flow; rally suggestion and winner; pose pre-filled on the near player, confirmed, exported, validated |
| 4 | Labelling guide | `docs/golden-set.md` | Describes the four steps and the pose format; its JSON example still validates |

## Decisions

- Pose model: MoveNet MultiPose Lightning (TensorFlow.js), because it outputs the 17 COCO joints directly and finds several people; it is loaded only when the Body step is used, and the step falls back to placing joints by hand when the model is unavailable.
- The first inference after loading returns nothing while WebGL compiles, so the detector runs one warm-up pass (found in the browser check).
- Court point names are drawn only for the selected point and the point under the cursor; drawing all 32 at once was unreadable (found in the browser check).
- Suggestions are never trusted on their own: audio hits, proposed court points and pre-filled joints all stay grey or editable until confirmed.
