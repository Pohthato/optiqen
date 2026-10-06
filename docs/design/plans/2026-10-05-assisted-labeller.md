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

## Review and fixes (2026-10-05)

An independent review found ways the tool could write wrong answers into the golden set without the person noticing. All critical and important findings were fixed test-first and re-checked in the browser:

- Mirrored or end-swapped corners are refused with an explanation (corner orientation and baseline-length checks); the four corner names are always drawn.
- A detected pose is tied to its frame: stale detections are dropped and a pose cannot be confirmed on another frame.
- Suggested hits are accepted only within 250 ms of the suggestion; a shot key relabels the selected hit instead of adding a duplicate.
- Deleting a hit deletes its pose; poses must sit on a hit and belong to the selected player (Python validator too); switching player re-detects.
- Low-confidence joints start unlabelled; right-click cycles visible, hidden, unlabelled.
- Court fits use one frame of human-placed points; each point records whether it was clicked, proposed or adjusted; proposals follow when a corner moves; evaluation ignores proposed points.
- Suggested rallies stay out of the export until a winner is chosen; start and end can be set from the playhead; winner keys use the rally at the playhead.
- Audio onsets use the 2-9 kHz band, a running noise floor and a sharp-attack rule (quiet hits and hits over noise are found; sustained sounds trigger once). On the test clip, candidates went from 85 to 61.
- Autosaves are restored only onto the same video file; loaded labels are validated and escaped; clicks place points and drags start only after movement; FPS detection stops on any pause.

Deferred minors: select boxes swallowing shortcuts, R active in every step, the AudioContext not closed on decode failure, re-running Find hits resurfacing dismissed hits, MoveNet's 256 px default input (the far player is rarely found), partial manual poses confirmable, the video always muted, blob URLs not revoked, no integrity hashes on CDN scripts.

Open question: detector recall and precision on real footage need hand-labelled clips to measure.
