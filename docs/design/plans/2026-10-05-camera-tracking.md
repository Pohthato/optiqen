# Per-Frame Camera Tracking (Phase 1b) — Plan and Record

**Goal:** A camera for every frame of a landscape phone video held steadily (or on a tripod), so floor positions, shuttle heights and landing spots can be measured in every frame, not only on the frame the court was labelled.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md` (L1, Phase 1b): line snapping of the court model, intrinsics once per clip, extrinsics per frame, re-anchoring when the court is clearly visible, floor error reported in cm.

**Capture assumption:** landscape and held steadily: small, smooth hand motion; no whip pans. Rolling shutter is out of scope.

## Tasks

| # | Task | Files | Acceptance |
| --- | --- | --- | --- |
| 1 | Lens occlusion in synthetic clips (a hand or a passer-by over the lens) | `worker/simulation/clip.py`, `worker/test_clip.py` | Covered frames are near-black and flagged in `truth.json` (`occluded`) |
| 2 | Line response and sub-pixel line search | `worker/geometry/lines.py`, `worker/test_lines.py` | Rendered lines respond and open floor does not; a known 2.5 px shift is recovered to < 0.3 px; the nearer of two parallel lines wins; wide lines are centred; areas wider than the search are rejected |
| 3 | Single-frame fit: court lines and net tape locked onto the image | `worker/geometry/tracking.py`, `worker/geometry/court_model.py`, `worker/test_tracking.py` | From a nudged camera, floor error < 2 cm behind, corner and side; focal length within 3 % from a four-point start; a covered lens and a different view are not confident |
| 4 | Video tracking with tracked / lost / re-anchored states | `worker/geometry/tracking.py` | A slow pan is followed without loss; after a 2 s cover the court is found on the first clear frame; no camera while covered |
| 5 | Tools: track a labelled real video; evaluate tracking on synthetic clips; phone views and lens cover in the generator | `worker/track_camera.py`, `worker/evaluation/track.py`, `worker/evaluation/golden.py`, `worker/simulation/clip.py` | Real clip tracked end to end with an overlay video; gate report on three views |
| 6 | This record | `docs/design/plans/2026-10-05-camera-tracking.md` | — |

## How it works

1. **Line response.** A white top-hat of the grey frame (square kernel, about 1/30 of the frame height) turns painted lines into thin bright ridges and leaves floors, walls and lighting gradients near zero.
2. **Model samples.** Every painted line (doubles and singles sidelines, baselines, service lines, centre lines) and the centre of the white net tape are sampled every 25 cm, with their ends left out (where lines cross).
3. **Search.** Each sample is projected through the current camera and the line response is searched along the projected line's normal. The search covers the expected position error plus half the line's width at that distance: a line 3 m from the phone can be 15+ px wide. The line centre is the centroid of the peak above half its height. Of peaks at least half as strong as the best, the nearest wins, so the doubles sideline does not steal the singles sideline's match.
4. **Fit.** The camera is refined so each sample lies on the line found for it, with a soft-L1 loss so players, shoes and clutter cannot drag it. The search runs coarse to fine (10, 5, 3 px at 1280 px wide, scaled to the frame).
5. **Intrinsics.** The focal length and one radial term are refined once, on the first frame where the court is found, then held. The net tape is 1.5 m above the floor, so it pins the focal length, which floor lines alone leave ambiguous for a phone straight behind the court.
6. **Confidence.** A fit is trusted only when it explains most of what should be visible:
   - at least 24 inliers (within 2 px);
   - 60 % of the matches are inliers;
   - inliers cover 35 % of the model samples in view;
   - RMS ≤ 1.5 px;
   - at least four lines, including both along-court and across-court lines.

   Tracked frames must also be within 2° and 25 cm of the previous camera.
7. **States.**
   - `anchored`: the first frame where the court is found.
   - `tracked`: a frame fitted from the previous one.
   - `lost`: no camera, because a wrong camera is worse than none.
   - `reanchored`: the court found again after a loss.
8. **Re-acquisition.** Used for the first frame and after any loss:
   - a wide coarse-to-fine fit from the last good camera;
   - plus fits from the three best whole-court image shifts (the model slid over a blurred line response).

   The fit that explains the most of the court wins.

## Results

**Synthetic gate** (`python -m evaluation.track`, 1280×720 at 60 fps, steadily held phone, lens covered 2–4 s, views behind / corner / side, anchored on singles-court points tapped with 1 px error):

| View | Visible frames with a camera | Median floor error | 90th percentile | Camera while covered | Court found again after cover |
| --- | --- | --- | --- | --- | --- |
| behind | 100 % | 0.07 cm | 0.18 cm | 0 | first clear frame |
| corner | 100 % | 0.09 cm | 0.11 cm | 0 | first clear frame |
| side | 100 % | 0.05 cm | 0.09 cm | 0 | first clear frame |

Gate (median floor error < 5 cm, court found within 0.5 s of the lens clearing): **passed**. Synthetic frames are cleaner than real ones (sharp lines, no compression, no worn paint), so these numbers are a check of the method, not a promise for real footage.

**Real clip smoke test** (`uploads/Hendry_Clip.mp4`, 1920×1080, 30 fps, 766 frames, phone at standing height beside the near half):
- anchored from eight points (both net posts, four floor corners near the right baseline);
- every frame tracked, none lost;
- median line RMS 0.63 px;
- focal length 1264 px, k1 −0.009;
- the overlay video shows the model on every painted line and the net tape throughout.

Tracking costs about 40 ms per 1080p frame; writing the overlay video roughly triples the run time.

Real floor accuracy is **not yet measured**: that needs court points labelled on several frames of real clips (the labeller records them with `timeMs`).

## Decisions

- **Lines, not features.** Court lines are the one structure whose world coordinates are known exactly, so locking onto them gives absolute accuracy every frame without drift. Frame-to-frame background tracking (spec L1) is deferred: on the steadily held clips so far, the previous frame's camera is close enough for the line search.
- **Lost means no camera.** Downstream phases must treat lost frames as unknown. Short gaps can be interpolated later if needed.
- **Found in the real clip.** The first version jumped toward banners and spectators when re-acquiring, and dropped near lines wider than its final search window. Both were fixed test-first, with synthetic reproductions (dense clutter above the far court; a phone close to the baseline).
- **Real-court quirk.** In the test clip the near-half centre line shows as a dark mat seam, not a white line. The robust fit tolerates missing lines; the model is not edited per venue.

## Usage

```
python track_camera.py clip.mp4 --golden clip.json --out clip.track.json --overlay clip.track.mp4
python -m simulation.clip --out runs/side --view side --occlude 2.0 4.0
python -m evaluation.track runs/behind runs/corner runs/side --out track-report.json
```

`clip.json` is a labeller export: the court points placed on one frame anchor the track. The output has the clip's intrinsics and, per frame, the state, `rvec`/`tvec` (world to camera) and line RMS.

## Deferred

- Wiring per-frame cameras into the analysis worker. Phase 3 (3D shuttle) is the first consumer and will take them from `iter_track`.
- Speed: process at 1280 px wide for 4K input.
- Frame-to-frame background flow for fast hand motion.
- Rolling shutter.
- Courts whose lines are yellow on a light floor (the response uses brightness only).
