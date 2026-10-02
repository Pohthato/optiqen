# Golden set: hand-labelled singles clips

Every capability gate in `docs/evaluation-protocol.md` is measured against these
clips. Build the set before trusting any model change. Target: **about 20
singles rallies**, 10–60 s each, **≥ 60 fps**, covering:

- tripod and **handheld** footage, portrait and landscape;
- behind-baseline elevated, low, side-on and corner placements;
- at least three phones, two venues, different lighting and floor colours;
- beginner, club and strong players; rallies of different lengths.

## Files

```
golden/<clipId>.json     hand labels (this schema)
results/<clipId>.json    the worker's job output for the same clip
```

Keep the source videos outside git (they are large and private); the labels and
results are small and belong in the repo or the team drive.

## Labelling a clip

1. **Court keypoints** — on one clear frame, click as many named court
   intersections as are visible (`back0_sl` = near baseline × left singles
   sideline; rows `back0 long0 short0 short1 long1 back1`, columns
   `dl sl c sr dr`; net: `post_left_base post_left_top post_right_base
   post_right_top net_centre_top`). Pixel coordinates in the original video
   resolution. More points give a tighter calibration check.
2. **Contacts** — the millisecond timestamp of every racket–shuttle contact,
   stepped frame by frame.
3. **Shots** — for each contact: the stroke label (`serve clear drop net lift
   drive push smash block other`) and, when visible, where the shuttle landed in
   court metres (`x` across from the left singles line, `y` from the near
   baseline) or `null`.
4. **Rallies** — start/end time and the winner as seen from the camera
   (`near`, `far`, or `unknown`).

## Format

```json
{
  "schemaVersion": 1,
  "clipId": "clip-001",
  "sourceFps": 60,
  "imageSize": [1920, 1080],
  "courtKeypoints": [
    { "name": "back1_sl", "x": 612.0, "y": 301.5 },
    { "name": "back1_sr", "x": 1310.0, "y": 298.0 }
  ],
  "contacts": [{ "timeMs": 1200 }, { "timeMs": 2100 }],
  "shots": [
    { "timeMs": 1200, "label": "serve", "landing": { "x": 2.4, "y": 9.8 } },
    { "timeMs": 2100, "label": "clear", "landing": null }
  ],
  "rallies": [{ "startMs": 800, "endMs": 4300, "winner": "near" }]
}
```

`worker/test_golden.py` validates this example, so it cannot drift from the schema.

## Running the baseline

1. Run each clip through the current worker (local `python worker/handler.py`
   test job or RunPod) and save the job output as `results/<clipId>.json`.
2. From `worker/`:

```bash
python -m evaluation.evaluate --golden ../golden --results ../results --out ../baseline-report.json
```

The report gives, per clip and in aggregate: contact precision/recall/F1 at ±33
and ±100 ms, shot macro-F1 (missed and unverified predictions count as wrong),
and calibration tier, pixel RMS and leave-one-out floor error. Commit the
baseline report before changing any model so every later phase can show lift.
