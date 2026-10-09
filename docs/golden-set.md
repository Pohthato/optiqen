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

## The labeller

`tools/labeller/index.html` is a zero-install browser tool that writes this schema
directly — open the file in Chrome or Edge (no server needed; the Body step loads
its pose model from the internet the first time). Open the clip and check the
detected FPS against the phone's setting (keep the tab in front while it detects;
press **Detect** to retry, or type the value). Then work through the four steps
across the top; each shows what to do next and how far you are.

1. **Court** — pause on a clear frame and click the four corners of the singles
   court (the name to click is pre-selected; `back0` is the baseline nearest the
   camera, `sl`/`sr` the left/right singles sidelines as you look at the court —
   the labeller refuses corners that are mirrored or have the ends swapped). **Propose the rest** draws every
   other line where the geometry says it must be; drag any point that is off and
   **Accept**. A magnifier follows the cursor for precise clicks.
2. **Hits** — **Find hits from audio** lists every racket "thwack". `N` jumps just
   before the next one (sound reaches the phone ~30 ms after the hit); step with
   `,` `.` to the frame where racket meets shuttle and click the shot (or press
   `1`–`9`, `0`). `X` skips a sound that isn't a hit (squeaks, a neighbouring
   court), `C` marks a hit the audio missed. Audio suggestions are a starting
   point, not the answer: watch each rally once at 0.5× and add any quiet hit
   they missed.
3. **Rallies** — **Suggest rallies** groups hits by the pauses between them; check
   each one and choose who won (near / far / ?, as seen from the camera). `R`
   starts or ends a rally by hand.
4. **Body** — choose your player once. `B` goes to the next hit frame and fills in
   that player's skeleton; drag any joint that is off, right-click a joint you
   can't see (hidden), `Enter` to confirm. If nobody is detected, **Place joints
   by hand** asks for each joint in turn (`H` skips one).

**Save labels** downloads `<clipId>.json`; work also autosaves in the browser, and
**Load labels** resumes a saved file. **Load worker result** imports the worker's
detected hits as a starting point. Shuttle landing points are left `null` in this
version.

## Labelling a clip

1. **Court keypoints** — on one clear frame, click as many named court
   intersections as are visible (`back0_sl` = near baseline × left singles
   sideline; rows `back0 long0 short0 short1 long1 back1`, columns
   `dl sl c sr dr`; net: `post_left_base post_left_top post_right_base
   post_right_top net_centre_top`). Pixel coordinates in the original video
   resolution. More points give a tighter calibration check. Each point
   records `timeMs`, the frame it was clicked on (the labeller fills it in);
   for handheld clips click all points on the same frame, since the camera
   moves between frames and the evaluation solves one frame at a time.
2. **Contacts** — the millisecond timestamp of every racket–shuttle contact,
   stepped frame by frame.
3. **Shots** — for each contact: the stroke label (`serve clear drop net lift
   drive push smash block other`) and, when visible, where the shuttle landed in
   court metres (`x` across from the left singles line, `y` from the near
   baseline) or `null`.
4. **Rallies** — start/end time and the winner as seen from the camera
   (`near`, `far`, or `unknown`).
5. **Body poses** — `selectedPlayer` (`near` or `far`) and, for each hit, that
   player's 17 COCO joints (nose, eyes, ears, shoulders, elbows, wrists, hips,
   knees, ankles) as `[x, y, visibility]` in pixels: visibility `2` visible, `1`
   hidden but placed, `0` not labelled. Stored as
   `"poses": [{ "timeMs": 1200, "player": "near", "keypoints": [[x, y, v], …17] }]`.
6. **Shuttle** — on a few frames around each hit (130 ms before it, then 70,
   200 and 400 ms after), where the shuttle's head is, or that it cannot be seen
   on that frame. The labeller does not show the detector's guesses here, because
   they are what is being measured. Clicks land on screen pixels, so a bigger
   window gives more precise points. Stored as `"shuttlePoints": [{ "timeMs":
   1267, "x": 980.5, "y": 412.0 }, { "timeMs": 1300, "visible": false }]`.

## Format

```json
{
  "schemaVersion": 1,
  "clipId": "clip-001",
  "sourceFps": 60,
  "imageSize": [1920, 1080],
  "courtKeypoints": [
    { "name": "back1_sl", "x": 612.0, "y": 301.5, "timeMs": 0 },
    { "name": "back1_sr", "x": 1310.0, "y": 298.0, "timeMs": 0 }
  ],
  "contacts": [{ "timeMs": 1200 }, { "timeMs": 2100 }],
  "shots": [
    { "timeMs": 1200, "label": "serve", "landing": { "x": 2.4, "y": 9.8 } },
    { "timeMs": 2100, "label": "clear", "landing": null }
  ],
  "rallies": [{ "startMs": 800, "endMs": 4300, "winner": "near" }],
  "shuttlePoints": [
    { "timeMs": 1267, "x": 980.5, "y": 412.0 },
    { "timeMs": 1300, "visible": false }
  ]
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
and ±100 ms, and shot macro-F1. Missed and unverified predictions count as wrong,
and a verified shot with no labelled shot near it counts as a false claim. The
aggregate numbers pool the counts over all clips rather than averaging per-clip
scores. Each clip also shows the camera solved from your labelled points
(`calibration`: tier, pixel RMS, leave-one-out floor error) next to the worker's
own camera summary (`workerCamera`). Commit the baseline report before changing
any model so every later phase can show lift.

## Scoring the shuttle detector

From `worker/`, with the TrackNetV3 weights in `worker/weights/`:

```bash
python detect_shuttle.py ../clips/clip-001.mp4 --out ../results/clip-001.shuttle.json
python -m evaluation.shuttle --golden ../golden/clip-001.json --detections ../results/clip-001.shuttle.json
```

A clicked shuttle counts as found when a detection is within the TrackNet tolerance (4 px at the
network's 512 px width: 15 px on a 1920 px video). The report gives recall over the frames you
saw the shuttle on, precision over all detections on labelled frames, the median and 90th
percentile pixel error, false alarms on frames marked "can't see it", and how many finds came
from filled gaps.
