# Coaching from Findings (Phase 8) — Plan

**Goal:** The app's coaching says only what the analysis found, and says how sure it is. Every sentence the coach writes names the finding behind it, and that finding names the frames it was seen in. A sentence that does not is not shown. Beside the coaching, the rally is drawn on the court: every flight, where each shot came down and how sure that is, and where both players stood.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`, section 2 ("An LLM may only phrase findings that carry a finding ID and evidence frames; a validator rejects any sentence without one") and Phase 8 in section 7. The spec's gate is a blind review by a coach with no harmful advice.

## What is already there

- **The whole analysis, from the command line only.** `worker/analyse_rally.py` takes a video and court points on one frame. It gives the camera in every frame, the hits, each flight in 3D, both players' bodies, and every shot's facts. Those are its name with its chance, speed, net clearance and landing with its call, the pressure on the receiver, and each player's chance of winning from that hit. It also gives the rally's end and winner, how it was won and where it turned. Every number carries its uncertainty, and a call too close to make is said to be.
- **The app does not run it.** An app job (`server/analysisWorker.ts` to RunPod) runs `worker/handler.py`, which still uses the pixel heuristics of `worker/badminton.py`: shot labels from pixel speed and no 3D. Your own changes to `handler.py` and the `Dockerfile` (the camera summary from the corner taps, refusing a job whose models are missing, `geometry/` in the image) have been in the working tree, uncommitted, since 2026-10-02.
- **The coach is given metrics, not findings.** `buildCoachingPrompt` (`server/analysis.ts`) asks DeepSeek to tie each recommendation to a metric, shot or timestamped event, but nothing checks the answer. The reply goes to the page as written.
- **The corner taps already solve a camera.** `worker/geometry/calibration_adapter.py` turns the four corner taps into a camera and a capture tier (A, B or C, spec section 5).

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 8a | **Findings.** `analyse_rally.py` adds a list of findings to its analysis. A finding has an ID, a kind, the player it is about, the hit or hits it concerns with their times and frames, its numbers with their uncertainty, and its call: made with its chance, or too close to call. The kinds: how the rally ended and who won; how it was won (winner, let in, forced or unforced error); where it turned; each shot's name; a shot's speed, net clearance and landing; a hit the receiver had no time to spare for. A claim about the court keeps the chance or the standard deviation the analysis gave it, which already holds the camera's doubt. Each finding also has one plain sentence the app can show as it is. | `worker/findings.py`, `worker/analyse_rally.py`, `worker/test_findings.py` | Every finding has at least one evidence frame and every number its uncertainty. No call is made that the analysis left too close to call. The five clips' findings are listed in this plan. |
| 8b | **Coaching that cites its findings.** When a result has findings, the coach is given those findings with their IDs and nothing else to make claims from. It must end every sentence with the IDs it rests on, as `[F3]`. A checker reads the reply sentence by sentence and drops any sentence that: names no finding, or one that does not exist; gives a number that none of its findings holds, to its rounding; speaks with certainty about a finding that was too close to call; or diagnoses an injury or gives medical advice. What is dropped is said, and when nothing is left the findings' own sentences are shown. | `server/coaching.ts`, `server/coaching.test.ts`, `server/deepseek.ts`, `server/routers.ts`, `server/analysisSchema.ts` | A written set of bad replies (no ID, unknown ID, invented number, certainty on a close call, injury talk) is dropped in full, and good replies pass in full. A result without findings is coached as today. |
| 8c | **The rally on the court.** A view of the court that can be turned, drawn without a new library: every flight as fitted, each contact, each landing with its 95 % region, the net, and where both players stood at each hit. Choosing a finding picks out its shots and moves the video to its first evidence frame. | `client/src/components/RallyCourt.tsx` and its test | Draws the five clips' analyses, checked in the browser; renders in its test; nothing is drawn that the analysis does not hold. |
| 8d | **Into app jobs.** The worker runs the analysis on a job's video, with the camera from its corner taps, and returns it with its findings as an additive `rally` field. The server and page use it when it is there. | `worker/rally_job.py`, `worker/handler.py`, `worker/Dockerfile`, `worker/test_rally_job.py` | A job on each of the five clips returns the same findings as `analyse_rally.py` does from the same court points. A job whose analysis fails still returns today's result, with the reason. |
| 8e | **The gate: a coach's blind review.** For each of a set of rallies, the coach sees the video and two coaching texts: the app's from its findings, and one from today's prompt. The coach does not know which is which, and marks every sentence right, wrong or harmful. | `docs/design/plans/2026-11-16-coaching-from-findings.md` (the protocol and the sheets) | The gate is met when the coach finds no harmful sentence and prefers the app's text. This needs a coach, which the spec lists as an open item. |

## Decisions

- **Findings are made by the worker, not by the language model.** The model only phrases them. The reasons, calls and chances are the analysis's own, so the coaching can be no surer than the analysis.
- **Calls follow their own chances, not the capture tier.** By the spec's provisional thresholds (a camera at least 12° above the court's centre), your five clips are tier B: the phone stands 1.23 m up at a corner. Yet each call carries its chance from that very camera. On the shots benchmark, which films from behind, the corner and the side, every call made on fresh seeds was right (19 of 19). A tier gate would only throw away calls that are sure.
- **A sentence that fails is dropped, not repaired.** Repairing would put words in the model's mouth that the checker then trusts.
- **The viewer is drawn by hand.** A projection of a few hundred points needs no 3D library, and every line drawn is one the analysis holds.
- **Step 8d builds on your worker changes.** You said on 2026-11-18 to go ahead. They went in first as their own commit (f8f8514), then 8d on top.
- **The analysis runs in its own process.** It takes minutes and holds the pose model, so a time limit can stop it without taking the job down, and a crash in it leaves today's result standing.

## Results

### 8a: findings

`worker/findings.py` makes the findings from an analysis as `analyse_rally.py` writes it, and `analyse_rally.py` now adds them to every analysis. A saved analysis gets them without its video. Each frame's time comes from the shuttle detections, so a phone video whose frame rate wanders still points at the right frame.

On your five clips, from their saved tracks, every finding has its frames:

| Clip | Findings | How it ended (F1) | How it was won (F2) |
| --- | --- | --- | --- |
| 6 | 21 | Too close to call: into the net 90 %, in 10 % | Too close to call: unforced error 90 %, winner 10 % |
| 7 | 22 | Far player's shot 16 into the net (100 %), near player won | Too close to call: unforced error 56 %, forced 44 % |
| 8 | 19 | Too close to call: out 58 %, in 42 % | Too close to call: four ways, none above 29 % |
| 9 | 14 | Too close to call: out 53 %, in 47 % | Too close to call: winner 47 %, unforced error 42 % |
| 10 | 21 | Near player's shot 15 out (100 %), far player won | Unforced error by the near player (100 %) |

- **Each clip has one finding per shot and one per player.** A shot's finding holds its name as a coach says it ("cross smash", or "net or block" when neither is sure alone), its speed, height and clearance over the net, how hard it made the receiver work, the hitter's body at the hit, where both stood and each player's chance of winning from it. Its sentence gives only the numbers that are sure: a speed whose standard deviation is under half of it, a clearance known to 0.25 m.
- **Where the rally turned is too close to call on four clips.** On Clip 10 it did not turn: the winner's last shot left the near player time to spare (100 %).
- **A shot's sentence does not say how the last shot came down.** The end says it once, from the rally's end, which also knows whether the shuttle got over the net.

### 8b: coaching that cites its findings

`server/coaching.ts` gives the coach the findings and checks its reply. `server/deepseek.ts` uses it whenever a result carries findings (`rally.findings`). A result without them is coached as before, and the prompt preview shows whichever prompt is sent.

- **The written bad replies are dropped in full, each with its reason.** They name no finding, or one the analysis did not find; give a number no cited finding holds; state as fact a finding that was too close to call; or talk of injury. A good reply passes in full.
- **A number passes when a cited finding holds it as said:** rounded to the digits given, a chance as a percentage (0.99 as 99 %), a time in milliseconds as seconds, and the shot and frame numbers. A heading, or any line with no cite, is dropped like any other sentence.
- **When nothing passes, the findings' own sentences are shown:** how the rally ended, how it was won, where it turned and the player's own finding, each with its ID. Under the text it says how many sentences were left out.
- **No reply from the live model has been checked yet.** That comes when app jobs carry findings (8d).

### 8c: the rally on the court

`client/src/components/RallyCourt.tsx` draws a rally analysis by hand in SVG. It shows the court and net, every flight as fitted with its shadow on the floor, coloured by hitter, each contact, and each landing's 95 % region coloured by its call. For a chosen shot or end it adds where both players stood, with their 95 % regions. It turns by dragging, or to four set views: above, behind, side and corner.

| Clip | Shots drawn | Landing drawn | Findings listed |
| --- | --- | --- | --- |
| 6 | 15 | the last shot's, too close to call | 21 |
| 7 | 16 | the last shot's, out | 22 |
| 8 | 13 | the last shot's, too close to call | 19 |
| 9 | 8 | the last shot's, too close to call | 14 |
| 10 | 15 | the last shot's, out | 21 |

- **All five clips' analyses draw in the browser, with no errors and no broken lines.** On Clip 10, choosing shot 4's finding picked out its flight and asked the video for 7.30 s, its first evidence frame.
- **It draws only what the analysis holds.** Only the last shot of each rally has a landing, since every shot before it was played on. At Clip 10's shot 4 only the near player's region is drawn, because the analysis has no position for the far player at that hit.
- **It is on no page yet.** The page gets it when app jobs carry the rally analysis (8d).

### 8d: into app jobs

`worker/rally_job.py` turns a job's four corner taps into court points (the singles or doubles corners, placed at the first frame) and runs `analyse_rally.py` on the job's video in its own process, with a time limit. `worker/handler.py` adds what comes back as `rally`, or the reason there is none as `rallyError` and a warning; the rest of the result is as before. The server's schema takes both, the coach works from the findings when there are some (8b), and the page draws the rally on the court under the evidence report (8c), where choosing a finding moves the video to its first evidence frame.

- **A job refuses, with its reason, what the analysis cannot do:** a video longer than 90 s (one rally is wanted), fewer than four corners, a run past 15 minutes, or a failed run, whose last error line is passed on. The job still completes with today's result.
- **The image now carries the analysis and everything it imports.** A test reads the imports of the handler, `rally_job.py` and `analyse_rally.py` and fails when a file they need is not copied. It caught two that were missing. Your Dockerfile copied `worker/best.pt`, which does not exist; the detector is `weights/best.pt`, already copied with the other weights.
- **The page no longer moves taps inwards.** It held every tap 5 % inside the frame. On your clips the far-left corner is 3.3 % from the left edge and the near-left one 95.4 % of the way down, and from the moved taps no camera could be solved on any clip. A tap now goes where it is put.
- **Clip 10 cannot be set up from four corners.** Its near-left corner is below the frame. It needs other court points, which the page cannot take yet.

From four taps where the labelled camera puts the corners (so this checks the job's path, not how well a person taps), against the analyses from the labelled court points:

| Clip | Camera apart (median px) | Shots | How it ended | Same sentence |
| --- | --- | --- | --- | --- |
| 6 | 0.00 | 15 and 15 | the same: too close to call, net 90 % | 18 of 21 |
| 7 | 0.03 | 16 and 17 | net (100 %), near player won; then no end at all | 0 of 22 |
| 8 | 0.07 | 13 and 13 | the same: too close to call, out 58 % | 14 of 19 |
| 9 | 0.08 | 8 and 8 | the same: too close to call, out 53 % | 5 of 14 |

- **The acceptance is not met, and the job is not the cause.** Most sentences that differ do so in the last digit of a number. The rest come from two weaknesses in the analysis itself, which a camera less than a tenth of a pixel away brings out.
- **Clip 7: the far player's last shot is cut short by a hit of their own.** At 16.92 s, with the shuttle 0.8 m short of the net and 0.7 m up, a hit by the same far player is found. The flight is cut there, so it never comes down and the rally gets no end. With the labelled camera that moment was not taken for a hit. A player cannot play their own shot again before it has crossed the net, and the shot-building does not yet use that.
- **Clip 9: some uncertainties jump while the values stay put.** Shot 4's contact agrees to the millimetre, yet its standard deviation went from 0.1 m to 1.4 m, so its sentence lost its speed and clearance. Each fact's uncertainty takes in a refit of the rally at one standard deviation more drag. Under this camera that refit dropped two hand-overs and a reach, so it measured another reading of the rally, not the drag.
- **No reply from the live model has been checked yet.** That needs a job run on RunPod with the new image.

**Both weaknesses are fixed (2026-11-19), and four taps now give the labelled camera's rallies.** Clip 9's refit at more drag now keeps the rally as it was read, and a hit from the same side before the shot before it has crossed the net is dropped when its flight more likely stays on its side (both in the rally's end plan). The same four taps again, against the labelled cameras' analyses with the same code:

| Clip | Shots | How it ended | Same sentence |
| --- | --- | --- | --- |
| 6 | 15 and 15 | the same: too close to call, net 90 % | 18 of 21 |
| 7 | 16 and 16 | the same: net (100 %), near player won | 8 of 22 |
| 8 | 13 and 13 | the same: too close to call, out 58 % and 59 % | 14 of 19 |
| 9 | 8 and 8 | the same: too close to call, out 53 % | 9 of 14 |

- **Every sentence that differs does so only in its numbers,** each by at most a quarter of its stated uncertainty (the largest: Clip 7's shot 14 at 23.8 against 24.7 ± 5.0 m/s). Four taps are not the labelled court points, and a camera a few hundredths of a pixel away moves the last digits. The job's path does not.
