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
| 8d | **Into app jobs.** The worker runs the analysis on a job's video, with the camera from its corner taps, and returns it with its findings as an additive `rally` field. The server and page use it when it is there. This builds on your uncommitted worker changes, so it waits until you have committed them or told me to. | `worker/rally_job.py`, `worker/handler.py`, `worker/Dockerfile`, `worker/test_rally_job.py` | A job on each of the five clips returns the same findings as `analyse_rally.py` does from the same court points. A job whose analysis fails still returns today's result, with the reason. |
| 8e | **The gate: a coach's blind review.** For each of a set of rallies, the coach sees the video and two coaching texts: the app's from its findings, and one from today's prompt. The coach does not know which is which, and marks every sentence right, wrong or harmful. | `docs/design/plans/2026-11-16-coaching-from-findings.md` (the protocol and the sheets) | The gate is met when the coach finds no harmful sentence and prefers the app's text. This needs a coach, which the spec lists as an open item. |

## Decisions

- **Findings are made by the worker, not by the language model.** The model only phrases them. The reasons, calls and chances are the analysis's own, so the coaching can be no surer than the analysis.
- **Calls follow their own chances, not the capture tier.** By the spec's provisional thresholds (a camera at least 12° above the court's centre), your five clips are tier B: the phone stands 1.23 m up at a corner. Yet each call carries its chance from that very camera. On the shots benchmark, which films from behind, the corner and the side, every call made on fresh seeds was right (19 of 19). A tier gate would only throw away calls that are sure.
- **A sentence that fails is dropped, not repaired.** Repairing would put words in the model's mouth that the checker then trusts.
- **The viewer is drawn by hand.** A projection of a few hundred points needs no 3D library, and every line drawn is one the analysis holds.
- **Step 8d waits for you.** Your worker changes have stayed uncommitted for six weeks, and I will not commit them for you or build over them without asking.

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
