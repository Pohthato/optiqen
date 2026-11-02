# The Rally's End (Phase 6) — Plan

**Goal:** Say how each rally ended (in, out or into the net) and who won it, each with its chance, and "too close to call" when the measurements cannot tell. Then say why it was won: whether an error was forced, and the shot where the rally turned.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`, L5 (rally reasoning). The spec's gate is blind agreement with a coach. There is no coach yet, so the gate proposed below is for you to agree.

## What is already there

- **The last shot's landing (Phase 3).** A rally's last shot is followed to the floor: where it landed, with its covariance, the chance it cleared the net, and in, out or too close to call (`worker/shots.py`).
- **Where both players stood at every hit (Phase 4)**, with its covariance (`body.standing_at`).
- **Shot names (Phase 5)**, each with its chance.
- **Your rally ends.** The notes in `worker/Clips/hittimes.txt` say how each of the five rallies ended: Clip 9's smash and Clip 8's clear were in, Clip 10's push went out long, and Clip 7's drop and Clip 6's cut went into the net.
- **The golden format** already has rallies with a start, an end and a winner (`worker/evaluation/golden.py`).
- **Simulated rallies** always end with the last shot landing, in court or within 40 cm of a line (`worker/simulation/rally.py`). None ends in the net, and nothing is seen after the shuttle comes down.

## Where it stands on your clips

On all five clips the last shot found is your last hit. What the app makes of it today:

| Clip | Your end | The app |
| --- | --- | --- |
| 9 | smash, in | landed 6 cm past the far baseline: too close to call (0.47 in) |
| 8 | clear, in | landed 4 cm past the far baseline: too close to call (0.42 in) |
| 10 | push, out long | no landing: a sighting 1.1 s after the hit, far off the court where no racket could have been, was taken as a reply |
| 7 | drop, into the net | no landing: the drop never crossed the net, but the next sighting 0.8 s later was taken as a reply |
| 6 | cut, into the net | no landing: the flight reached the net at the tape's height (clearance −0.01 ± 0.14 m), and the shuttle falling down the net was taken as a reply |

The shuttle falling down the net also gave two extra hits each on clips 6 and 7: half of the eight extra hits on your clips.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 6a | **Your rally ends in the evaluation.** Read how each rally ended from its notes ("in", "out", "into the net", next to "end of rally"), and the winner from that. Score an analysis against it: the shot that ended the rally, how it ended, and who won. | `worker/evaluation/hand_labels.py` | Reproduces the table above. |
| 6b | **Simulated rally ends.** The simulator ends a rally in one of three ways: in court, out past a line, or into the net (the shuttle meets the net below the tape, stops and falls on the hitter's side). After the end the shuttle lies where it came down, and in some rallies a player picks it up and sends it back over the net to the server. The hits benchmark's scenarios gain these ends. | `worker/simulation/rally.py`, `worker/evaluation/hits_benchmark.py` | A net end stops at the net and comes down at its foot. The detector sees the shuttle on the floor and the pickup. The hits benchmark's old seeds give the same rallies as before. |
| 6c | **How the rally ended, called.** The rally ends when the shuttle comes down, to the floor or into the net. When it is next seen, it has either gone on with the same flight (it went over), come back over the net (it was played), or neither (it stopped). A sighting is a missed hit only when a racket could have met the shuttle there, on the side it had crossed to, or when, with a flight that surely got over the net still up or just down, the shuttle is seen off that flight or a hit is heard. The end is in, out or net, each with its chance: net is the chance it did not get over, and in and out share the rest by where it landed. One is called when it is 95 % likely, as landings are now; otherwise the end is too close to call, with the two most likely. The winner has its own chance, since out and net both lose the rally for the hitter. Hits after the end are not the rally's. | `worker/shots.py`, `worker/evaluation/rally_end_benchmark.py` | Held-out simulation, standard and stressed: every end and winner called at 95 % is right at least 95 % of the time, for in, out and net alike. How often each is too close to call, by its distance from the line or the tape. |
| 6d | **Into the one command, and your rallies.** The summary gets the end with its chances and the winner. The overlay writes them when the shuttle comes down, and the diagram marks the landing or the net. Then your five clips are run again from their saved camera track, shuttle and poses. | `worker/analyse_rally.py` | Your five ends and winners. The hits table again, with the hits after the end left out. |
| 6e | **Pressure at each hit.** For every shot, how hard the receiver had to work to reply: how far they had to go from where they stood at the hit to the next contact, less their reach; the time the flight gave them; the speed that needed; and how high they met it. Each comes with its uncertainty, from the players' standing covariance and the contact's. | `worker/pressure.py` | Exact on simulated players. The stated uncertainty covers the truth on simulated rallies 95 % of the time. |
| 6f | **Why the rally was won.** Rules for you to agree. A shot that lands in is a winner when the receiver could not reach it in time. An error (out or net) is forced when its hitter was under pressure at that hit, and unforced when not. The rally turned at the first shot after which the loser was under pressure at every hit. Each comes with its chance, or too close to call. | `worker/rally_reasoning.py` | Your labels for the five rallies (below), scored blind: the app's answers are written down before yours are read. |

## What 6f needs from you

`hittimes.txt` says how each rally ended but not why. One line per clip would do: for Clip 10, 7 and 6 (out, net, net), whether the error was forced or unforced, and for every clip the shot where the rally turned, if one did. Five rallies only test the rules on your own judgement; a coach's labels on more rallies are what the spec's gate asks for.

## Results

### 6c: how the rally ended, on held-out simulation

`evaluation.rally_end_benchmark` ends the hits benchmark's rallies in turn in court, within 40 cm of a line, 0.3 to 1.2 m out, or into the net. In half of them the shuttle then lies where it came down and is sent back over the net. There were four rounds. In each, the rules were frozen and scored on seeds never looked at, in both worlds. Then every wrong call was traced and the rules changed. So only round 4 measures the rules as they stand; rounds 1 to 3 measure the rules their traces replaced. Each cell is right of called.

| Round | Seeds | World | In | Out | Net | Winner | Winner too close | Last shot right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6000–6119 | Standard | 15 of 19 | 14 of 14 | 5 of 7 | 43 of 47 | 41 | 75 of 120 |
| 1 | 6000–6119 | Stressed | 15 of 18 | 15 of 16 | 7 of 8 | 42 of 49 | 39 | 70 of 120 |
| 2 | 7000–7119 | Standard | 18 of 22 | 20 of 20 | 4 of 5 | 45 of 49 | 29 | 80 of 120 |
| 2 | 7000–7119 | Stressed | 15 of 17 | 8 of 8 | 4 of 5 | 27 of 31 | 30 | 69 of 120 |
| 3 | 8000–8119 | Standard | 22 of 23 | 25 of 25 | 6 of 6 | 49 of 49 | 29 | 89 of 120 |
| 3 | 8000–8119 | Stressed | 18 of 18 | 14 of 16 | 6 of 6 | 33 of 34 | 26 | 72 of 120 |
| 4 | 9000–9239 | Standard | 39 of 41 | 35 of 35 | 9 of 9 | 80 of 81 | 60 | 158 of 240 |
| 4 | 9000–9239 | Stressed | 26 of 27 | 29 of 31 | 6 of 6 | 60 of 63 | 41 | 136 of 240 |

What each round's traces found:

- **Round 1.** The shuttle sent back after the end was taken for the last shot. The last shot's following forgot a hit heard while it was out of sight. Some hits were heard but never seen: one rally's shuttle was sighted in 133 of 482 frames. Flights seen going on past the net were called net.
- **Round 2.** The rule added for that last cause was itself wrong from behind the court, where a shuttle falling down the net looks like a flight going on past it: four wrong "in" calls. A sighting now counts as past the net only when it lies outside the net's outline in the image. The rest were a lone sent-back flight, a 6 ms "flight" at the floor, a 1.9 s flight that never reached the net, and hits between the last two shots that made no shot.
- **Round 3.** A landing known only to tens of metres read as 99 % out, because the court is a small part of the floor. A landing with a spread over 3 m is no longer called. Two more were left alone: a last hit neither seen nor heard (the winner was still right), and two silent hits missed before a 3.6 s gap.
- **Round 4.** Traced for this record, and nothing was changed, so three causes stand. A missed reply: in one standard rally (behind, 60 fps, silent) only three of seven hits were found, and a shot already played on was followed to the floor. In another, in both worlds (corner, 30 fps), the last reply was heard, but the little seen of its flight fitted moving away from the net, so it made no shot and the shot before it stood. Both were rallies into the net called in, and the second kept its winner only because that player won either way. A landing too sure: both wrong outs in the stressed world came from the side at 60 fps with a miscalibrated camera, on landings 1.5 and 1.9 m from the truth whose stated spread did not cover that (one was 34 cm in and called out at 0.999). The shuttle falling down the net taken for a shot: one stressed rally got the end right and the winner wrong, because the fall was given to the other player.

Round 4 is the measure of the gate. "At least" is the 95 % Wilson lower bound.

| Called | Standard | At least | Stressed | At least |
| --- | --- | --- | --- | --- |
| In | 39 of 41 (95.1 %) | 83.9 % | 26 of 27 (96.3 %) | 81.7 % |
| Out | 35 of 35 (100 %) | 90.1 % | 29 of 31 (93.5 %) | 79.3 % |
| Net | 9 of 9 (100 %) | 70.1 % | 6 of 6 (100 %) | 61.0 % |
| Winner | 80 of 81 (98.8 %) | 93.3 % | 60 of 63 (95.2 %) | 86.9 % |

**The gate is not shown to be met.** The standard world reaches 95 % on every line. The stressed world reaches it on in, net and the winner, and misses by one call on out. No lower bound reaches 95 % in either world, and few could: even a run with no wrong call needs about 73 calls of a kind to show it, and only the winner had that many.

What the honesty costs, 240 rallies in each world:

| True end | World | Rallies | Called right | Too close | Called wrong | No end seen |
| --- | --- | --- | --- | --- | --- | --- |
| In | Standard | 97 | 39 | 30 | 0 | 28 |
| In | Stressed | 98 | 26 | 21 | 2 | 49 |
| Out | Standard | 83 | 35 | 19 | 0 | 29 |
| Out | Stressed | 82 | 29 | 13 | 0 | 40 |
| Net | Standard | 60 | 9 | 7 | 2 | 42 |
| Net | Stressed | 60 | 6 | 6 | 1 | 47 |

And by how far the truth was from the line or the tape. Each cell is right, too close, wrong and no end seen, of the rallies at that distance:

| True end | Distance | Standard | Stressed |
| --- | --- | --- | --- |
| In | under 10 cm | 1, 5, 0, 4 of 10 | 2, 4, 1, 4 of 11 |
| In | 10–30 cm | 7, 12, 0, 4 of 23 | 4, 8, 0, 11 of 23 |
| In | 30 cm–1 m | 11, 9, 0, 12 of 32 | 5, 5, 1, 19 of 30 |
| In | over 1 m | 20, 4, 0, 8 of 32 | 15, 4, 0, 15 of 34 |
| Out | under 10 cm | 0, 7, 0, 5 of 12 | 1, 3, 0, 7 of 11 |
| Out | 10–30 cm | 2, 6, 0, 1 of 9 | 2, 6, 0, 1 of 9 |
| Out | 30 cm–1 m | 29, 5, 0, 19 of 53 | 21, 4, 0, 28 of 53 |
| Out | over 1 m | 4, 1, 0, 4 of 9 | 5, 0, 0, 4 of 9 |
| Net | under 10 cm | 0, 1, 0, 1 of 2 | 0, 1, 0, 1 of 2 |
| Net | 10–30 cm | 3, 1, 1, 18 of 23 | 2, 2, 0, 18 of 22 |
| Net | 30 cm–1 m | 6, 5, 1, 23 of 35 | 4, 3, 1, 28 of 36 |

- **Calls are few.** 85 of 240 rallies get an end called in the standard world, and 64 in the stressed one. Doubt goes towards no landing (see Decisions), and the price shows here: 99 and 136 rallies have no end seen.
- **Net ends cost most.** The last shot was the true one on 20 of 60 net ends, against 44 to 48 of 60 for the other kinds (stressed, 19 against 32 to 46). Only 9 of 60 net ends were called, and 6 of 60 stressed.
- **A lean is worth something, but it is not a call.** On the 56 ends left too close, the likelier end was the true one 43 times, where the stated chances averaged 0.735 (stressed, 29 of 40 at 0.74). The in-or-out leans do not favour either side: 15 of 34 leaned out, and 16 were out (stressed, 12 of 23, and 11).
- **By view:** behind in 20 of 21, out 11 of 11, net 4 of 4; corner in 14 of 15, out 17 of 17, net 2 of 2; side in 5 of 5, out 7 of 7, net 3 of 3. **After the end:** rallies that went on past it got in 21 of 21, out 17 of 17, net 4 of 4; the others in 18 of 20, out 18 of 18, net 5 of 5. Both wrong ends had nothing after them.
- **The side view is the weak one under stress:** in 4 of 4, out 4 of 6, net 2 of 2, winner 10 of 12. The stressed world's behind view is in 11 of 11, out 9 of 9, net 2 of 2, and its corner in 11 of 12, out 16 of 16, net 2 of 2.

### 6d: your rallies

The five clips through the one command, from their saved camera track, shuttle and poses, scored with `evaluation.hand_labels`. The rules were frozen before this run. Clips 6 to 10 had been looked at while they were written, so these five are not held out.

| Clip | Your end | The app | Winner |
| --- | --- | --- | --- |
| 9 | smash, in | too close to call (0.47 in, 0.53 out) | too close to call |
| 8 | clear, in | too close to call (0.42 in, 0.58 out) | too close to call |
| 10 | push, out long | out (1.00), right | far, right |
| 7 | drop, into the net | net (1.00), right | near, right |
| 6 | cut, into the net | too close to call (0.71 in, 0.29 net) | too close to call |

- **The last shot is your last hit on all five.** Two ends are called, both right, and no winner is called wrong: this part of the gate is met.
- **All three leans are the wrong way.** None was called, but if the stated chances were fair, three wrong leans would happen about 6 % of the time (0.47 × 0.42 × 0.29). Clips 8 and 9 landed on the far baseline from the low corner camera, the worst-measured line. Three rallies cannot say more than that, and the leans on simulated rallies show no such thing.

The hits table again, with the hits after the end left out:

| Clip | Labelled | Found | Matched | Extra | Timing error, median / max | Hitter right |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 8 | 9 | 8 | 1 | 20 / 37 ms | 8 of 8 |
| 8 | 14 | 14 | 14 | 0 | 23 / 93 ms | 13 of 13 |
| 10 | 20 | 19 | 18 | 1 | 26 / 207 ms | 14 of 15 |
| 7 | 16 | 16 | 16 | 0 | 19 / 38 ms | 16 of 16 |
| 6 | 16 | 18 | 16 | 2 | 31 / 73 ms | 15 of 15 |
| All | 74 | 76 | 72 | 4 | 19–31 ms (medians) | 66 of 67 |

Precision rises from 90 % to 95 %, and recall stays at 97 %. The four hits gone are the ones on clips 6 and 7 after the shuttle met the net. The four left are flights split in two, as in the players plan.

### 6e: pressure at each hit

`evaluation.pressure_benchmark` fits a simulated player over a whole rally, as the body benchmark does, and measures the pressure of each of the opponent's hits that player answered. The contact is the true one off by 5 cm on each axis, and the flight's time off by 12 ms at each end, and each is said to be. The truth is the same measure on the true body, contact and times. The rallies were 108 (seeds 3000–3002 and 3100–3105, each for the three views, both halves, 30 and 60 fps), with 294 hits answered.

| Fact | Median error | 90th percentile | Within twice the stated uncertainty |
| --- | --- | --- | --- |
| How far they had to go | 3.5 cm | 12.4 cm | 92.9 % (89.3–95.3) |
| The speed that needed | 0.03 m/s | 0.10 m/s | 93.2 % (89.7–95.6) |
| How high they met it | 3.0 cm | 7.2 cm | 98.3 % (96.1–99.3) |

- **Coverage is a little under 95 % for the way to go and the speed** (92.9 % and 93.2 %), but 95 % lies inside both intervals, so 294 hits cannot tell them from it. The error over its stated size has an rms of 0.98 and 0.97, so the size is right on the whole and the misses come from a few large errors. The height covers more than it should (98.3 %, rms 0.87): its uncertainty is a little cautious.
- **By view**, the way to go is within twice its uncertainty on 94 % of hits from behind, 90 % from the corner and 95 % from the side.
- **The arm reach** (half the shoulders, the arm and the racket) is within 1.4 mm of the true body's at the median and 2.8 mm at the 90th percentile, and within twice its uncertainty on all 108 rallies.
- **In the one command** each shot gets its `pressure` (to the receiver's next contact, or to where the shot came down when nothing met it), and each half its `armReachM`. A shot has none when the receiver was not seen at the hit, when it went into the net, or when neither a reply nor a landing followed it.

## Decisions

- **The end comes from the flight.** The players' reactions and the sound could add to it later; the flight is what every camera tier sees.
- **The rally ends when the shuttle comes down.** What follows (the shuttle on the floor, a pickup, someone in white in the background) is not the rally's unless a racket could have met the shuttle there.
- **Lines are in, and a shot that clips the tape and falls in the opponent's court is in.** A fitted flight cannot see the tape touch: such a shot comes out with a clearance near zero and an end too close to call, unless it is seen to land.
- **A missed reply is worse than a false one.** A false reply costs a call: the end reads as not seen. A missed reply makes a wrong call. So doubt goes towards no landing: none is called on a last shot that may be the shuttle sent back, on a flight that does not explain its sightings, or on a landing that could be almost anywhere. The price is how many ends are called at all.
- **The same honesty as Phases 3 and 5.** An end the measurements cannot settle is too close to call. The far baseline from a low corner camera is measured worst, which is where clips 8 and 9 landed.
- **The rules in 6f are written down, not learnt.** Five rallies cannot teach them. A learned value of each position is Phase 7.

## Proposed gate (for you to agree)

- **Held-out simulation**, standard and stressed: ends and winners called at 95 % are right at least 95 % of the time, for each kind of end.
- **Your five rallies:** no end and no winner called wrong.
- **Why it was won:** your blind labels (and a coach's, when there is one) agree with the app on forced or unforced and on the turning shot. The level of agreement is set once there are more than five rallies to measure it on.

## Not in this phase

- **The score.** In singles the server serves from the right when their score is even, and the rally's winner serves next, so serve positions check winner calls. A clip holds one rally, so there is no next serve to check against yet.
- **Faults other than out and net:** the shuttle touching a player, a double hit, service faults.
