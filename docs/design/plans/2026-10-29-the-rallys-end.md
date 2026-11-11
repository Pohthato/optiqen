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
| 6f | **Why the rally was won.** Rules for you to agree. Each hit's pressure gets the time it left the receiver to spare: the flight's time less a 250 ms start and the way beyond their reach, less a 0.5 m lean at a stretch, at 3.5 m/s. A shot that lands in is a winner when the receiver had no time to spare, and let in when they had. (Fixed in 7b: let in only as often as professionals leave a shot they could get to.) An error (out or net) is forced when its hitter had less than 250 ms to spare after the shot before, and unforced when not; a serve is never under pressure. The rally turned at the first of the winner's shots from which every one of theirs left the loser under pressure. Each comes with its chance, or too close to call. | `worker/rally_reasoning.py` | Your labels for the five rallies (below), scored blind: the app's answers are written down before yours are read. |

## What 6f needs from you

`hittimes.txt` says how each rally ended but not why. One line per clip in `worker/Clips/reasons.txt`, not in the chat, so they stay unread until the app's answers are pushed: for Clip 10, 7 and 6 (out, net, net), whether the error was forced or unforced, and for every clip the time of the shot where the rally turned, as in `hittimes.txt`, or "no turn". For example: `Clip 10: unforced; turned at 18.2 s`. Five rallies only test the rules on your own judgement; a coach's labels on more rallies are what the spec's gate asks for.

How they are scored was fixed before any label was read. Forced or unforced is scored given the error, as you label it: the app's chance of forced over its chance of either error, called at 95 %. The turn agrees when both say no turn, or when the app's turning shot is within 250 ms of your time (the hits' tolerance). Too close to call is no call, and its lean is reported apart.

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

### 6f: why the rally was won

The rules were committed (`4dd7d66`) before any clip was run with them. The answers below were pushed before any of your labels was read.

**A check on your clips found the pressure wrong, and it was fixed before any label was read.** Every hit a player answered was reached, so the time to spare should not say otherwise. With the rules as first committed it said 13 of the 51 answered hits on your clips could not have been reached, 9 of them at 95 %. The movement was not at fault: from the hit to their own contact the receivers moved at 1.9 m/s at the median after the 250 ms start, and 3.7 m/s at the 90th percentile. The positions were. When the receiver met the shuttle, the contact was further from their hips than their arms reach on 55 % of the 44 hits with them seen at both hits, and by 3 m or more on a tenth, against a stated uncertainty of about 0.3 m. The 6e benchmark could not see this: it takes the true contact off by 5 cm, not the one the shuttle's flight gives.

The fix (`7cba1ef`): a hit's pressure is not known when the reply's contact is out of the receiver's reach at a stretch from where they stood when they met it, or when they would have covered the way faster than 5 m/s, each by more than twice the stated uncertainty. And the time to spare allows for the shoulders being up to 0.5 m ahead of the hips at a stretch. On your clips it sets aside 9 of the 51 answered hits. Of the 42 left, 2 are still called out of reach at 95 %, about the 5 % such a call allows.

`evaluation.pressure_benchmark` again, on held-out seeds 3200–3208 (108 rallies, 336 hits answered, none set aside):

| Fact | Median error | 90th percentile | Within twice the stated uncertainty |
| --- | --- | --- | --- |
| How far they had to go | 3.6 cm | 11.7 cm | 95.5 % (92.8–97.3) |
| The speed that needed | 0.03 m/s | 0.12 m/s | 95.5 % (92.8–97.3) |
| The time to spare | 13 ms | 36 ms | 97.6 % (95.4–98.8) |
| How high they met it | 3.5 cm | 8.0 cm | 96.4 % (93.9–97.9) |

- **Under pressure or not** is called on 331 of the 336 hits, all right (lower bound 98.9 %); 18 were truly under pressure, and all 18 were called so. Of the 5 too close to call, 3 lean the right way.
- **This does not vouch for your clips' contacts.** The benchmark's contact is the true one off by 5 cm; on your clips the contact from the shuttle's flight is the weak part, which the check above sets aside only when it contradicts the body.

Your five rallies, with the rules as pushed:

| Clip | Your end | How it was won | Where it turned |
| --- | --- | --- | --- |
| 9 | smash, in | too close to call (0.53 unforced error, 0.33 let in, 0.14 winner) | too close to call (0.59 no turn) |
| 8 | clear, in | too close to call (0.42 let in; an error at even odds forced or not) | too close to call (0.71 no turn) |
| 10 | push, out long | unforced error (1.00) | no turn (1.00) |
| 7 | drop, into the net | unforced error (1.00) | no turn (1.00) |
| 6 | cut, into the net | too close to call (0.71 let in; an error would be unforced, 1.00) | no turn (0.98) |

- **The rules as first committed** said the same of how each rally was won, with Clip 10 at 0.95 unforced. On the turn they called Clip 8 no turn (1.00), where the last hit's pressure is now set aside, and left Clip 6 too close to call (0.80 no turn). Your labels score both.
- **No rally is called as turning at a shot.** The rules see where the receiver stood at each hit, not which way they were moving, so a player wrong-footed or slow to recover is not seen as under pressure.

**The winner rule failed on professionals, and was fixed before any of your labels was read.** On ShuttleSet's professional rallies (Phase 7, 7b) it called 183 winners let in, and agreed on 106 of the 289 shots it called. Most winners there left the receiver time to get to the shot but not to play it. A shot that came down in is now let in with the chance that the receiver could get there, times the chance a professional left such a shot. The second is learnt by how far inside the lines it came down and whether it was a clear or a lift, a push or another shot. Your five rallies again, from their saved track, shuttle and poses:

| Clip | Your end | How it was won | Where it turned |
| --- | --- | --- | --- |
| 9 | smash, in | too close to call (0.53 unforced error, 0.46 winner, 0.01 let in) | too close to call (0.59 no turn) |
| 8 | clear, in | too close to call (0.22 winner, 0.20 let in; an error at even odds forced or not) | too close to call (0.71 no turn) |
| 10 | push, out long | unforced error (1.00) | no turn (1.00) |
| 7 | drop, into the net | unforced error (1.00) | no turn (1.00) |
| 6 | cut, into the net | too close to call (0.71 winner; an error would be unforced, 1.00) | no turn (0.98) |

- **Only winner and let in changed.** Your labels score forced or unforced and the turn, which are as before. Clip 8's clear came down near the baseline, the kind of clear professionals leave, so it is let in at about even odds if it was in. Clip 9's smash and Clip 6's cut are now winners if they were in.

**Your labels** came on 2026-11-09, after all three sets of answers were pushed: your notes in `hittimes.txt`, written into `worker/Clips/reasons.txt`. Clip 8's let in and the unforced errors on Clips 10 and 6 are in your words. Three labels are read from your notes. Clip 9 is a winner ("sent it cross court away from the player"). Clip 7 was forced ("had to jump up to reach"), and it turned at the near player's push at 15.28 s ("things resettle and the near player pushes"); you confirmed this reading of Clip 7. Your notes mark no other turn. The scoring script was written before any label was read.

Each cell is the app's chance of your answer against the other answer for that kind of end (winner or let in for a shot that was in, forced or unforced for an error). It is called at 0.95.

| Clip | Your label | As first committed | After the pressure fix | After the winner fix (in the app) |
| --- | --- | --- | --- | --- |
| 9 | winner | 0.68, too close to call | 0.31, too close to call | 0.99, right |
| 8 | let in | 1.00, right | 1.00, right | 0.49, too close to call |
| 10 | unforced | 0.95, right | 1.00, right | 1.00, right |
| 7 | forced | 0.00, wrong | 0.00, wrong | 0.00, wrong |
| 6 | unforced (slightly forced) | 1.00, right | 1.00, right | 1.00, right |

- **Forced or unforced:** called on all three errors in every version, right on two. Clip 7 is wrong at full confidence.
- **Winner or let in** moved with the winner fix: it got Clip 9 right and left Clip 8 at even odds, leaning the wrong way.
- **The turn:** your one turn, Clip 7 at 15.28 s, was called no turn in every version, which is wrong.
- **Five rallies set no level of agreement.** Any later fix that these clips prompt is checked on new labelled rallies, not on these.

**Clip 7 was a jump, and the time to spare left jumps out. Fixed after reading the labels.** The far player met the near player's push at 2.87 m, 0.27 m above their standing reach, with their feet 0.29 m off the floor; the frame at 16.10 s shows the jump. The time to spare counted only the way along the floor. From where the far player stood at the push, the contact was within their reach, so the push left them 0.59 s to spare and the error was unforced at 1.00.

The fix: a jump now costs its push-off and its rise. The push-off is 0.2 s on the floor before take-off. Professionals' jump smashes take about 0.22 s, and a quick rebound under 0.2 s (Rambely and Wan Abas, ISBS 2008). The rise goes up to how high the lower foot was off the floor, at a take-off speed anywhere between one that just gets there and a professional's jump smash (its centre of mass rose 0.54 m in the same study). A foot less than 0.15 m off the floor, the shot names' line for a jump, is no jump.

- **Clip 7's push now leaves 0.24 s to spare** (± 0.04 s). The error is too close to call at 0.56 forced. The turn is too close to call too: 0.44 no turn, 0.28 at the push.
- **Three other hits got some chance of a jump** (22 % to 41 %), from feet whose height is uncertain, and so less time to spare with a wider spread. The only answer this moved is how Clip 9 would have been won had its smash been an error: forced 0.20 where it was 0.00. Its turn, which your notes do not mark, is still too close to call (0.38 no turn, was 0.59).
- **Nothing else changed.** The pressure benchmark and the ShuttleSet runs (7b, 7c) see no feet, so they are as before. The value uses only the shots' names. Nothing measures what a jump costs yet.
- **These five rallies prompted the fix, so they do not test it.** The push-off is counted in full even when the player ran to the shot, though part of it is then the last step's.
- **The far player's positions on Clip 7 are off too, and that is not fixed here.** At all seven of their hits the contact is beyond their reach and lean from where the app put their hips: 2.7 m apart at the median, against 1.5 m and 1.8 m on Clips 9 and 6. The camera sees the far half nearly edge on.

**The far player was nearly where the app put them. The shuttle's depth at their hits is what is off.** Two things were checked against the frames of Clip 7.

The first was the ankle. The fit took it to be 0.07 m above the floor, a bare foot's height. MediaPipe's own heel and toe points put it 0.10 m up on these clips (median 0.096 m on the near player, 0.110 m on the far one), which is where an ankle sits in a court shoe. From this low corner camera, 3 cm of height is 0.1 to 0.4 m of depth. So the fit now takes a shod ankle at 0.10 m, and the player tracker uses the same height in place of its own 0.08 m. The simulator's people now wear shoes with 0.03 m soles, so the fit and the synthetic world still agree.

The second was the contact. At the far player's hits the hit times are right to the frame, and in the image the shuttle sits at their racket. The fitted contact lies on the same line of sight but 2 to 3 m nearer the camera. At 14.55 s on Clip 7 it is 10.0 m from the near baseline, while the player's feet are on the back lines at 12.9 m. The player's position is right and the shuttle's depth is wrong. A far flight's depth rests on how the shuttle sags and slows, which this camera sees nearly edge on.

- **The players moved 0.1 to 0.4 m towards the camera.** The far player's median distance from hips to their own contact went from 2.14 m to 2.09 m (Clip 7: 2.73 m to 2.45 m). 15 of 27 far hits are still beyond reach and lean, against 5 of 29 near ones. The near player's median went from 1.26 m to 1.36 m.
- **On your labels:** Clip 7 is still too close to call and now leans unforced (forced 0.46, was 0.56). The turn leans no turn (0.54, was 0.44, with the push at 0.23, was 0.28). No other call changed. The change rests on the heel and toe measurement, not on these five rallies.
- **The benchmarks were not rerun.** Their people now stand 3 cm taller, and the fit expects it.
- **Not fixed here: the far contacts.** The hitter's feet fix their depth far better than the flight does. Holding each contact within reach of its hitter would fix it, but that changes the flight fit, which every later step uses.

**Each contact is now held within reach of its hitter, and a far flight's depth comes from the hitter's feet.** The rally fit is told where each hitter stood at the hit: the body fit's hips over the floor, with their spread. It is also told how far their racket reaches at a stretch: arm and racket, and 0.5 m of lean. A contact beyond that reach costs what a gap at a hand-over costs, and one within it costs nothing. A hitter the contact's line of sight never passes within reach of is someone else, such as the other player. A hitter the flight cannot reach without missing its own sightings is let go, as a wrong hand-over is. Sightings in the first 70 ms after a hit count for less. On the clips' near flights they miss 2.4 times as much as later ones, because the racket and the body surround the shuttle there.

That alone moved few far contacts, and the flight model is why. Of the 19 far flights whose own fit started beyond reach, 7 could start at the hitter at no cost to their sightings after the first 100 ms. The other 12 missed those sightings by 0.3 to 1.7 px more, unless the flight's own time ran a few per cent faster than the clip's. Over all 22 far flights, the clock that let each start at its hitter was 1.06 in the middle, and 7 % from the clip's (root mean square). A shuttle coming at a low camera barely moves in the image, so its depth rests on how fast it seems to fall. The cause is not known. It is not drag (terminal velocities of 5 to 10 m/s were tried), not a rolling shutter (Clip 7's far shuttle crossed only 15 rows of the image in a whole flight), and not repeated frames (Clip 7's frames are evenly timed at 30 fps).

So a flight now gets its own clock when its contact stays more than two standard deviations beyond its hitter's reach with the reach pulling it in, or when its hitter cannot be met any other way. The flight that handed the shuttle over at that contact gets one too. Each clock is free to 7 % either way. Freeing every flight's clock was tried first. On the benchmark it doubled the contact and landing errors and made a wrong call, because where the model is right a free clock only adds slack.

- **On the five clips, every far contact is now within 0.44 m of its hitter's reach.** Before, 15 of 27 were beyond it, 7 of them by 0.9 to 3.6 m. The far player's hips are themselves known to about 0.25 m. Clip 7's far hit at 14.55 s is now 2.1 m from the hitter's hips, where it was 3.3 m. Its depth was stated to 0.19 m and is now stated to 0.83 m. Its speed went from 18 ± 1 to 24 ± 5 m/s.
- **One far shot went.** On Clip 10 at 21.02 s the contact was 3 m behind the far baseline. It is no longer a shot. The rally's end is the same.
- **On your labels, nothing changed.** The calls made on forced or unforced still agree on Clips 10 and 6. Clip 7 is still too close to call, at 0.43 forced (was 0.46). Clip 6's last shot now leans into the net (0.90, was 0.29), which fits your unforced error better, but it is still too close to call.
- **The shots benchmark sees no loss.** It is the stressed world on held-out seeds 13000 to 13059, with each hitter's hips placed as the body fit places them (0.02 m off per metre from the camera). Its flights follow the flight model exactly, so it cannot show the far depth getting better, only that nothing else got worse. The launch speed is the exception. Its median error went from 0.67 to 0.76 m/s, because the first sightings, which carry it, now count for less.

| Held-out, stressed (seeds 13000–13059) | Before | Reach only | Every clock free | Clock on evidence (in the app) |
| --- | --- | --- | --- | --- |
| Contact point, median / p90 | 0.21 / 0.81 m | 0.20 / 0.80 m | 0.46 / 1.26 m | 0.20 / 0.80 m |
| Contact, far hitter, median | not measured | 0.22 m | 0.49 m | 0.22 m |
| Contact inside its 95 % region, near / far | not measured | 91 / 97 % | 95 / 97 % | 91 / 97 % |
| Launch speed, median | 0.67 m/s | 0.76 m/s | 1.25 m/s | 0.76 m/s |
| Landing error, median / p90 | 16 / 37 cm | 15 / 36 cm | 40 / 91 cm | 15 / 34 cm |
| Calls made, right | 14 of 14 | 13 of 13 | 4 of 5 | 12 of 12 |
| Too close to call | 26 of 40 | 26 of 39 | 33 of 38 | 26 of 38 |

- **The near contacts' stated uncertainty is too tight.** Their 95 % region holds the truth 91 % of the time. This is the first time it was measured.
- **The 7 % comes from the same five clips.** Near flights may need a clock too, but nothing shows it, since their hitters' reach is already met. The cause of the far flights' clock is still to be found. A cause found at the camera or the detections would replace the clock.

## Decisions

- **The end comes from the flight.** The players' reactions and the sound could add to it later; the flight is what every camera tier sees.
- **The rally ends when the shuttle comes down.** What follows (the shuttle on the floor, a pickup, someone in white in the background) is not the rally's unless a racket could have met the shuttle there.
- **Lines are in, and a shot that clips the tape and falls in the opponent's court is in.** A fitted flight cannot see the tape touch: such a shot comes out with a clearance near zero and an end too close to call, unless it is seen to land.
- **A missed reply is worse than a false one.** A false reply costs a call: the end reads as not seen. A missed reply makes a wrong call. So doubt goes towards no landing: none is called on a last shot that may be the shuttle sent back, on a flight that does not explain its sightings, or on a landing that could be almost anywhere. The price is how many ends are called at all.
- **The same honesty as Phases 3 and 5.** An end the measurements cannot settle is too close to call. The far baseline from a low corner camera is measured worst, which is where clips 8 and 9 landed.
- **The rules in 6f are written down, not learnt.** Five rallies cannot teach them. A learned value of each position is Phase 7.
- **Time to get to a shot is not enough to call it let in.** Professionals' winners mostly leave the receiver that time, and not the time to play the shot.
- **Measurements that contradict each other say nothing.** A hit whose contact is out of the receiver's reach from where they stood when they met it has no pressure, rather than one from whichever of the two is wrong.
- **The flight model gives way only where it is shown wrong.** A far flight's own clock is freed only when its contact cannot be brought within its hitter's reach. Freed everywhere, it loosened what the model already got right.
- **A jump the player chose costs as much time as one they needed.** The rules cannot tell the two apart. A jump smash off a high lift still has time to spare.

## Proposed gate (for you to agree)

- **Held-out simulation**, standard and stressed: ends and winners called at 95 % are right at least 95 % of the time, for each kind of end.
- **Your five rallies:** no end and no winner called wrong.
- **Why it was won:** your blind labels (and a coach's, when there is one) agree with the app on forced or unforced and on the turning shot. The level of agreement is set once there are more than five rallies to measure it on.

## Not in this phase

- **The score.** In singles the server serves from the right when their score is even, and the rally's winner serves next, so serve positions check winner calls. A clip holds one rally, so there is no next serve to check against yet.
- **Faults other than out and net:** the shuttle touching a player, a double hit, service faults.
