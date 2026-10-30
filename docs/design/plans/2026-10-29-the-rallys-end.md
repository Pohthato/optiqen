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
| 6c | **How the rally ended, called.** The rally ends when the shuttle comes down, to the floor or into the net. When it is next seen, it has either gone on with the same flight (it went over), come back over the net (it was played), or neither (it stopped). A sighting is a missed hit only when a racket could have met the shuttle there, on the side it had crossed to. The end is in, out or net, each with its chance: net is the chance it did not get over, and in and out share the rest by where it landed. One is called when it is 95 % likely, as landings are now; otherwise the end is too close to call, with the two most likely. The winner has its own chance, since out and net both lose the rally for the hitter. Hits after the end are not the rally's. | `worker/shots.py`, `worker/evaluation/rally_end_benchmark.py` | Held-out simulation, standard and stressed: every end and winner called at 95 % is right at least 95 % of the time, for in, out and net alike. How often each is too close to call, by its distance from the line or the tape. |
| 6d | **Into the one command, and your rallies.** The summary gets the end with its chances and the winner. The overlay writes them when the shuttle comes down, and the diagram marks the landing or the net. Then your five clips are run again from their saved camera track, shuttle and poses. | `worker/analyse_rally.py` | Your five ends and winners. The hits table again, with the hits after the end left out. |
| 6e | **Pressure at each hit.** For every shot, how hard the receiver had to work to reply: how far they had to go from where they stood at the hit to the next contact, less their reach; the time the flight gave them; the speed that needed; and how high they met it. Each comes with its uncertainty, from the players' standing covariance and the contact's. | `worker/pressure.py` | Exact on simulated players. The stated uncertainty covers the truth on simulated rallies 95 % of the time. |
| 6f | **Why the rally was won.** Rules for you to agree. A shot that lands in is a winner when the receiver could not reach it in time. An error (out or net) is forced when its hitter was under pressure at that hit, and unforced when not. The rally turned at the first shot after which the loser was under pressure at every hit. Each comes with its chance, or too close to call. | `worker/rally_reasoning.py` | Your labels for the five rallies (below), scored blind: the app's answers are written down before yours are read. |

## What 6f needs from you

`hittimes.txt` says how each rally ended but not why. One line per clip would do: for Clip 10, 7 and 6 (out, net, net), whether the error was forced or unforced, and for every clip the shot where the rally turned, if one did. Five rallies only test the rules on your own judgement; a coach's labels on more rallies are what the spec's gate asks for.

## Decisions

- **The end comes from the flight.** The players' reactions and the sound could add to it later; the flight is what every camera tier sees.
- **The rally ends when the shuttle comes down.** What follows (the shuttle on the floor, a pickup, someone in white in the background) is not the rally's unless a racket could have met the shuttle there.
- **Lines are in, and a shot that clips the tape and falls in the opponent's court is in.** A fitted flight cannot see the tape touch: such a shot comes out with a clearance near zero and an end too close to call, unless it is seen to land.
- **The same honesty as Phases 3 and 5.** An end the measurements cannot settle is too close to call. The far baseline from a low corner camera is measured worst, which is where clips 8 and 9 landed.
- **The rules in 6f are written down, not learnt.** Five rallies cannot teach them. A learned value of each position is Phase 7.

## Proposed gate (for you to agree)

- **Held-out simulation**, standard and stressed: ends and winners called at 95 % are right at least 95 % of the time, for each kind of end.
- **Your five rallies:** no end and no winner called wrong.
- **Why it was won:** your blind labels (and a coach's, when there is one) agree with the app on forced or unforced and on the turning shot. The level of agreement is set once there are more than five rallies to measure it on.

## Not in this phase

- **The score.** In singles the server serves from the right when their score is even, and the rally's winner serves next, so serve positions check winner calls. A clip holds one rally, so there is no next serve to check against yet.
- **Faults other than out and net:** the shuttle touching a player, a double hit, service faults.
