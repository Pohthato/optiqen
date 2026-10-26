# Naming the Shots (Phase 5) — Plan

**Goal:** Name every shot (clear, drop, smash, lift, net, push, drive, block, serve) from what its flight did in 3D, not from pixels. Say how sure the name is, and give two names when the measurements cannot tell them apart. Then add the details a coach names a shot by: cross or straight, forehand or backhand, from a jump.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`, L4 (shots are classified from physical features, so the classifier works for any camera). The spec's gate is a macro-F1 "set after Phase 0"; the target proposed below is for you to agree. `docs/evaluation-protocol.md` forbids heuristic smash/clear/drop labels: the names come from a model fitted to labelled shots and scored on shots it never saw.

## What is already there

- **The shot facts (Phase 3).** Every shot has its contact point, launch speed and angle, apex, net crossing, flight time and path, each with its uncertainty (`worker/shots.py`).
- **The hitter's body (Phase 4).** Most shots have the hitter's facts at contact: height against reach, contact ahead and aside, off the floor (`worker/body.py`).
- **The labels.** The golden set's shot labels are `serve clear drop net lift drive push smash block other` (`worker/evaluation/golden.py`), and `evaluation.metrics.macro_f1` scores them.
- **Simulated shots.** The rally simulator plays the same eight families with known names, each from its own contact heights to its own depth, flight time and receiving height (`worker/simulation/rally.py`). The shots benchmark runs such rallies through the real hit finding and flight fitting (`worker/evaluation/shots_benchmark.py`), so their facts carry real measurement error.
- **Your named shots.** `worker/Clips/hittimes.txt` names all 74 hits on clips 6–10. Matched to the hits the app found, they are the real test.

## Your shot names as labels

Every name maps to one golden label, except three that are open:

| Label | Your names | Shots |
| --- | --- | --- |
| serve | serve | 5 |
| net | net, middle net, net middle, cross net, net from low, stop net, stopping net, chop net, long net | 15 |
| clear | clear, cross clear, punch clear, backhand save clear | 9 |
| lift | lift, high lift, lift cross, cross lift, lift from low, cross lift from low, flat lift | 9 |
| block | block, backhand block, cross block, half dive block | 8 |
| push | push, flat push, push (out long) | 8 |
| drop | drop, backhand drop, cross flat drop, cross flat slice/drop, china jump slice drop, jump out stop drop | 7 |
| smash | smash, cross smash, half smash, jump out smash, scissor body smash, intercepting flat smash | 6 |
| drive | drive, cross drive, backhand drive | 4 |
| push or lift | cross push/lift, lift/push | 2 |
| not scored | backhand cut off cross | 1 |

- **"push or lift"** counts as right when the app says either. Your note says the two are subjective.
- **"long net"** is read as a net shot. If you meant a lift from the net, it moves.
- **"backhand cut off cross"** is left out until you name its family.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 5a | **Your labels in the repo's evaluation.** Read the hittimes format: hits, hitters, serves and names. Score an analysis against it: hits (recall, precision, timing, hitter, as in the players plan) and shot names (macro-F1 and the confusion table). | `worker/evaluation/hand_labels.py` | Reproduces the hits table in the players plan exactly. |
| 5b | **The facts a name rests on, and simulated shots to learn from.** Per shot, the facts below, from the shot facts and the rally around it. Simulated rallies through the real measurement give each fact its real error. Seeds for training and held-out seeds for testing are kept apart, as in every benchmark. | `worker/shot_names.py`, `worker/evaluation/shot_names_benchmark.py` | The facts are exact on true flights. The held-out set is written once and never trained on. |
| 5c | **The model.** Multinomial logistic regression on the facts and their pairwise products, fitted with numpy and scipy (no new package). The weights are a small JSON file in the repo, and the benchmark regenerates them. Each name's probability is averaged over draws of the facts from their uncertainty. The name is given when it is at least 80 % likely; otherwise the two most likely names are given. | `worker/shot_names.py`, `worker/shot_names.json` | Held-out simulation: macro-F1, set beside the macro-F1 the true facts give (how far the families overlap by themselves). Calibrated: names given at 80 % or more are right at least 80 % of the time, for every family, and in the stressed world too. |
| 5d | **Into the one command, and the real test.** Each shot in `analyse_rally.py` gets its name with the probabilities. The overlay writes it next to the speed, and the diagram's legend names each flight. Then the five labelled clips are scored. | `worker/analyse_rally.py` | Real macro-F1 by family and by camera (the corner tripod of clips 6–9, Clip 10). It is reported twice: on the shots whose hit was found, and end to end, where a missed hit counts as a wrong name. |
| 5e | **Cross or straight, forehand or backhand, from a jump.** Cross: the flight crosses the court diagonally. Backhand: the racket hand meets the shuttle across the body, on the side away from the racket arm, beyond the uncertainty. The racket arm is the one the player hits with on most shots of the clip. Jump: both feet are off the floor at contact (4c already measures it). Each is "too close to call" when the measurements cannot tell. | `worker/shot_names.py` | Simulation: each one right within its stated uncertainty. Real: your labels mark 17 cross shots, 7 backhands and 3 jumps. Unmarked shots are not taken as straight or forehand, so for those the report gives only how often the app said cross or backhand. |

**The facts a name rests on (5b):**

- **Where it was hit from:** the contact's height, and its distance from the net.
- **How it left the racket:** speed and angle above or below the horizontal.
- **What it did in the air:** the apex above the contact, and the net clearance.
- **Where it went:** how far past the net it ended, how high it was at the end (the floor, or the next racket), and how long it flew.
- **What came before:** the speed and angle of the shot it answered. A block is the reply to a smash or a drive.

## Decisions

- **The model learns from simulation and is tested on your shots.** The 74 real shots are too few to learn from and test on at once. Learning from them would also leave nothing honest to test on. The real score therefore measures how far the simulated families are from real play. If it is low, the next step is more labelled real rallies, which the labeller already records.
- **The same honesty as Phase 3.** A name that the measurements cannot settle comes as two names with their probabilities, never a guess. The far court from a low corner camera is measured worst, and its probabilities spread accordingly.
- **The serve comes from the rally, not the model.** The first hit of a rally is the serve. Whether it was high or low comes from the flight.
- **"other" is never given by the model.** It is for a person to use.
- **Logistic regression, not a network.** About ten facts and eight families: a linear model on the facts and their products can draw curved boundaries, and its weights can be read. A network would need more data than this phase has.

## Proposed gate (for you to agree)

- **Held-out simulation:** macro-F1 within 0.05 of what the true facts give, and calibrated as in 5c.
- **Your shots:** macro-F1 of at least 0.70 on the shots whose hit was found, both cameras pooled, with each camera reported.

## Not in this phase

- **The rally's end** (who won, out, net, unreturnable) is Phase 6. Half of the extra hits on the real clips come after the rally's last hit, and they belong there.
- **Learning from public data** (ShuttleSet): its licence needs checking, and it gives positions on the court, not flights in 3D.
