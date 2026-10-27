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
| 5b | **The facts a name rests on, and simulated shots to learn from.** Per shot, the facts below, from the shuttle's state leaving the racket. Each shot now keeps that state with its whole uncertainty, so draws from it are flights that agree with themselves. The model learns from the true facts of simulated shots. The test shots are simulated rallies from seeds the learning never uses (from 5000; the first sixty are the hits benchmark's held-out rallies), run through the real hit finding and flight fitting, so their facts carry the real error. (Learning from measured facts and then drawing them from their uncertainty again would count the error twice.) | `worker/shots.py`, `worker/shot_names.py`, `worker/evaluation/shot_names_benchmark.py` | The facts are exact on true flights. The launch's uncertainty agrees with each fact's own. The test set is written once and never learned from. |
| 5c | **The model.** Multinomial logistic regression on the facts and their pairwise products, fitted with numpy and scipy (no new package). The weights are a small JSON file in the repo, and the benchmark regenerates them. Each name's probability is averaged over draws of the facts from their uncertainty. The name is given when it is at least 80 % likely; otherwise the two most likely names are given. | `worker/shot_names.py`, `worker/shot_names.json` | Held-out simulation: macro-F1, set beside the macro-F1 the true facts give (how far the families overlap by themselves). Calibrated: names given at 80 % or more are right at least 80 % of the time, for every family, and in the stressed world too. |
| 5d | **Into the one command, and the real test.** Each shot in `analyse_rally.py` gets its name with the probabilities. The overlay writes it next to the speed, and the diagram's legend names each flight. Then the five labelled clips are scored. | `worker/analyse_rally.py` | Real macro-F1 by family and by camera (the corner tripod of clips 6–9, Clip 10). It is reported twice: on the shots whose hit was found, and end to end, where a missed hit counts as a wrong name. |
| 5e | **Cross or straight, forehand or backhand, from a jump.** Cross: the flight crosses the court diagonally. Backhand: the racket hand meets the shuttle across the body, on the side away from the racket arm, beyond the uncertainty. The racket arm is the one the player hits with on most shots of the clip. Jump: both feet are off the floor at contact (4c already measures it). Each is "too close to call" when the measurements cannot tell. | `worker/shot_names.py` | Simulation: each one right within its stated uncertainty. Real: your labels mark 17 cross shots, 7 backhands and 3 jumps. Unmarked shots are not taken as straight or forehand, so for those the report gives only how often the app said cross or backhand. |

**The facts a name rests on (5b):**

- **Where it was hit from:** the contact's height, and its distance from the net.
- **How it left the racket:** speed and angle above or below the horizontal.
- **What it did in the air:** the apex above the contact, and the net clearance.
- **Where it went:** how far past the net it ended, how high it was at the end (the floor, or the next racket), and how long it flew.
- **Not what came before.** A block is the reply to a smash or a drive, but the simulator picks each shot without regard to the one before, so the model could not learn that from it. Blocks and net shots will be told apart less well than the rest.

## Results

### 5c: the model on held-out simulation

The model learnt from 14,762 shots of 3,000 rallies. Its hold was chosen on 600 more rallies by log loss: 0.0867 at 1e-7, 0.0855 at 1e-6 (chosen), 0.0898 at 1e-5, 0.0981 at 1e-4. The test shots are the 120 scenarios of the hits benchmark from seed 5000, in the standard world and in the stressed one, matched to true hits. Serves are left out. Macro-F1 has a 95 % interval from resampling whole scenarios.

| World | Shots | True facts | Measured facts | Measured, over their draws |
| --- | --- | --- | --- | --- |
| Standard | 459 | 0.869 (0.823–0.907) | 0.853 (0.817–0.885) | 0.855 (0.819–0.887) |
| Stressed | 442 | 0.869 (0.826–0.907) | 0.861 (0.816–0.899) | 0.851 (0.807–0.889) |

- **The measurement costs little.** Over the draws, macro-F1 is 0.014 under what the true facts give in the standard world and 0.018 in the stressed one.
- **Net and block are the whole gap.** Every other family's F1 is 0.87 to 1.00. Net is 0.20 to 0.34, and block 0.80 to 0.85. Of the wrong names on true facts, 21 of 25 are net called block or block called net. The simulator's two families overlap in depth, flight time and where they are played from. What tells them apart in play is what came before, and the simulator does not have it.
- **A name given alone is right.** The name came alone for 86 % of the shots in the standard world and 82 % in the stressed one. Given alone, it was right this often:

| Name | Standard | Stressed |
| --- | --- | --- |
| clear | 66 of 66 | 53 of 53 |
| drop | 88 of 88 | 80 of 80 |
| smash | 46 of 46 | 43 of 44 |
| drive | 5 of 5 | 3 of 4 |
| push | 14 of 15 | 13 of 16 |
| lift | 133 of 133 | 122 of 122 |
| net | never alone | 1 of 1 |
| block | 40 of 41 | 38 of 41 |

- **One family misses the 80 % calibration bar:** drive in the stressed world, one wrong of four. The rest pass in both worlds. Four shots are too few to say more.
- **Two names hold the truth.** When two names were given, the true one was among them for 63 of 65 shots (standard) and 75 of 81 (stressed).
- **The sureness means what it says above 0.8.** Shots at 0.8 to 0.9 (mean 0.85) were right 95 % and 93 % of the time. Shots at 0.9 and more (mean 0.99) were right 100 % and 98 %. Under 0.8 the model is less steady: at 0.5 to 0.65 (mean 0.59) it was right 45 % and 65 % of the time. Those shots get two names.
- **By view** (over the draws, 44 to 97 shots each): corner at 60 fps is best (0.92 in both worlds), side at 60 fps worst (0.69, 0.54–0.82; stressed 0.64, 0.54–0.82). One or two net or drive shots move a view's macro-F1 a long way, so the views differ less than these numbers.

## Decisions

- **The model learns from simulation and is tested on your shots.** The 74 real shots are too few to learn from and test on at once. Learning from them would also leave nothing honest to test on. The real score therefore measures how far the simulated families are from real play. If it is low, the next step is more labelled real rallies, which the labeller already records.
- **The same honesty as Phase 3.** A name that the measurements cannot settle comes as two names with their probabilities, never a guess. The far court from a low corner camera is measured worst, and its probabilities spread accordingly.
- **The serve comes from the rally, not the model.** The first hit of a rally is the serve.
- **"other" is never given by the model.** It is for a person to use.
- **Logistic regression, not a network.** About ten facts and eight families: a linear model on the facts and their products can draw curved boundaries, and its weights can be read. A network would need more data than this phase has.

## Proposed gate (for you to agree)

- **Held-out simulation:** macro-F1 within 0.05 of what the true facts give, and calibrated as in 5c.
- **Your shots:** macro-F1 of at least 0.70 on the shots whose hit was found, both cameras pooled, with each camera reported.

## Not in this phase

- **The rally's end** (who won, out, net, unreturnable) is Phase 6. Half of the extra hits on the real clips come after the rally's last hit, and they belong there.
- **Learning from public data** (ShuttleSet): its licence needs checking, and it gives positions on the court, not flights in 3D.
