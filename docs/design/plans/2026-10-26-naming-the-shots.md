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
| 5e | **Learning from your shots.** Your named shots move the simulator's model. Its weights are scaled as a whole, and the intercepts and the single facts' weights move from there, held by a hold chosen on held-out clips. Each clip is named by a model learnt without it. Each analysis keeps each shot's launch, so it can be named again without the video. | `worker/shot_names.py`, `worker/evaluation/shot_names_benchmark.py`, `worker/evaluation/hand_labels.py`, `worker/analyse_rally.py` | Real macro-F1, each clip named by a model learnt without it, beside 5d's. Names given alone are right about as often as they say. |
| 5f | **Cross or straight, forehand or backhand, from a jump.** Cross: the flight crosses the court diagonally. Backhand: the racket hand meets the shuttle across the body, on the side away from the racket arm, beyond the uncertainty. The racket arm is the one the player hits with on most shots of the clip. Jump: both feet are off the floor at contact (4c already measures it). Each is "too close to call" when the measurements cannot tell. | `worker/shot_names.py` | Simulation: each one right within its stated uncertainty. Real: your labels mark 17 cross shots, 7 backhands and 3 jumps. Unmarked shots are not taken as straight or forehand, so for those the report gives only how often the app said cross or backhand. |

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

### 5d: your shots

The five labelled clips through the one command, scored against your labels with `evaluation.hand_labels`. The hits are the same as in the players plan. Serves are named by the rally and scored with the rest; the eight families alone are beside them. The intervals (95 %) come from resampling shots. One rally's shots are not independent, so the intervals are too narrow, if anything.

| Clips | Found shots right | Macro-F1 | Eight families only | End to end |
| --- | --- | --- | --- | --- |
| All five | 31 of 66 | 0.51 (0.41–0.57) | 0.46 (0.37–0.53) | 0.47 (0.37–0.55) |
| Corner tripod, clips 6–9 | 23 of 51 | 0.53 (0.40–0.60) | 0.47 (0.33–0.55) | 0.52 (0.40–0.59) |
| Clip 10 | 8 of 15 | 0.34 (0.22–0.55) | 0.38 (0.25–0.60) | 0.30 (0.16–0.49) |

The gate proposed above (0.70 on found shots) is not met.

| Your label | Found | Named |
| --- | --- | --- |
| clear | 5 | clear 5 |
| lift | 10 | lift 10 |
| smash | 6 | smash 5, block 1 |
| serve | 5 | serve 4, push 1 |
| drive | 4 | drive 3, block 1 |
| drop | 7 | drop 3, drive 3, smash 1 |
| push | 8 | drive 4, lift 3, push 1 |
| net | 14 | push 8, block 5, drop 1 |
| block | 7 | push 5, drive 1, drop 1 |

A "push or lift" shot is counted under the name given when that name is either, so both of them are in the lift row.

- **The long shots are right; the short game is not.** Clears, lifts and smashes: 20 of 21. Net shots, blocks and pushes: 1 of 29.
- **The simulator's short game is not the one in your clips.** Most of your net shots (11 of 14) were measured leaving the racket 1.4 to 2.6 m from the net, 0.8 to 1.4 m high, at 7 to 14 m/s and 21 to 35° up. They were played again 1.0 to 2.5 m past the net, 0.3 to 1.3 m high. The simulator's net shots end 0.4 to 1.6 m past the net, 0.2 to 0.8 m high. Its pushes end 2.5 to 4.5 m past, 0.8 to 1.6 m high. Your net shots fall between the two, nearer push. All but one of your pushes went further, 3.8 to 7.7 m past the net, so they read as drives and lifts. Your blocks read as the simulator's pushes.
- **The model is sure when it should not be.** On your shots, a name given alone was right 22 of 37 times, while its stated probability averaged 0.96. When two names were given, yours was among them 10 of 24 times. The probabilities carry the measurement's uncertainty, but not the distance between the simulator's families and real play.
- **One serve was not called a serve.** The hit finder found Clip 10's serve from its flight, not as a serve, so the model named it ("push or block").
- **What would close the gap** is shots named in real play to learn from. The 66 here could be used now, each clip named by a model learnt without it (simulation plus the other four clips); 5e does this. More labelled rallies would do it better. Either way the names on the simulator's families alone should not be shown as sure.

### 5e: learning from your shots

The model learns from 61 of your shots: every named hit that was found as a shot, serves left out. A "push or lift" counts as either. It starts from the simulator's model. Its weights are scaled as a whole, and the intercepts and the single facts' weights move from there, held towards them. The products of facts stay the simulator's.

Each clip is named by a model learnt from the other four. That model's hold is chosen by naming each of those four in turn from the remaining three, on log loss over the draws. The model kept (`worker/shot_names.json`) is learnt from all five clips, its hold chosen the same way: 0.01 (log loss 46.2; 52.0 at 0.1, 73.5 at 0.001). Every clip's own choice was 0.01 too. The simulator's model stays beside it as `worker/shot_names_simulated.json`.

| Clips | Found shots right | Macro-F1 | Eight families only | End to end |
| --- | --- | --- | --- | --- |
| All five | 46 of 66 | 0.65 (0.53–0.72) | 0.62 (0.51–0.71) | 0.61 (0.50–0.69) |
| Corner tripod, clips 6–9 | 36 of 51 | 0.67 (0.55–0.76) | 0.63 (0.49–0.73) | 0.66 (0.53–0.75) |
| Clip 10 | 10 of 15 | 0.54 (0.32–0.75) | 0.62 (0.35–0.84) | 0.50 (0.27–0.69) |

In 5d, from the simulator alone, it was 31 of 66 and 0.51. The proposed gate (0.70) is still not met, though the interval reaches it.

| Your label | Found | Named |
| --- | --- | --- |
| net | 14 | net 13, drop 1 |
| clear | 5 | clear 5 |
| smash | 6 | smash 5, drive 1 |
| drop | 7 | drop 6, net 1 |
| serve | 5 | serve 4, net 1 |
| lift | 9 | lift 6, push 3 |
| push | 8 | push 5, lift 2, drive 1 |
| block | 7 | net 3, block 2, push 1, drop 1 |
| drive | 4 | block 2, smash 1, push 1 |
| push or lift | 1 | clear 1 |

The other "push or lift" shot is counted under the name given.

- **The sureness is now close to honest.** A name given alone was right 23 of 27 times, at a mean stated probability of 0.89 (5d: 22 of 37 at 0.96). When two names were given, yours was among them 31 of 35 times (5d: 10 of 24). The kept model scales the simulator's weights by 0.11: the simulator was about nine times too sure for real play.
- **Net shots and drops are now right**, 13 of 14 and 6 of 7, and clears and smashes stay right (10 of 11). Lifts lost some to push: 6 of 9, from 10 of 10.
- **Blocks and drives are still wrong:** 2 of 11. Blocks read as net shots. There are only 7 blocks and 4 drives to learn from, and what tells a block from a net shot is the shot before it.
- **Clip 10, the other venue, is named worse** (0.54 against 0.67 for the corner tripod), but on 15 shots its interval is wide.
- **Two ways of moving the model were tried.** This one was kept because its held-out log loss was lower (46.2 against 66.7). The other let the products move too; fitted to a few shots, they named two of Clip 10's fast clears "net" at 0.82 and 0.90. Choosing between two on the same shots makes the scores above a little optimistic.
- **What it costs on simulated shots.** On 5c's held-out simulation the kept model's macro-F1 is 0.58 (0.53–0.62) in the standard world and 0.53 (0.50–0.57) in the stressed one, against 0.855 and 0.851 for the simulator's model. The simulator's families are not real play's, and the model now follows real play.
- **A draw that cannot be flown is left out.** Clip 10's shot at 20.81 s (the smash given to the wrong hitter) has a launch speed uncertain by about 2000 m/s. One of its 200 draws left the racket too fast for the flight's integration and spoilt every average it was in.
- **What would help most now** is more named rallies, above all from other venues and with blocks and drives.

### 5f: cross or straight, forehand or backhand, from a jump

Each shot now has three details. Each comes with its chance, and is said when that chance is at least 0.8 either way; between, it is "too close to call".

- **Cross:** the shot ended on the other side of the court's centre line from where it was hit, at least 1.3 m across from it (a quarter of the singles width, so a shot from the middle that drifts over the line does not count). The chance is the share of the flight's 200 draws that went cross: the same draws the family is named over.
- **Backhand:** the contact was on the side of the hips away from the racket arm. A shot's arm is the one whose wrist was nearer the contact, and a pose can take one wrist for the other. So the racket arm is only as sure as the player's shots agree on it: right with the share of their shots hit with the right arm. The backhand's chance takes both arms, each at that share.
- **Jump:** the lower foot more than 0.15 m off the floor at contact. That is higher than an ankle rises on tiptoe or in a split step's hop.

The one command writes them into each shot's name, with the name as a coach would say it, the sure details first ("cross backhand lift"). The overlay and the diagram show that name, and the summary gives each player's racket arm.

**Simulation: backhand and jump** (`python -m evaluation.body_benchmark --seeds 30 --first-seed 100`). These are thirty held-out rallies, every view, both halves, at 30 and 60 fps. The first three rallies, 4c's, have no contact high enough for the simulator to jump for, so thirty are used. The truth is the same call on the true body at the exact moment of contact.

| Detail, truth | Hits at 30 fps | Said | Hits at 60 fps | Said |
| --- | --- | --- | --- | --- |
| Jump | 24 | jump 24 | 24 | jump 24 |
| No jump | 444 | no jump 442, too close to call 2 | 444 | no jump 442, too close to call 2 |
| Backhand | 6 | too close to call 6 | 6 | backhand 2, too close to call 4 |
| Forehand | 444 | forehand 433, too close to call 11 | 444 | forehand 437, too close to call 7 |
| Racket arm unsure on the true body | 18 | too close to call 18 | 18 | too close to call 18 |

- **Neither detail is ever said wrong.** Every jump said was one, and so was every backhand and forehand.
- **The simulated players are right-handed, and nearly every contact is on their racket side.** The six true backhands are one contact in each of two rallies, seen from the three views. The simulator has no real backhand strokes and no left-handers, so this tests forehands not being called backhand far more than it tests backhands.
- **A few shots are not enough to be sure of the racket arm.** Even on the true body, the left wrist is sometimes nearer the contact. For two players with three hits each, one hit went that way, so their arm is right at only 0.67 and all their backhands are too close to call. The arm came out left for no player.
- **The body's facts are as sure as they say.** The lower foot off the floor and the contact aside are each within twice their stated uncertainty for 98 to 99 % of hits.

**Simulation: cross** (`measure`, then `score`, as in 5c). This uses the same 120 held-out scenarios and the same shots, serves left out. The truth is the true flight's contact and end.

| World | Shots | Truly cross | Said cross | Said straight | Too close to call |
| --- | --- | --- | --- | --- | --- |
| Standard | 459 | 162 | 137, all cross (stated 0.98) | 268, 267 straight (stated 0.99) | 54, 24 cross (stated 0.47) |
| Stressed | 442 | 158 | 126, 123 cross (stated 0.98) | 260, 254 straight (stated 0.99) | 56, 29 cross (stated 0.53) |

- **Cross is right about as often as it says.** In the stressed world, shots said straight were cross 6 times against the 3 or 4 their stated chances allow; everywhere else the calls match their chances.
- **About one shot in eight is too close to call**, most of them from the corner and the side. There the line of sight runs across the court, so where the shuttle is across it is measured worst.

**The 5f gate.** Simulation: each detail is right within its stated uncertainty. Met for cross, jump and forehand; backhand strokes are not in the simulator. Your clips: cross 12 of 14 found, backhand 4 of 7, jump 1 of 1 with a body.

**Your clips.** Your names mark 17 cross shots, 7 backhands and 3 jumps. A name without the word is not taken as straight or forehand, so for the other shots only the number of times the app said the detail is given. Serves are left out: a serve goes cross by the rules.

| Detail | Marked | Found as a shot | What the app said | Said on the other found shots |
| --- | --- | --- | --- | --- |
| Cross | 17 | 14 | cross 12, straight 2 | cross on 5 of 48 |
| Backhand | 7 | 7 | backhand 4, forehand 2, too close to call 1 | backhand on 5 of 46 with the body seen |
| Jump | 3 | 3 | jump 1; no body at the hit 2 | jump on 1 of 52 with the body seen |

- **Cross is right on 12 of the 14.** The two said straight:
  - Clip 7's cross drive at 7.74 s went 2.7 m across but stayed on one side of the centre line. It was hit from 6.3 m, outside the doubles line as measured.
  - Clip 6's "backhand cut off cross" at 15.01 s moved 0.7 m across as measured.
- **Five shots you did not mark were said to be cross.** Each crossed the centre line and moved 1.7 to 3.2 m across: two middle net shots (Clip 9 at 1.07 s, Clip 10 at 4.13 s), Clip 9's intercepting smash, and Clip 7's backhand save clear and long net. Whether those are cross is for you to say.
- **Backhand is right on 4 of 7**, said at 0.86 to 0.95.
  - Clip 7's backhand save clear (11.85 s) is too close to call (0.57).
  - Clip 6's backhand block (13.30 s) and backhand cut off cross (15.01 s) were said forehand at 0.02 and 0.19. The contact was measured 0.4 m and 0.2 m to the racket side of the hips.
  - Of the five other shots said backhand, three are blocks and drives played at the body, as backhands often are.
  - Three serves were said backhand. Backhand serves are common in singles, but your names do not say.
- **The racket arm's sureness matters.** Clip 8's near player hit with the right arm on 2 of 5 shots, so every backhand of theirs is too close to call. Taking the arm of most shots instead made them left-handed, and five of their shots were called backhand at 0.98 to 1.00. Clip 9's far player is right on 2 of 3 (0.67). Both of Clip 7's players are right on 0.86 and 0.88 of their shots, so a backhand of theirs is said at no more than that.
- **Jump is right on the one jump with a body.** That was Clip 7's jump out stop drop at 16.13 s, with the lower foot 0.29 ± 0.03 m up. The other two jumps (Clip 8 at 9.48 s, Clip 6 at 8.66 s) have no body at the hit. Clip 10's high lift at 9.14 s was said to be a jump: the lower foot read 0.19 ± 0.03 m up, either a hop or the pose's foot off the floor. Two shots whose foot read 0.12 to 0.13 m up are too close to call.

## Decisions

- **The model learns from simulation, then from your shots, and each clip is tested by a model that never saw it.** The 61 real shots are too few to learn names from alone. They are enough to correct the simulator's model: how sure it is, and where its families sit. Leaving one clip out at a time keeps the test honest. More labelled rallies, which the labeller already records, would do more.
- **The same honesty as Phase 3.** A name that the measurements cannot settle comes as two names with their probabilities, never a guess. The far court from a low corner camera is measured worst, and its probabilities spread accordingly.
- **The serve comes from the rally, not the model.** The first hit of a rally is the serve.
- **The details are measured, not learnt.** Cross comes from where the flight went, backhand and jump from the body at contact. Their lines (1.3 m across, the side of the hips, 0.15 m off the floor) come from the court and the body, not from fitting your marks. Your clips' measurements were looked at while they were chosen, so the real scores in 5f are not wholly held out.
- **"other" is never given by the model.** It is for a person to use.
- **Logistic regression, not a network.** About ten facts and eight families: a linear model on the facts and their products can draw curved boundaries, and its weights can be read. A network would need more data than this phase has.

## Proposed gate (for you to agree)

- **Held-out simulation:** macro-F1 within 0.05 of what the true facts give, and calibrated as in 5c. The simulator's model meets it (5c), apart from drive in the stressed world (3 of 4).
- **Your shots:** macro-F1 of at least 0.70 on the shots whose hit was found, both cameras pooled, with each camera reported. Not met: 0.65 (0.53–0.72), each clip named by a model learnt without it (5e).

## Not in this phase

- **The rally's end** (who won, out, net, unreturnable) is Phase 6. Half of the extra hits on the real clips come after the rally's last hit, and they belong there.
- **Learning from public data** (ShuttleSet): its licence needs checking, and it gives positions on the court, not flights in 3D.
