# The Value of Each Position (Phase 7) — Plan

**Goal:** At every hit, the chance each player goes on to win the rally, learnt from rallies whose winners are known, with its uncertainty. The rally turned where that chance swung for good.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`, L5 ("a learned rally-value model later"). The spec's gate is offline lift over the rules: on rallies it never saw, the learned value must tell who wins better than the 6f rules do.

## What is already there

- **Pressure at each hit (6e):** how far the receiver had to go, the time the flight gave them and the time they had to spare, each with its uncertainty (`worker/pressure.py`).
- **The 6f rules** (`worker/rally_reasoning.py`): winner or let in, forced or unforced, and the shot where the rally turned. The app's answers on your five rallies are pushed (`0fd9934`). Your labels for them are not in yet.
- **Where both players stood at every hit**, with its covariance (Phase 4), each shot's name with its chance (Phase 5), and the end and winner (6c).
- **Nothing to learn a value from.** A simulated rally's length and end are drawn before it is played, so who wins it has nothing to do with where anyone stood. A model learnt there would learn nothing. Five rallies of yours are far too few.

## The data: ShuttleSet

ShuttleSet (Wang et al., KDD 2023) holds 44 professional singles matches from 2018 to 2021, between 27 of the top men and women. That is 104 games, 3,685 rallies and 36,492 shots, labelled by hand from broadcast video. For every shot it gives the hitter, the frame, the shot's type (18 of them), where both players stood at the hit, and where the shot was met or landed. Each rally's last shot says how it ended (landed in, out, into the net or not over it) and who won.

- **Positions are pixels in the broadcast image.** Each video comes with its four court corners, so they map to the floor in metres. There are no heights.
- **The licence is MIT.** The data is in `github.com/wywyWang/CoachAI-Projects`, whose licence is MIT (copyright Wei-Yao Wang, 2022). The dataset's README asks for the paper to be cited and sets no terms of its own. MIT allows commercial use as long as the notice is kept.
- **It can test who wins, and little of why.** A shot that came down in is marked either a winner or misjudged by the receiver, which is let in (47 of the 1,189 kept). No error is marked forced or unforced.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 7a | **ShuttleSet in the app's court.** Read every game into rallies: the hitter, the time (from the frame and the video's frame rate), the shot's name among the app's eight or the serve, both players' positions and the contact on the floor in metres, the end and the winner. The mapping of the 18 types to the app's names is written down here. Each position gets a covariance from a few pixels of clicking error carried through that video's court corners, so the far end is less sure. A serve ShuttleSet does not name was not labelled, so its rally starts at the return. Rallies with another unknown shot or a missing position are set aside and counted. The test matches are chosen here, before any model is fitted: as the stressed test, every match of one man and one woman drawn at random among those with two to five matches, so it sees players never learnt from; and one in four of the other matches. | `worker/evaluation/shuttleset.py` | Every rally's winner agrees with the score after it. Landings marked out lie outside the singles lines, and landings in lie inside, within their uncertainty, 95 % of the time; this also settles whether the corners are the singles or the doubles court. The counts kept and set aside. |
| 7b | **The rules on professionals.** The 6f time to spare at every hit, from ShuttleSet's positions and times, with a reach of 1.4 ± 0.1 m (the app finds 1.37 to 1.39 m on your clips) and each hit's frame one frame out. ShuttleSet does not say when a shot that came down in reached the floor, so its flight takes as long as answered shots of its name took over the same way. The rules' value: at each hit, how pressed the hitter was by the shot they are answering, made into a chance of winning the rally by one curve fitted on the training matches. | `worker/evaluation/value_benchmark.py` | How often a pressed hitter goes on to lose, with its interval. On the rallies that ended landing in, the 6f rule's winner or let in against ShuttleSet's (47 let in of 1,189). Many let-in calls on winners would say the 250 ms start or the 3.5 m/s is wrong for professionals. |
| 7c | **The learned value.** At each hit, the chance its hitter wins the rally, from what the app also measures: where both players stood, where the hitter met the shuttle, the time the shot before gave them and their time to spare, the names of the shot before and the shot being hit (as their chances), and the same for the hit before. The hitter is always put on the same half. A logistic model on these facts and their products, as in 5c, and something bigger only if it does better on folds of the training matches. Nothing in it sees where the shot being hit comes down, so a rally's end cannot leak in. Its uncertainty comes from refits on resampled training matches and from draws of its inputs within their covariance. | `worker/rally_value.py`, `worker/rally_value.json`, `worker/evaluation/value_benchmark.py` | On the held-out matches and the held-out players, it tells who wins better than the rules (log loss), with the difference's 95 % interval, resampled by match, clear of zero. Its chances are calibrated, in tenths. Both again with the held-out positions as uncertain as the app's on your clips. |
| 7d | **Into the one command, and your rallies.** Each shot gets its `value`: each player's chance of winning the rally from that hit, with its interval. The turn is the first hit from which the eventual winner's chance stayed above even, with its chance, or too close to call; when that is the first hit, there was no turn. Your five clips are run again from their saved analyses. | `worker/analyse_rally.py` | Your five rallies' values and turns, pushed before your 6f labels are read, and scored on them by 6f's rule. |

## Results

### 7a: ShuttleSet on the court

You gave the go-ahead on 2026-11-04, and agreed to a value learnt from professionals. ShuttleSet is in `worker/ShuttleSet/` (git-ignored, 4.3 MB), taken from the repository's commit `45517f7` (2026-07-17) with its README and MIT licence. `python -m evaluation.shuttleset ShuttleSet` reads it and prints the checks below.

**The corners are the doubles court's.** Mapped so, where the last shot came down agrees with how ShuttleSet says each rally ended:

| End | Rallies | Agree, within twice their uncertainty |
| --- | --- | --- |
| Winner: came down in | 1,142 | 1,142 (99.7–100 %) |
| Let in: misjudged, came down in | 47 | 47 (92.4–100 %) |
| Out | 997 | 987, 99.0 % (98.2–99.5) |
| Into the net: on the hitter's side, or at the net | 806 | 777, 96.4 % (94.9–97.5) |
| Not over the net: the same | 354 | 314, 88.7 % (85.0–91.6) |

- **Let in and out sit at the lines.** The median let in came down 14 cm inside a line, and the median out 20 cm beyond one.
- **Every winner agrees with the score.** On all 3,314 rallies whose score before them is known, the winner's score went up by one and the other's did not.
- **The 40 "not over" ends on the receiver's side are left as they are.** Nothing that follows uses where a net end came down.
- **Players were clicked above their feet.** As read, servers stood 3.00 m from the net in the near half and 3.27 m in the far, and receivers 3.78 and 4.10 m. Moving each click on a player 3.42 px down the image puts the servers at 3.11 m in both halves (fitted), and the receivers at 3.88 and 3.93 m (not fitted, so a check). The shuttle's clicks are not moved.
- **A click 3 px out either way** is then 9 cm on the near half and 15 cm on the far, at the median. That is assumed, not measured.
- **Frame rates:** 25 fps for 19 videos, 30 for 24 and 29.97 for one.
- **Kept and set aside.** 3,346 of the 3,683 rallies are kept, with 33,408 hits. Set aside: 175 with no end, 128 with a shot after the serve not named, 19 whose winner is not the one the end gives, 11 missing a position and 4 faults. In the rallies with an end, 1,073 serves are not named, nearly all without positions, and those rallies start at the return.
- **Names** in the app's terms: net 7,306, lift 5,346, smash 4,513, block 3,435, drop 3,296, push 2,808, clear 2,755, serve 2,382 and drive 1,567. The rush, a hit down from the net, is called a smash, since the app has no name for it.
- **The test matches, fixed before any model.** The stressed test is every match of SHI Yuqi and Pornpawee Chochuwong (5 matches, 311 rallies). The held-out test is 10 of the other 39 matches (853 rallies). The 29 left (2,182 rallies, 22,165 hits) are all a model may learn from.

### 7b: the rules on professionals

This runs on the 29 training matches (2,182 rallies) only. The test matches were not opened. `python -m evaluation.value_benchmark ShuttleSet` prints it.

- **The time to spare holds up on the shots that were answered.** Every answered shot was reached, and the rules mostly say so. Of the 19,423 answered hits whose positions agree, 3.4 % (2.8–4.0) are called out of reach at 95 %, within the 5 % such a call allows. Another 560 (2.8 %) contradict each other and have no pressure. 17.5 % are called under pressure.
- **Being under pressure says almost nothing about who wins the rally.** A hitter called under pressure by the shot they answered goes on to lose 51.3 % of the time (50.2–52.5), and one called not under pressure 50.0 % (49.6–50.5). The rules' value, the curve through that, scores 0.6929 nats a hit, against 0.6930 for even odds. That is the bar for 7c, and it is low.
- **The winner rule fails on professionals.** Of the rallies that ended with the shuttle down in:

| ShuttleSet says | Called winner | Called let in | Too close to call |
| --- | --- | --- | --- |
| Winner | 85 | 183 | 464 |
| Let in | 0 | 21 | 17 |

Of the 289 called, 106 agree (37 %). The rule calls a shot a winner only when the receiver had no time to get there. Most professional winners left them time to get there but not to play the shot. Smash winners came down in about half a second, leaving the receiver 85 ms to spare at the median after a 250 ms start, and drops and net shots did the same. The let-ins, mostly clears, lifts and pushes misjudged at the back, are called rightly. The flight to the floor is taken as short, which favours a winner, so true flight times would call even more winners let in.

**The fix, learnt from the training matches.** A shot that came down in is let in with the chance that the receiver could get there, times the chance a professional left such a shot. The second is learnt by how far inside the lines it came down and whether it was a clear or a lift, a push or another shot (`rally_reasoning.LET_IN`). The let-ins were mostly clears and lifts on or near a line: of 38 let in, 25 were clears or lifts, 26 came down within 20 cm of a line, and no drop or net shot was let in. Leaving each training match out of its own fit:

| ShuttleSet says | Called winner | Called let in | Too close to call |
| --- | --- | --- | --- |
| Winner | 668 | 0 | 64 |
| Let in | 6 | 0 | 32 |

Of the 674 called, 668 agree (99.1 %). It scores 0.125 nats a rally against 0.197 for the let-in rate alone. It never calls a shot let in: a professional left only about three in four even of the clears on a line that they could get to. Without the chance of getting there, by line and name alone, it scores 0.125 too, so the time to spare adds little. The test matches will score it once, with 7c.

- **It is learnt from professionals.** Club players may leave more shots they could get to. Your labels will say.
- **Errors**, which ShuttleSet does not mark forced or unforced: 268 are called forced, 668 unforced and 476 too close to call.
- **Flight times.** Answered shots spread around their name's line by 124 ms (net shots) to 306 ms (lifts).

### 7c: the learned value

This runs on the 29 training matches only. The model was pushed before the test matches were opened. `python -m evaluation.value_benchmark ShuttleSet learn` chooses it, fits it and writes `worker/rally_value.json`.

**Professional rallies are close to a coin toss at every hit.** Of the 22,165 training hits, the hitter went on to win 49 %. A player hitting a smash won 58 %, and one hitting any other shot 45 to 51 %. Where the receiver stood moves that by 2 points at most, and so does where the hitter met the shuttle. What decides a rally is whether the next shot ends it, and nothing known at the hit shows that: the last hitter won 35 % of the time and the one before them 65 %, and the swing is as large five hits from the end.

Each model on five folds of the training matches, each fold told by a model fitted on the other four, in nats a hit (even odds is 0.6930):

| Model | Weights | Best l2 | Log loss |
| --- | --- | --- | --- |
| The rules (7b's curve) | 2 | | 0.6930 |
| Names: the shot being hit, the shot before and the one before that | 29 | 0.1 | 0.6908 |
| Names and their products | 435 | 1 | 0.6910 |
| All the facts | 47 | 0.1 | 0.6909 |
| All the facts and their products, as in 5c | 1,128 | 1 | 0.6916 |

- **The names alone are kept.** Nothing bigger was tried, since the products did no better. On the folds the names tell who wins 0.0022 nats a hit better than the rules (0.0013 to 0.0032, resampled by match).
- **Positions do not enter it,** so the value is as sure on your clips as on ShuttleSet's. Its interval comes from 20 refits on the training matches drawn with replacement.

**The gate, fixed before the test matches were opened.** On the held-out matches, and apart on the held-out players' matches:

- the value tells who wins better than the rules, the 95 % interval of the difference, resampled by match, above zero;
- its chances are calibrated: refitting the share won on their log odds, the intercept's 95 % interval holds 0 and the slope's holds 1. The tenths are shown.

With positions as uncertain as the app's on your clips, both are reported with no bar. That moves each player by 0.13 m in the near half and 0.25 m in the far, and where the shuttle was met by 0.14 m across and 0.16 m along in the near half and 0.25 m and 0.45 m in the far: the medians on your five clips. The let-in rule fixed in 7b is scored on the same matches.

A pass would say the value's chances can be trusted, not that it can say where a rally turned. On the training hits its chances run from 0.44 to 0.58, so it is never sure who will win.

**On the test matches, opened once** with `python -m evaluation.value_benchmark ShuttleSet test`, after `8f394ca` was pushed. Log loss in nats a hit:

| Matches | Hits | The value | The rules | The value's lift | Calibration intercept | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- |
| Held out (10) | 8,582 | 0.6911 | 0.6930 | 0.0019 (0.0006 to 0.0034) | −0.00 (−0.02 to 0.02) | 1.01 (0.66 to 1.40) |
| Held-out players (5) | 2,661 | 0.6929 | 0.6928 | −0.0001 (−0.0057 to 0.0046) | −0.02 (−0.06 to 0.03) | 0.54 (−0.92 to 1.83) |

- **The gate is not met.** On the held-out matches the value tells who wins a little better than the rules, and its chances are calibrated: hits it gave 0.48 at the mean were won 47.9 % of the time, and hits it gave 0.54 were won 53.3 % (51.1 to 56.2). On the matches of the two players it never learnt from, it does no better than the rules, and the slope's interval is too wide to say whether its chances mean anything there.
- **With positions as uncertain as on your clips, nothing changes.** The value does not use positions, and the rules' value stays at 0.6930 and 0.6928.
- **The let-in rule fixed in 7b holds.** On the held-out matches it called 255 winners winner and none let in, and left 43 winners and 8 let-ins too close to call. That is 0.079 nats a rally, against 0.128 for the training matches' let-in rate. On the held-out players' matches it called 92 winners winner, none wrongly, and left 21 too close to call: 0.066 against 0.077.

**What this means for 7d.** As far as the app can measure it, a position is worth nearly the same everywhere at the top level. The chance of winning stays between 0.44 and 0.58, so no hit can be called the turn, and the value did not carry over to new players. Put into the app as planned, it would give each shot a number that is too close to call on almost every shot. 7d waits on your decision.

## The gate

Fixed in 7c before the test matches were opened, from what was proposed here:

- **Held-out matches and held-out players:** the learned value tells who wins the rally better than the rules, the difference's 95 % interval clear of zero, and its chances are calibrated.
- **With positions as uncertain as on your clips:** reported, with no bar, since that is what the app will see.
- **Your rallies:** reported, not claimed. Five rallies cannot test a value learnt from professionals. Your labels and a coach's on more rallies can.

## Decisions

- **Learnt from professionals.** What a top player wins from a position is not what you or your opponent would. The value says what a position is worth at the top level. A value for club players needs club rallies with their winners; your labelled rallies could later move it, as your 66 shots moved the names in 5e.
- **Only what both ShuttleSet and the app measure.** Positions on the floor, times and names. Not heights, flights in 3D or bodies, which ShuttleSet lacks.
- **The test matches are chosen before any model is fitted, and used once,** as held-out seeds are.
- **Held-out players have two to five matches.** The first draw, among all players, took Viktor Axelsen, whose 11 matches with the random quarter would have left 24 of the 44 to learn from. The rule was changed before any model was fitted or any outcome looked at.
- **The rules stay.** The value adds a chance and a turn. It does not replace how 6f says a rally was won, which ShuttleSet cannot test.
- **Nothing sees the future.** The value at a hit uses only what is known when the shot is hit.

## Not in this phase

- **Why errors happen.** ShuttleSet does not mark errors forced or unforced.
- **ShuttleSet22**, the 2022 matches in the same repository: a later test by year.
- **The score**, and doubles.
