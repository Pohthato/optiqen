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
- **It can test who wins, not why.** In the two games looked at so far, every rally that ended with the shuttle landing in is marked a winner, whether or not the receiver could have got there, and no error is marked forced or unforced.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 7a | **ShuttleSet in the app's court.** Read every game into rallies: the hitter, the time (from the frame and the video's frame rate), the shot's name among the app's eight or the serve, both players' positions and the contact on the floor in metres, the end and the winner. The mapping of the 18 types to the app's names is written down here. Each position gets a covariance from a few pixels of clicking error carried through that video's court corners, so the far end is less sure. Rallies with an unknown shot or a missing position are set aside and counted. The test matches are chosen here, before any model is fitted: one match in four at random, and, as the stressed test, every match of two players, so it sees players never learnt from. | `worker/evaluation/shuttleset.py` | Every rally's winner agrees with the score after it. Landings marked out lie outside the singles lines, and landings in lie inside, within their uncertainty, 95 % of the time; this also settles whether the corners are the singles or the doubles court. The counts kept and set aside. |
| 7b | **The rules on professionals.** The 6f time to spare at every hit, from ShuttleSet's positions and times, with a reach of 1.4 ± 0.1 m (the app finds 1.37 to 1.39 m on your clips). The rules' value: at each hit, how pressed the hitter was by the shot they are answering, made into a chance of winning the rally by one curve fitted on the training matches. | `worker/evaluation/value_benchmark.py` | How often a pressed hitter goes on to lose, with its interval. On the rallies that ended landing in, how many the 6f rule calls a winner rather than let in: professionals let few in, so many let-in calls would say the 250 ms start or the 3.5 m/s is wrong for them. |
| 7c | **The learned value.** At each hit, the chance its hitter wins the rally, from what the app also measures: where both players stood, where the hitter met the shuttle, the time the shot before gave them and their time to spare, the names of the shot before and the shot being hit (as their chances), and the same for the hit before. The hitter is always put on the same half. A logistic model on these facts and their products, as in 5c, and something bigger only if it does better on folds of the training matches. Nothing in it sees where the shot being hit comes down, so a rally's end cannot leak in. Its uncertainty comes from refits on resampled training matches and from draws of its inputs within their covariance. | `worker/rally_value.py`, `worker/rally_value.json`, `worker/evaluation/value_benchmark.py` | On the held-out matches and the held-out players, it tells who wins better than the rules (log loss), with the difference's 95 % interval, resampled by match, clear of zero. Its chances are calibrated, in tenths. Both again with the held-out positions as uncertain as the app's on your clips. |
| 7d | **Into the one command, and your rallies.** Each shot gets its `value`: each player's chance of winning the rally from that hit, with its interval. The turn is the first hit from which the eventual winner's chance stayed above even, with its chance, or too close to call; when that is the first hit, there was no turn. Your five clips are run again from their saved analyses. | `worker/analyse_rally.py` | Your five rallies' values and turns, pushed before your 6f labels are read, and scored on them by 6f's rule. |

## What 7a needs from you

- **The go-ahead to download ShuttleSet:** the `ShuttleSet` folder of `github.com/wywyWang/CoachAI-Projects` (the whole repository is 37 MB, and only that folder is needed). It would sit in a git-ignored folder, as your clips do, and never be committed. The MIT notice and the paper's citation go in the docs beside the model learnt from it.
- **Whether a value learnt from professionals is one you want.** See the first decision.

## Proposed gate (for you to agree)

- **Held-out matches and held-out players:** the learned value tells who wins the rally better than the rules, the difference's 95 % interval clear of zero, and its chances are calibrated.
- **With positions as uncertain as on your clips:** reported, with no bar, since that is what the app will see.
- **Your rallies:** reported, not claimed. Five rallies cannot test a value learnt from professionals. Your labels and a coach's on more rallies can.

## Decisions

- **Learnt from professionals.** What a top player wins from a position is not what you or your opponent would. The value says what a position is worth at the top level. A value for club players needs club rallies with their winners; your labelled rallies could later move it, as your 66 shots moved the names in 5e.
- **Only what both ShuttleSet and the app measure.** Positions on the floor, times and names. Not heights, flights in 3D or bodies, which ShuttleSet lacks.
- **The test matches are chosen before any model is fitted, and used once,** as held-out seeds are.
- **The rules stay.** The value adds a chance and a turn. It does not replace how 6f says a rally was won, which ShuttleSet cannot test.
- **Nothing sees the future.** The value at a hit uses only what is known when the shot is hit.

## Not in this phase

- **Why errors happen.** ShuttleSet does not mark errors forced or unforced.
- **ShuttleSet22**, the 2022 matches in the same repository: a later test by year.
- **The score**, and doubles.
