# The Players in 3D (Phase 4) — Plan

**Goal:** Understand each player's body at every moment, and above all at each hit: where they stood, how high and how far in front they met the shuttle, how they reached, lunged and recovered. Everything is in court metres, with an uncertainty, so coaching (Phase 8) can say "you met it 40 cm behind your body" and show the frames that prove it.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md`, L3 (player) and L6 (biomechanics). The spec's gate is "joint error and jitter within agreed limits"; the limits proposed below are for you to agree.

## What is already there

- **The legacy worker** runs an Ultralytics YOLO pose model (17 COCO joints) and picks the near and far player by image depth and continuity (`worker/handler.py`, `PlayerSelector`). It reads contacts and split steps from 2D wrists and ankles (`worker/badminton.py`).
- **Pose training.** `worker/train_pose.py` and `docs/pose-training.md` (uncommitted, yours) train a 17-joint badminton model in that format.
- **Pose labels.** The golden set already takes labelled poses at hits: `poses` with the player and 17 `[x, y, visibility]` joints.
- **Phase 3** gives every hit's time, its 3D contact point and the hitter's half, and every frame's camera.

## Decision for you: the pose model's licence

Ultralytics YOLO is AGPL-3.0. Running it in a hosted service can oblige you to publish the service's source. RTMPose (MMPose, Apache-2.0) is the same class of model under a permissive licence. Phase 4's geometry takes 2D joints from any detector through one small interface, so this choice does not block the work; it decides which weights ship.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 4a | **Who is who.** Per frame, every detected person's ankles are put on the floor through the camera. The two players are the people standing in the two halves of the court (with a margin); spectators, umpire and line judges are not. Identity carries frame to frame by position and stature, and swaps only with the ends. The hitter of every Phase 3 hit is checked against the player nearest its contact point. | `worker/players.py`, `worker/track_players.py` | Synthetic people (players plus bystanders on and off court, crossing in the image): identity right in ≥ 99 % of frames. Real: on labelled poses, the player and the 2D joint error reported per view. |
| 4b | **The skeleton in 3D.** Per player, bone lengths are fitted once for the clip. Every frame's 3D joints are then fitted to the 2D joints through the camera, confidence-weighted and robust. Constraints: the bones keep their length, a foot that is still is on the floor, and motion is smooth (a sliding window). Uncertainty comes from the fit. | `worker/geometry/skeleton.py` | Synthetic skeletons (stance, lunge, overhead reach, jump) from the three views at 30 and 60 fps, with 2D noise and dropout. Proposed limits: median joint error < 10 cm behind and from the corner, < 15 cm from the side; jitter (frame-to-frame acceleration) reported. |
| 4c | **The body at each hit.** See the facts listed below. | `worker/body.py` | Synthetic: each fact exact on known skeletons, and within its stated uncertainty on fitted ones. Real: against labelled poses at hits (2D now; 3D needs the two-phone session in the spec). |
| 4d | **Into the one command.** `analyse_rally.py` adds the players: shots carry the hitter's body facts, the overlay draws the skeletons, and the diagram adds where each player stood at each hit. | `worker/analyse_rally.py` | Runs end to end on the Hendry clip; the result validates. |

**The body facts (4c):**

- **Contact height** against the player's standing reach, from their own skeleton.
- **Contact position** in front of, behind and beside the body: relative to the pelvis, along the player's facing.
- **Arm extension at contact:** the elbow angle.
- **Trunk rotation:** the shoulder line against the hip line.
- **For low shots,** lunge depth: front-knee flexion and pelvis height.
- **Split step:** the pelvis dip and its timing against the opponent's hit.
- **Recovery:** the time back to the base position after the hit.
- **Centre of mass** from segment mass fractions (de Leva), and whether it is over the base of support.

## Decisions

- **Feet fix the scale.** One camera sees no depth, but a player's foot on the floor is a known point in 3D. Bone lengths that do not change carry it to the rest of the body.
- **The same honesty as Phase 3.**
  - Every fact has an uncertainty.
  - A side-on camera that cannot see a lunge's depth says so instead of guessing.
  - Facts that need 60 fps are withheld at 30 fps.
- **Synthetic first, real second.** As with the shuttle: synthetic skeletons measure the geometry exactly. Your labelled poses measure the 2D detector on real footage. True 3D accuracy needs the two-phone session in the spec.

## 4a results (who is who)

**How it works (`worker/players.py`, `worker/track_players.py`).**

1. **Where they stand.** Each detected person's ankles are put on the floor (at ankle height) through the frame's camera.
2. **Really standing there.** The person must really be standing there: the ray through their nose (or shoulders) must pass a human height (0.8–2.6 m, a lunge to a jump) over that floor point. From the side, the umpire's raised feet land in the court, but their head would be about 2.7 m above that point.
3. **In play.** A player stands in one half, within 1 m beside the doubles sidelines and 2.5 m behind the baseline.
4. **Identity carries.** Identity carries by position: the person within reach of a 9 m/s run from where the player was last seen. The gate widens with how far a few pixels of detector error move that person's floor point (most of a metre at the far baseline seen from behind). A lost player is found again because the reach grows while they are unseen.

**Synthetic check (`worker/test_players.py`).**

- **The scenes.** Random rallies' two players among:
  - an umpire on a high chair;
  - seated line judges;
  - spectators behind the court;
  - a player on the next court.
- **The detector.** Everyone is seen from the three views at 30 and 60 fps through a steadily held phone, with 3 px of joint noise, 5 % of player detections missed and the detections in random order.
- **Result.** Over 16,161 player frames (4 rallies, 3 views, 2 frame rates), each player was the person picked in every frame they were detected: 100 % in all three views. No bystander was ever taken for a player.

**What this does not show yet.**

- **The people are simple.** They are standing figures. A detector on real footage merges, splits and half-sees people, and players overlap in the image.
- **Real accuracy needs your labels.** The labeller's poses at hits measure this on real footage.
- **No detector in this step.** `track_players.py` reads any detector's people from a JSON file. Which detector ships waits for the licence decision above.
