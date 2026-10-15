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

**Recommended: MediaPipe's pose landmarker (BlazePose GHUM, Apache 2.0).** Besides 2D joints it gives each person's pose in 3D, in metres. One camera cannot see that depth, so this matters: 4b uses it as its learned pose prior, and that halves the error (below). `worker/detect_poses.py` runs it.

- **The model file.** It needs `pose_landmarker_heavy.task` (about 29 MB) from Google's MediaPipe model storage, downloaded once.
- **The worker image.** The `mediapipe` package brings `opencv-contrib-python`, which replaces the image's `opencv-python-headless`. That is settled when 4d puts it in the pipeline.

## Steps

Each step is pushed on its own.

| Step | What | Files | Acceptance |
| --- | --- | --- | --- |
| 4a | **Who is who.** Per frame, every detected person's ankles are put on the floor through the camera. The two players are the people standing in the two halves of the court (with a margin); spectators, umpire and line judges are not. Identity carries frame to frame by position and stature, and swaps only with the ends. The hitter of every Phase 3 hit is checked against the player nearest its contact point. | `worker/players.py`, `worker/track_players.py` | Synthetic people (players plus bystanders on and off court, crossing in the image): identity right in ≥ 99 % of frames. Real: on labelled poses, the player and the 2D joint error reported per view. |
| 4b | **The skeleton in 3D.** Per player, bone lengths are fitted once for the clip. Every frame's 3D joints are then fitted to the 2D joints through the camera, confidence-weighted and robust. Constraints: the bones keep their length, a foot that is still is on the floor, and motion is smooth (a sliding window). Uncertainty comes from the fit. | `worker/geometry/skeleton.py`, `worker/detect_poses.py`, `worker/lift_players.py` | Synthetic skeletons (stance, lunge, overhead reach, jump) from the three views at 30 and 60 fps, with 2D noise and dropout. Proposed limits: median joint error < 10 cm behind and from the corner, < 15 cm from the side; jitter (frame-to-frame acceleration) reported. |
| 4c | **The body at each hit.** See the facts listed below. | `worker/body.py`, `worker/evaluation/body_benchmark.py` | Synthetic: each fact exact on known skeletons, and within its stated uncertainty on fitted ones. Real: against labelled poses at hits (2D now; 3D needs the two-phone session in the spec). |
| 4d | **Into the one command.** `analyse_rally.py` adds the players: shots carry the hitter's body facts, the overlay draws the skeletons, and the diagram adds where each player stood at each hit. | `worker/analyse_rally.py` | Runs end to end on the Hendry clip; the result validates. |

**The body facts (4c):**

- **Contact height** against the player's standing reach, from their own skeleton.
- **Contact position** in front of, behind and beside the body: relative to the pelvis, along the player's facing.
- **Arm extension at contact:** the elbow angle.
- **Trunk rotation:** the shoulder line against the hip line.
- **For low shots,** lunge depth: front-knee flexion and pelvis height.
- **Split step:** the pelvis dip and its timing against the opponent's hit.
- **Recovery:** as built, how far from base the player is when the opponent hits. The plan said the time back to base. Distance at that moment is what matters for the next shot, and it needs no threshold for "back".
- **Centre of mass** from segment mass fractions (de Leva). As built: how far it is in front of or behind the middle of the feet.

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

## 4b results (the skeleton in 3D)

**How it works.**

1. **Every person, with a 3D pose (`worker/detect_poses.py`).** MediaPipe's pose landmarker looks at the whole frame, then again in square crops. A far player is a hundred pixels tall, too small for the whole frame, so the crops are sized from the camera so that a player is about 40 % of a crop. Each frame looks again around everyone found on the court in the frame before, and every 10 frames (or when a half has nobody) at crops tiling everywhere a player could stand. A person found twice is kept once. A 3D pose is kept only when it agrees with its own 2D joints. The output is `track_players.py`'s input, with the 3D pose added, and the player tracker carries it through.
2. **The bodies (`worker/geometry/skeleton.py`, `worker/lift_players.py`).** The whole clip is fitted at once, after a first pass that fits each frame from the one before. The constraints:
   - **What a detector sees.** The 2D joints, through each frame's camera.
   - **The learned pose.** It pulls the body's joints about the hips' middle towards the model's 3D pose.
   - **The body.** The bones keep the player's own lengths, re-estimated over the clip. Knees bend forward. The trunk is roughly upright, and the shoulders and hips roughly level.
   - **The floor.** No foot goes below the floor. Each foot is on it unless clearly off it.
   - **Motion.** No joint moves faster than a swinging wrist. The hips' middle accelerates no harder than a player can. A foot on the floor does not slide.
   - **Hard and soft rules.** The rules about the body, the floor and motion are firm. Only what a detector or a model can get wrong is weighed robustly: a joint, the model's pose, and whether a foot is down.
   - **The output.** Each joint's uncertainty comes from the fit.

**What the testing found and fixed.**

- **Holding only the lower foot to the floor pulled the body toward the camera.** That rule penalised a foot below the floor much harder than one above it. Pixel noise therefore nudged the body up, and seen from above, up along the line of sight is toward the camera. The far player from behind sat 10–20 cm too near the camera, even with a perfect pose model. Holding each foot to the floor on its own removes the bias: 15.0 → 7.1 cm median on the tuning seeds.
- **A robust loss on every rule let the rules give way.** On one jump smash, the far player was put 3 m away with both feet 20 cm below the floor, and stayed there for 3 s. The robust loss is now kept to the detector's and the model's values.
- **A jump held to the floor looks like stepping back.** In one frame, a player jumping and a player standing half a metre further away look alike. Fitting the whole clip settles it. Faking a jump would take both feet sliding a metre and back, and the hips' middle swinging out and in, which the no-slide and acceleration rules forbid. A foot well off the floor is no longer pulled down at all.
- **The synthetic players' feet slid.** The motion generator now plants each foot until it lifts, swings it to its next place, and steps the lunging foot out and back.

**Synthetic check (`python -m evaluation.skeleton_benchmark`).**

- **The scenarios.** Held-out rallies (seeds 100–102, none used in tuning), 5 s each. Every view, both halves, at 30 and 60 fps, through a steadily held phone. The detector's joints carry 3 px of noise, and 5 % of joints are missed.
- **The pose model is emulated.** Each frame it reports the true pose turned 8°, scaled 7 % and with each joint 6 cm off.

| View, player | Joint error, median / p90 (with model) | Placement along the line of sight, median (with model) | Airborne, median (with model) | Within 2σ (with model) | Joint error, median / p90 (geometry alone) |
| --- | --- | --- | --- | --- | --- |
| Behind, near | 4.1 / 9.2 cm | +0.7 cm | 6.8 cm | 96 % | 11.1 / 56.8 cm |
| Behind, far | 7.0 / 20.0 cm | +0.9 cm | 20.1 cm | 81 % | 16.9 / 50.8 cm |
| Corner, near | 3.7 / 8.4 cm | +0.8 cm | 9.6 cm | 97 % | 8.0 / 44.7 cm |
| Corner, far | 6.7 / 17.3 cm | +1.4 cm | 20.1 cm | 84 % | 14.8 / 41.3 cm |
| Side, near | 3.2 / 7.3 cm | +0.3 cm | 5.9 cm | 98 % | 4.6 / 23.6 cm |
| Side, far | 3.8 / 8.8 cm | +0.4 cm | 10.0 cm | 97 % | 7.3 / 26.9 cm |

**The gate.**

- **Met in simulation:** a median joint error under 10 cm behind and from the corner, and under 15 cm from the side, with the emulated model. Overall: 4.4 cm median, 12.0 cm at the 90th percentile.
- **Not met by geometry alone.** The far player is at 16.9 cm from behind and 14.8 cm from the corner, and 1 joint in 10 is off by over 40 cm. Only 73 % of errors are within 2σ. One camera needs the learned pose.
- **The jitter.** It is the fitted joints' frame-to-frame acceleration beyond the truth's: 77 m/s² at 30 fps and 235 m/s² at 60 fps (median).

**What this does not show yet.**

- **The model is emulated.** How close MediaPipe's real 3D pose comes on badminton players is unknown, and those players are fast, stretched, far away and blurred. Running it needs the model file. Measuring it needs labelled frames.
- **Detection of real far players is untested.** The crop sizes are a reasoned guess at what the landmarker needs.
- **Too confident for the far player.** 81–84 % of errors are within 2σ, where at least 95 % should be.
- **Speeds need smoothing.** Joint speeds and accelerations (4c) must come from a fit over several frames, not frame differences.
- **A foot just off the floor is held a little down.** At a step's start and end the foot is a few centimetres up. With perfect detections the body is 2 cm off.
- **The far player in the air is the weakest case:** 20 cm median, against 7 cm overall.

## 4c results (the body at each hit)

**What each hit gets (`worker/body.py`).** Every fact has an uncertainty.

At contact:

- **Contact above standing reach.** The contact's height minus the player's own standing reach with the racket. The reach comes from their fitted bones: legs straight, arm straight up, racket in line. Below it, they let the shuttle drop; above it, they jumped.
- **Contact ahead and aside.** Where the contact was against the middle of the hips: in front along the way the hips face, and out towards the racket arm. The racket arm is the one whose wrist is nearer the contact.
- **Elbow angle** of the racket arm (180° is straight).
- **Side-on.** How far the shoulder line is turned from the hip line, the racket shoulder back.
- **Lunge depth.** The hips' height, and the front knee's angle.
- **Off the floor.** The lower foot's height: a jump.
- **Weight ahead.** The centre of mass (de Leva's segment masses) in front of the middle of the feet.

Before the shot, at the opponent's hit that it answers:

- **Distance from base.** Base is the centre line, 2.8 m from the net.
- **The split step.** The hips rise (the hop), then drop lowest around the opponent's hit as the legs load to push off. Reported: how far they dropped from the hop's top, and when they were lowest against the opponent's hit. "No split step" when the lowest point is not inside that window, the drop is under 6 cm, or the drop is not clear of its uncertainty.

**How the uncertainty is worked out.** The body at the hit is interpolated between the frames either side. Each fact is worked out on 400 bodies drawn from the fit's covariance. This matters because the joints' errors go together: the whole body slides along the line of sight. The contact point's uncertainty and the reach's go in too. Facts are attached to `shots.ShotFacts` by `rally_body_facts`.

**Changes to 4b along the way.**

- **The fit keeps each frame's full covariance,** not one number per joint. A fact such as "contact ahead of the hips" depends on how the errors of the contact, the hips and the line of sight go together.
- **Its uncertainty now includes what the fit itself leaves out.** The floor rule can be about 1.2 cm wrong about a foot's height, for a foot just off the floor. That moves the whole body along the line of sight by that amount over the sine of the sight's angle to the floor: a few centimetres close up, over 10 cm for a far player seen from behind.
  - **Before:** 91 % of joints within 2σ on the tuning rallies, and only 84 % for far players. A single scale on everything would have made near players over-cautious and still left far players short.
  - **After:** this one term puts every view, near and far, at 95 % or better on the tuning rallies. It is checked below on held-out ones.
- **The floor holds a foot a little harder (cap 4.5 → 7).** With the richer motion below, a far player stepping deep in the court could have both feet fitted as off the floor. The body then floated 30–50 cm towards the camera for a second. On the tuning rallies the far player's worst 10 % went from 29.5 cm to 15.0 cm, and jumps improved too (15.6 → 9.9 cm).

**Richer synthetic players (`worker/simulation/people.py`).**

- **Moving between shots.** After their hit, players head back towards base at 2.5 m/s. They run to the next shot at up to 3.5 m/s on average, and reach for it when that is not enough.
- **Split steps.** At most opponent's hits they split-step: a 5 cm hop, landing between 80 ms before and 120 ms after the hit, then a 7 cm dip. One in seven they skip.
- **Strokes.** In each stroke the shoulders turn side-on, up to 45° for an overhead.
- **Contact distance.** The racket shoulder is an arm (60–100 % stretched) and a racket from the shuttle.
- **Lunges.** The front foot plants and stays put, with the front knee bent.

**Exact check (`worker/test_body.py`).** On skeletons built with known values, every fact comes out exactly:

- reach;
- contact ahead and aside;
- elbow, side-on and knee angles;
- jump height;
- centre of mass;
- a split step's drop and timing, and no split step on a still or steadily sinking player.

**Synthetic check (`python -m evaluation.body_benchmark`).** Held-out rallies (seeds 100–102), every view, both halves, at 30 and 60 fps. The detector, the pose model and the camera are as in 4b. The contact point is off by 5 cm on each axis, and said to be. The truth is the same facts on the true body at the exact moment of contact; for the split step, the true body 240 times a second.

| Fact | Median / p90 error, 30 fps | Within 2σ, 30 fps | Median / p90 error, 60 fps | Within 2σ, 60 fps |
| --- | --- | --- | --- | --- |
| Contact above standing reach | 3.3 / 8.4 cm | 96 % | 4.0 / 9.8 cm | 94 % |
| Contact ahead of the hips | 7.2 / 14.6 cm | 100 % | 5.5 / 22.1 cm | 96 % |
| Contact aside, to the racket arm | 7.8 / 14.4 cm | 100 % | 7.9 / 16.2 cm | 98 % |
| Elbow angle | 8.2 / 22.3 ° | 96 % | 9.2 / 21.3 ° | 98 % |
| Side-on | 7.4 / 15.1 ° | 98 % | 4.8 / 19.0 ° | 92 % |
| Hips' height | 0.9 / 2.1 cm | 100 % | 0.5 / 2.4 cm | 98 % |
| Front knee angle | 5.6 / 15.6 ° | 100 % | 5.0 / 15.4 ° | 100 % |
| Lower foot off the floor | 0.3 / 0.7 cm | 100 % | 0.2 / 0.4 cm | 100 % |
| Centre of mass ahead of the feet | 1.1 / 2.3 cm | 100 % | 1.1 / 2.2 cm | 98 % |
| Distance from base at the opponent's hit | 3.1 / 11.7 cm | 95 % | 4.2 / 13.3 cm | 95 % |
| Standing reach (per clip) | 0.7 / 0.9 cm | 100 % | 0.5 / 0.8 cm | 100 % |
| Split step, 30 fps | found 30 of 30; found where there was none 0 of 12 | timing 14.9 / 39.5 ms, 100 % within 2σ | drop 1.6 / 3.2 cm | 100 % |
| Split step, 60 fps | found 30 of 30; found where there was none 0 of 12 | timing 15.9 / 31.0 ms, 100 % within 2σ | drop 1.7 / 2.2 cm | 100 % |

51 hits at 30 fps and 51 at 60 fps.

The far player alone, from behind and the corner (both frame rates):

| Fact | Median / p90 error | Within 2σ |
| --- | --- | --- |
| Contact above standing reach | 3.2 / 11.0 cm | 88 % |
| Contact ahead of the hips | 10.2 / 26.9 cm | 94 % |
| Elbow angle | 13.4 / 25.8 ° | 94 % |
| Side-on | 7.8 / 22.3 ° | 88 % |
| Front knee angle | 8.1 / 20.1 ° | 100 % |
| Distance from base at the opponent's hit | 9.7 / 16.9 cm | 88 % |

**The gate.**

- **Exact on known skeletons:** met.
- **Within the stated uncertainty on fitted ones:** met overall. Every fact has 92–100 % of its errors within 2σ.
- **The far player seen from behind or the corner is a little under:** 88 % for reach, side-on and distance from base.
- **Nothing needs withholding at 30 fps.** The plan was to withhold facts that need 60 fps. On this evidence none does: the split step is timed within its stated uncertainty at 30 fps too.

**The bodies themselves on this richer motion** (`skeleton_benchmark`, the same held-out seeds):

| View, player | Joint error, median / p90 | Along the line of sight, median | Airborne, median | Within 2σ |
| --- | --- | --- | --- | --- |
| Behind, far | 9.5 / 25.9 cm | +3.7 cm | 9.9 cm | 91 % |
| Behind, near | 4.8 / 10.3 cm | +2.5 cm | 5.8 cm | 98 % |
| Corner, far | 8.5 / 25.0 cm | +2.0 cm | 8.9 cm | 91 % |
| Corner, near | 4.2 / 9.1 cm | +1.9 cm | 5.4 cm | 99 % |
| Side, far | 4.7 / 11.2 cm | +0.4 cm | 5.9 cm | 98 % |
| Side, near | 3.9 / 9.0 cm | +1.8 cm | 5.8 cm | 99 % |

All: 5.4 cm median, 15.2 cm at the 90th percentile, 96 % within 2σ.

**What this does not show yet.**

- **The motion is still synthetic.** Real strokes are faster: a smash's wrist moves over half a metre between frames at 30 fps. The elbow at contact will be less sure than here.
- **The split step's drop comes out about 2 cm low.** The floor rule holds a foot only a few centimetres up a little down, which flattens the hop. Its stated uncertainty includes those 2 cm.
- **Base is a fixed point.** It is the centre line, 2.8 m from the net. Real bases shift with the opponent's position, so "distance from base" is a reference, not a verdict.
- **Not yet in the pipeline.** That is 4d.
