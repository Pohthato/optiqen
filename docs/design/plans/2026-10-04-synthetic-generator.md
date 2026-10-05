# Synthetic Badminton Video Generator — Implementation Plan


**Goal:** Render synthetic badminton clips (frames, per-frame cameras, 3D shuttle truth, contacts, audio) so Phases 1b–3 can be tested against exact ground truth before real labels exist.

**Architecture:** Production shuttle physics in `worker/geometry/shuttle_physics.py` (reused by Phase 3). Test/tooling package `worker/simulation/`: rally builder, camera paths, audio, renderer, clip assembly + writer + CLI. Pure numpy/OpenCV/SciPy; no ffmpeg.

**Tech Stack:** Python 3.11+, numpy (1.26 and 2.x), OpenCV 4.10, SciPy 1.13, `unittest`.

**Spec:** `docs/design/specs/2026-10-04-perception-program-design.md` (Step 1), on top of `docs/design/specs/2026-10-01-badminton-understanding-design.md`.

## Global Constraints

- Court frame from `geometry/court_model.py`: x across from the left singles sideline (0–5.18 m), y from the near baseline (0–13.40 m), z up; net plane y = 6.70 m; net posts 1.55 m.
- Drag model `dv/dt = −g e_z − (g / v_t²)|v|v`, `g = 9.81`, feather `v_t = 6.7 m/s`; `terminal_velocity=None` means no drag.
- Capture assumption: landscape, steadily held phone (shake ≤ 0.3° rotation and ≤ 1.5 cm translation per axis by default); tripod is zero motion.
- `worker/simulation/` is tooling, never copied into the worker image; `worker/geometry/shuttle_physics.py` is production code.
- Tests use `unittest`, run from `worker/`: `python -m unittest <module> -v`.

## Review Focus

1. A shot that cannot clear the net or never comes down to the receiver must be refused with a clear error, never silently produce an impossible rally. Tested in Task 2.
2. Scene objects behind the camera (a side camera, a shuttle behind the phone) must not crash rendering or wrap around the image. Tested in Task 5.
3. Truth must line up with frames exactly: frame `i` is time `i / fps`, and `shuttle[i]` / `cameras[i]` are for that time. Tested in Task 6.
4. Writing must work with only OpenCV + SciPy (no ffmpeg) and read back with OpenCV. Tested in Task 6.
5. Determinism: the same seed gives byte-identical frames and audio. Tested in Tasks 4–5.

---

### Task 1: Shuttle physics

**Files:**
- Create: `worker/geometry/shuttle_physics.py`
- Test: `worker/test_shuttle_physics.py`

**Interfaces:**
- Produces: `GRAVITY`, `FEATHER_TERMINAL_VELOCITY`, `STEP_S`, `simulate(p0, v0, times, terminal_velocity=6.7) -> np.ndarray (N, 3)`, `time_at_height(p0, v0, height, terminal_velocity=6.7, max_time=6.0) -> float | None` (first descending pass), `solve_launch(p0, target, flight_time, terminal_velocity=6.7) -> np.ndarray (3,)` (raises `ValueError` if the target cannot be reached within 1 cm or `flight_time <= 0`).

- [ ] **Step 1: Write the failing test**

```python
# worker/test_shuttle_physics.py
import math
import unittest

import numpy as np

from geometry.shuttle_physics import GRAVITY, simulate, solve_launch, time_at_height


class SimulateTests(unittest.TestCase):
    def test_without_drag_it_is_an_exact_parabola(self):
        p0, v0 = np.array([1.0, 2.0, 1.5]), np.array([3.0, 10.0, 8.0])
        times = np.array([0.0, 0.5, 1.0])
        expected = p0 + np.outer(times, v0) - 0.5 * GRAVITY * np.outer(times**2, [0, 0, 1])
        np.testing.assert_allclose(simulate(p0, v0, times, terminal_velocity=None), expected, atol=1e-9)

    def test_vertical_drop_matches_the_analytic_drag_solution(self):
        vt = 6.7
        times = np.array([0.5, 1.0, 2.0])
        expected_z = 20.0 - (vt**2 / GRAVITY) * np.log(np.cosh(GRAVITY * times / vt))
        z = simulate([0, 0, 20.0], [0, 0, 0], times, terminal_velocity=vt)[:, 2]
        np.testing.assert_allclose(z, expected_z, atol=1e-4)

    def test_falling_speed_approaches_terminal_velocity(self):
        z = simulate([0, 0, 50.0], [0, 0, 0], [3.0, 3.01])[:, 2]
        self.assertAlmostEqual((z[0] - z[1]) / 0.01, 6.7, delta=0.07)

    def test_time_zero_is_the_launch_point_and_times_must_increase(self):
        np.testing.assert_allclose(simulate([1, 2, 3], [5, 5, 5], [0.0])[0], [1, 2, 3])
        with self.assertRaises(ValueError):
            simulate([0, 0, 1], [1, 1, 1], [0.5, 0.2])


class TimeAtHeightTests(unittest.TestCase):
    def test_descending_crossing_matches_the_parabola(self):
        expected = (10 + math.sqrt(100 - 4 * (GRAVITY / 2) * 1.0)) / GRAVITY
        t = time_at_height([0, 0, 1.0], [0, 0, 10.0], 2.0, terminal_velocity=None)
        self.assertAlmostEqual(t, expected, places=4)

    def test_height_above_the_apex_is_never_reached(self):
        self.assertIsNone(time_at_height([0, 0, 1.0], [0, 0, 5.0], 10.0))


class SolveLaunchTests(unittest.TestCase):
    def test_a_clear_lands_on_its_target_with_a_realistic_speed(self):
        p0, target = np.array([3.5, 1.4, 2.4]), np.array([1.5, 12.9, 0.0])
        v0 = solve_launch(p0, target, 1.9)
        np.testing.assert_allclose(simulate(p0, v0, [1.9])[0], target, atol=0.01)
        self.assertTrue(15 < np.linalg.norm(v0) < 70)

    def test_non_positive_flight_time_is_refused(self):
        with self.assertRaises(ValueError):
            solve_launch([0, 0, 1], [0, 5, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_shuttle_physics -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'geometry.shuttle_physics'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/shuttle_physics.py
"""Shuttlecock flight: a projectile with quadratic air drag.

    dv/dt = -g e_z - (g / v_t^2) |v| v

v_t is the terminal velocity (about 6.7 m/s for a feather shuttle). Positions are court-frame
metres (z up). Integration is fixed-step RK4 on plain floats, fast enough to call inside an
optimiser (a 2 s flight is ~400 steps).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

GRAVITY = 9.81
FEATHER_TERMINAL_VELOCITY = 6.7
STEP_S = 0.005
LAUNCH_TOLERANCE_M = 0.01

Vector = tuple[float, float, float]


def _drag(terminal_velocity: float | None) -> float:
    return 0.0 if terminal_velocity is None else GRAVITY / terminal_velocity**2


def _acceleration(v: Vector, k: float) -> Vector:
    speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (-k * speed * v[0], -k * speed * v[1], -GRAVITY - k * speed * v[2])


def _rk4(p: Vector, v: Vector, h: float, k: float) -> tuple[Vector, Vector]:
    a1 = _acceleration(v, k)
    v2 = (v[0] + 0.5 * h * a1[0], v[1] + 0.5 * h * a1[1], v[2] + 0.5 * h * a1[2])
    a2 = _acceleration(v2, k)
    v3 = (v[0] + 0.5 * h * a2[0], v[1] + 0.5 * h * a2[1], v[2] + 0.5 * h * a2[2])
    a3 = _acceleration(v3, k)
    v4 = (v[0] + h * a3[0], v[1] + h * a3[1], v[2] + h * a3[2])
    a4 = _acceleration(v4, k)
    p_next = tuple(p[i] + h / 6 * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i]) for i in range(3))
    v_next = tuple(v[i] + h / 6 * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]) for i in range(3))
    return p_next, v_next  # type: ignore[return-value]


def _state(p0, v0) -> tuple[Vector, Vector]:
    p = tuple(float(c) for c in np.asarray(p0, dtype=np.float64).reshape(3))
    v = tuple(float(c) for c in np.asarray(v0, dtype=np.float64).reshape(3))
    return p, v  # type: ignore[return-value]


def simulate(p0, v0, times, terminal_velocity: float | None = FEATHER_TERMINAL_VELOCITY) -> np.ndarray:
    """Positions (N, 3) at non-negative, non-decreasing times after launch."""
    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(times_arr) and (times_arr[0] < 0 or np.any(np.diff(times_arr) < 0)):
        raise ValueError("times must be non-negative and non-decreasing")
    k = _drag(terminal_velocity)
    p, v = _state(p0, v0)
    t = 0.0
    out = np.empty((len(times_arr), 3))
    for index, target in enumerate(times_arr):
        while t < target - 1e-12:
            h = min(STEP_S, target - t)
            p, v = _rk4(p, v, h, k)
            t += h
        out[index] = p
    return out


def time_at_height(
    p0, v0, height: float, terminal_velocity: float | None = FEATHER_TERMINAL_VELOCITY, max_time: float = 6.0
) -> float | None:
    """First time the shuttle passes `height` while descending, or None within max_time."""
    k = _drag(terminal_velocity)
    p, v = _state(p0, v0)
    t = 0.0
    while t < max_time:
        p_next, v_next = _rk4(p, v, STEP_S, k)
        if p[2] > height >= p_next[2] and v_next[2] < 0:
            low, high = 0.0, STEP_S
            for _ in range(40):
                middle = (low + high) / 2
                if _rk4(p, v, middle, k)[0][2] > height:
                    low = middle
                else:
                    high = middle
            return t + high
        p, v, t = p_next, v_next, t + STEP_S
    return None


def solve_launch(p0, target, flight_time: float, terminal_velocity: float | None = FEATHER_TERMINAL_VELOCITY) -> np.ndarray:
    """Launch velocity that puts the shuttle at `target` exactly `flight_time` seconds after leaving p0."""
    if flight_time <= 0:
        raise ValueError("flight_time must be positive")
    start = np.asarray(p0, dtype=np.float64).reshape(3)
    goal = np.asarray(target, dtype=np.float64).reshape(3)
    guess = (goal - start) / flight_time + np.array([0.0, 0.0, 0.5 * GRAVITY * flight_time])
    result = least_squares(lambda v: simulate(start, v, [flight_time], terminal_velocity)[0] - goal, guess, method="lm")
    miss = float(np.max(np.abs(result.fun)))
    if miss > LAUNCH_TOLERANCE_M:
        raise ValueError(f"no launch reaches the target in {flight_time} s (miss {miss:.3f} m)")
    return result.x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_shuttle_physics -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/geometry/shuttle_physics.py worker/test_shuttle_physics.py
git commit -m "feat(worker): add shuttlecock flight physics with quadratic drag"
```

---

### Task 2: Rally builder

**Files:**
- Create: `worker/simulation/__init__.py`
- Create: `worker/simulation/rally.py`
- Test: `worker/test_rally.py`

**Interfaces:**
- Consumes: `simulate`, `solve_launch`, `time_at_height`, `STEP_S`, `FEATHER_TERMINAL_VELOCITY`; court constants.
- Produces: frozen dataclasses `Shot(kind, target: (x, y), flight_time, receive_height | None)`, `Contact(time, position: (x, y, z), hitter: "near"|"far", kind)`, `Flight(start_time, end_time, p0, v0)`, `Rally(contacts, flights, landing: (x, y), end_time, terminal_velocity)` with `Rally.shuttle_at(t) -> np.ndarray | None`; `build_rally(serve_position, shots, start_time=0.5, terminal_velocity=6.7) -> Rally` (raises `ValueError` mentioning "net" or "receive height" for impossible shots); `canned_rally() -> Rally`; `player_positions(rally, t) -> {"near": (x, y), "far": (x, y)}` (body centre, offset from the racket contact point).

- [ ] **Step 1: Write the failing test**

```python
# worker/test_rally.py
import unittest

import numpy as np

from geometry.court_model import NET_POST_HEIGHT_M, NET_Y_M, SINGLES_WIDTH_M, COURT_LENGTH_M
from geometry.shuttle_physics import simulate
from simulation.rally import BODY_OFFSET_M, Shot, build_rally, canned_rally, player_positions


def net_crossing_height(flight, terminal_velocity):
    times = np.arange(0.0, flight.end_time - flight.start_time, 0.002)
    xyz = simulate(flight.p0, flight.v0, times, terminal_velocity)
    side = np.sign(xyz[:, 1] - NET_Y_M)
    index = int(np.where(np.diff(side) != 0)[0][0])
    return float(xyz[index, 2])


class CannedRallyTests(unittest.TestCase):
    def setUp(self):
        self.rally = canned_rally()

    def test_seven_contacts_alternate_from_the_near_server(self):
        hitters = [contact.hitter for contact in self.rally.contacts]
        self.assertEqual(hitters, ["near", "far", "near", "far", "near", "far", "near"])
        kinds = [contact.kind for contact in self.rally.contacts]
        self.assertEqual(kinds, ["serve", "clear", "drop", "lift", "smash", "block", "net"])
        times = [contact.time for contact in self.rally.contacts]
        self.assertEqual(times, sorted(times))
        self.assertTrue(6.0 < self.rally.end_time < 10.0)

    def test_every_flight_clears_the_net(self):
        for flight in self.rally.flights:
            self.assertGreater(net_crossing_height(flight, self.rally.terminal_velocity), NET_POST_HEIGHT_M)

    def test_contacts_are_on_the_hitters_side(self):
        for contact in self.rally.contacts:
            on_near_side = contact.position[1] < NET_Y_M
            self.assertEqual(on_near_side, contact.hitter == "near", contact)

    def test_the_rally_ends_with_a_landing_inside_the_far_half(self):
        x, y = self.rally.landing
        self.assertTrue(0 <= x <= SINGLES_WIDTH_M and NET_Y_M < y <= COURT_LENGTH_M)
        np.testing.assert_allclose(self.rally.shuttle_at(self.rally.end_time), [x, y, 0.0], atol=0.01)

    def test_shuttle_position_is_continuous_and_starts_at_the_serve(self):
        self.assertIsNone(self.rally.shuttle_at(0.0))
        self.assertIsNone(self.rally.shuttle_at(self.rally.end_time + 0.1))
        for contact in self.rally.contacts:
            np.testing.assert_allclose(self.rally.shuttle_at(contact.time), contact.position, atol=1e-6)
            before = self.rally.shuttle_at(contact.time - 1e-4)
            if before is not None:
                self.assertLess(float(np.linalg.norm(before - np.asarray(contact.position))), 0.01)

    def test_players_stand_behind_their_contact_point(self):
        first = self.rally.contacts[0]
        near = player_positions(self.rally, first.time)["near"]
        self.assertAlmostEqual(near[1], first.position[1] - BODY_OFFSET_M[1])
        self.assertEqual(set(player_positions(self.rally, 3.0)), {"near", "far"})


class BuildRallyRefusalTests(unittest.TestCase):
    def test_a_flat_low_shot_into_the_net_is_refused(self):
        with self.assertRaisesRegex(ValueError, "net"):
            build_rally((2.6, 5.0, 0.6), [Shot("drive", (2.6, 8.0), 0.25, None)])

    def test_a_shot_that_stays_on_its_own_side_is_refused(self):
        with self.assertRaisesRegex(ValueError, "net"):
            build_rally((2.6, 3.0, 1.0), [Shot("clear", (2.6, 0.5), 1.5, None)])

    def test_a_receive_height_above_the_apex_is_refused(self):
        with self.assertRaisesRegex(ValueError, "receive height"):
            build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, 9.0), Shot("clear", (3.5, 0.6), 1.8, None)])

    def test_only_the_last_shot_may_land(self):
        with self.assertRaises(ValueError):
            build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, None), Shot("clear", (3.5, 0.6), 1.8, None)])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_rally -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'simulation'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/simulation/__init__.py
"""Synthetic badminton clips with exact ground truth, for testing perception."""
```

```python
# worker/simulation/rally.py
"""A singles rally as a sequence of physically simulated shots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.court_model import COURT_LENGTH_M, NET_POST_HEIGHT_M, NET_Y_M, SINGLES_WIDTH_M
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, STEP_S, simulate, solve_launch, time_at_height

# Conservative: the tape is 1.55 m at the posts and 1.524 m at the centre.
NET_CLEARANCE_M = NET_POST_HEIGHT_M
# The body stands behind and to the side of the racket contact point (near player; mirrored for far).
BODY_OFFSET_M = (-0.3, 0.5)
DEFAULT_BODY = {"near": (SINGLES_WIDTH_M / 2, 3.2), "far": (SINGLES_WIDTH_M / 2, COURT_LENGTH_M - 3.2)}


@dataclass(frozen=True)
class Shot:
    kind: str
    target: tuple[float, float]
    flight_time: float
    receive_height: float | None  # None only for the rally's last shot, which lands


@dataclass(frozen=True)
class Contact:
    time: float
    position: tuple[float, float, float]
    hitter: str  # "near" | "far"
    kind: str


@dataclass(frozen=True)
class Flight:
    start_time: float
    end_time: float
    p0: tuple[float, float, float]
    v0: tuple[float, float, float]


@dataclass(frozen=True)
class Rally:
    contacts: tuple[Contact, ...]
    flights: tuple[Flight, ...]
    landing: tuple[float, float]
    end_time: float
    terminal_velocity: float

    def shuttle_at(self, time: float) -> np.ndarray | None:
        for flight in self.flights:
            if flight.start_time <= time <= flight.end_time:
                return simulate(flight.p0, flight.v0, [time - flight.start_time], self.terminal_velocity)[0]
        return None


def _check_net(position: np.ndarray, v0: np.ndarray, duration: float, terminal_velocity: float, label: str) -> None:
    times = np.arange(0.0, duration + STEP_S, STEP_S)
    xyz = simulate(position, v0, times, terminal_velocity)
    side = np.sign(xyz[:, 1] - NET_Y_M)
    crossings = np.where(np.diff(side) != 0)[0]
    if not len(crossings):
        raise ValueError(f"{label} does not cross the net before it is played")
    i = int(crossings[0])
    fraction = (NET_Y_M - xyz[i, 1]) / (xyz[i + 1, 1] - xyz[i, 1])
    height = xyz[i, 2] + fraction * (xyz[i + 1, 2] - xyz[i, 2])
    if height <= NET_CLEARANCE_M:
        raise ValueError(f"{label} hits the net ({height:.2f} m at the net plane)")


def build_rally(
    serve_position: tuple[float, float, float],
    shots: list[Shot],
    start_time: float = 0.5,
    terminal_velocity: float = FEATHER_TERMINAL_VELOCITY,
) -> Rally:
    if not shots:
        raise ValueError("a rally needs at least one shot")
    if any(shot.receive_height is None for shot in shots[:-1]) or shots[-1].receive_height is not None:
        raise ValueError("every shot but the last needs a receive height; the last shot lands")
    position = np.asarray(serve_position, dtype=np.float64)
    hitter = "near" if position[1] < NET_Y_M else "far"
    time = start_time
    contacts: list[Contact] = []
    flights: list[Flight] = []
    for index, shot in enumerate(shots):
        label = f"shot {index} ({shot.kind})"
        target = np.array([shot.target[0], shot.target[1], 0.0])
        if (target[1] - NET_Y_M) * (position[1] - NET_Y_M) >= 0:
            raise ValueError(f"{label} does not cross the net: its target is on the hitter's side")
        v0 = solve_launch(position, target, shot.flight_time, terminal_velocity)
        if shot.receive_height is None:
            duration = shot.flight_time
        else:
            duration = time_at_height(position, v0, shot.receive_height, terminal_velocity, max_time=shot.flight_time)
            if duration is None:
                raise ValueError(f"{label} never comes down to the receive height {shot.receive_height} m")
        _check_net(position, v0, duration, terminal_velocity, label)
        contacts.append(Contact(time, tuple(float(c) for c in position), hitter, shot.kind))
        flights.append(Flight(time, time + duration, tuple(float(c) for c in position), tuple(float(c) for c in v0)))
        position = simulate(position, v0, [duration], terminal_velocity)[0]
        time += duration
        hitter = "far" if hitter == "near" else "near"
    return Rally(tuple(contacts), tuple(flights), (float(position[0]), float(position[1])), time, terminal_velocity)


def canned_rally() -> Rally:
    """A realistic club-level singles rally (~8 s): serve, clear, drop, lift, smash, block, net."""
    return build_rally(
        (3.3, 3.6, 1.0),
        [
            Shot("serve", (1.5, 12.9), 1.9, 2.4),
            Shot("clear", (3.5, 0.6), 1.8, 2.4),
            Shot("drop", (2.0, 8.3), 1.2, 0.5),
            Shot("lift", (1.2, 0.7), 1.7, 2.5),
            Shot("smash", (4.0, 11.6), 0.75, 1.0),
            Shot("block", (3.2, 5.0), 1.1, 0.6),
            Shot("net", (2.6, 7.8), 1.1, None),
        ],
    )


def player_positions(rally: Rally, time: float) -> dict[str, tuple[float, float]]:
    """Body centre of each player: at their contact points (offset behind the racket), moving linearly between them."""
    positions: dict[str, tuple[float, float]] = {}
    for side, sign in (("near", 1.0), ("far", -1.0)):
        own = [contact for contact in rally.contacts if contact.hitter == side]
        if not own:
            positions[side] = DEFAULT_BODY[side]
            continue
        times = [contact.time for contact in own]
        xs = [contact.position[0] + sign * BODY_OFFSET_M[0] for contact in own]
        ys = [contact.position[1] - sign * BODY_OFFSET_M[1] for contact in own]
        positions[side] = (float(np.interp(time, times, xs)), float(np.interp(time, times, ys)))
    return positions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_rally -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/simulation/__init__.py worker/simulation/rally.py worker/test_rally.py
git commit -m "feat(sim): add physically simulated singles rally builder"
```

---

### Task 3: Camera paths

**Files:**
- Create: `worker/simulation/camera_path.py`
- Test: `worker/test_camera_path.py`

**Interfaces:**
- Consumes: `Camera` (`rotation`, `centre`, constructor).
- Produces: `tripod(camera, frames) -> list[Camera]`; `stable_handheld(camera, frames, fps, seed=0, rotation_deg=0.3, translation_m=0.015) -> list[Camera]` — each rotation axis (camera frame) and translation axis (world) is a smooth signal bounded by the given amplitude.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_camera_path.py
import math
import unittest

import cv2
import numpy as np

from geometry.camera import Camera
from simulation.camera_path import stable_handheld, tripod

BASE = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, (1920, 1080))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(cv2.Rodrigues(a @ b.T)[0])))


class CameraPathTests(unittest.TestCase):
    def test_tripod_never_moves(self):
        path = tripod(BASE, 5)
        self.assertEqual(len(path), 5)
        self.assertTrue(all(camera is BASE for camera in path))

    def test_handheld_is_deterministic_per_seed(self):
        a = stable_handheld(BASE, 30, 60.0, seed=1)
        b = stable_handheld(BASE, 30, 60.0, seed=1)
        c = stable_handheld(BASE, 30, 60.0, seed=2)
        np.testing.assert_array_equal(a[10].rvec, b[10].rvec)
        self.assertFalse(np.allclose(a[10].rvec, c[10].rvec))

    def test_shake_stays_within_the_limits_but_does_move(self):
        path = stable_handheld(BASE, 600, 60.0, seed=3, rotation_deg=0.3, translation_m=0.015)
        angles = [angle_deg(camera.rotation, BASE.rotation) for camera in path]
        offsets = [float(np.linalg.norm(camera.centre - BASE.centre)) for camera in path]
        self.assertLessEqual(max(angles), 0.3 * math.sqrt(3) + 1e-9)
        self.assertLessEqual(max(offsets), 0.015 * math.sqrt(3) + 1e-9)
        self.assertGreater(max(angles), 0.06)
        self.assertGreater(max(offsets), 0.003)

    def test_motion_is_smooth_between_frames(self):
        path = stable_handheld(BASE, 600, 60.0, seed=4)
        steps = [angle_deg(path[i + 1].rotation, path[i].rotation) for i in range(len(path) - 1)]
        self.assertLess(max(steps), 0.07)

    def test_intrinsics_are_unchanged(self):
        camera = stable_handheld(BASE, 3, 60.0)[2]
        self.assertEqual((camera.focal_px, camera.cx, camera.cy, camera.k1), (BASE.focal_px, BASE.cx, BASE.cy, BASE.k1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_camera_path -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'simulation.camera_path'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/simulation/camera_path.py
"""Camera motion for synthetic clips: a tripod, or a steadily held landscape phone."""
from __future__ import annotations

import cv2
import numpy as np

from geometry.camera import Camera

SHAKE_FREQUENCIES_HZ = (0.35, 0.9, 1.7)


def tripod(camera: Camera, frames: int) -> list[Camera]:
    return [camera] * frames


def _shake(frames: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth signal in [-1, 1]: mean of sinusoids with random phase and jittered frequency."""
    t = np.arange(frames) / fps
    signal = np.zeros(frames)
    for frequency in SHAKE_FREQUENCIES_HZ:
        signal += np.sin(2 * np.pi * frequency * rng.uniform(0.8, 1.2) * t + rng.uniform(0, 2 * np.pi))
    return signal / len(SHAKE_FREQUENCIES_HZ)


def stable_handheld(
    camera: Camera,
    frames: int,
    fps: float,
    seed: int = 0,
    rotation_deg: float = 0.3,
    translation_m: float = 0.015,
) -> list[Camera]:
    """A steadily held phone: small smooth rotation (camera axes) and translation (world axes)."""
    rng = np.random.default_rng(seed)
    rotation = np.stack([_shake(frames, fps, rng) for _ in range(3)], axis=1) * np.radians(rotation_deg)
    translation = np.stack([_shake(frames, fps, rng) for _ in range(3)], axis=1) * translation_m
    base_rotation, base_centre = camera.rotation, camera.centre
    path: list[Camera] = []
    for delta_rotation, delta_centre in zip(rotation, translation):
        rotation_matrix = cv2.Rodrigues(delta_rotation.reshape(3, 1))[0] @ base_rotation
        centre = base_centre + delta_centre
        path.append(
            Camera(
                camera.focal_px,
                camera.cx,
                camera.cy,
                cv2.Rodrigues(rotation_matrix)[0].ravel(),
                -rotation_matrix @ centre,
                camera.k1,
            )
        )
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_camera_path -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/simulation/camera_path.py worker/test_camera_path.py
git commit -m "feat(sim): add tripod and stable-handheld camera paths"
```

---

### Task 4: Audio track

**Files:**
- Create: `worker/simulation/audio.py`
- Test: `worker/test_sim_audio.py`

**Interfaces:**
- Produces: `SAMPLE_RATE = 48000`, `HIT_DURATION_S`, `DISTRACTOR_SPACING_S`, `render_audio(contact_times, duration_s, sample_rate=48000, seed=0, distractors=0, offset_s=0.0, hit_amplitude=0.5, noise_amplitude=0.01) -> (np.ndarray float32 mono, list[float] distractor_times)`. A hit starts exactly at `time + offset_s`; distractors are quarter-amplitude and at least `DISTRACTOR_SPACING_S` from every hit.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_sim_audio.py
import unittest

import numpy as np

from simulation.audio import DISTRACTOR_SPACING_S, SAMPLE_RATE, render_audio

HITS = [0.5, 1.4, 2.2]


def peak(samples: np.ndarray, start_s: float, length_s: float = 0.005) -> float:
    a = int(start_s * SAMPLE_RATE)
    return float(np.max(np.abs(samples[a : a + int(length_s * SAMPLE_RATE)])))


class RenderAudioTests(unittest.TestCase):
    def test_length_type_and_range(self):
        samples, _ = render_audio(HITS, 3.0)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 3 * SAMPLE_RATE)
        self.assertLessEqual(float(np.max(np.abs(samples))), 1.0)

    def test_each_hit_is_a_loud_onset_over_quiet_background(self):
        samples, _ = render_audio(HITS, 3.0)
        for hit in HITS:
            self.assertGreater(peak(samples, hit), 0.2)
            self.assertLess(peak(samples, hit - 0.02, 0.015), 0.06)
        quiet = samples[int(0.1 * SAMPLE_RATE) : int(0.4 * SAMPLE_RATE)]
        self.assertLess(float(np.sqrt(np.mean(quiet**2))), 0.02)

    def test_offset_shifts_every_hit(self):
        samples, _ = render_audio(HITS, 3.0, offset_s=0.05)
        for hit in HITS:
            self.assertGreater(peak(samples, hit + 0.05), 0.2)
            self.assertLess(peak(samples, hit - 0.01, 0.015), 0.06)

    def test_distractors_are_quieter_and_away_from_hits(self):
        samples, distractors = render_audio(HITS, 3.0, distractors=3, seed=5)
        self.assertEqual(len(distractors), 3)
        for time in distractors:
            self.assertTrue(all(abs(time - hit) >= DISTRACTOR_SPACING_S for hit in HITS))
            self.assertTrue(0.05 < peak(samples, time) < 0.2)

    def test_same_seed_same_audio(self):
        a, _ = render_audio(HITS, 3.0, seed=9, distractors=2)
        b, _ = render_audio(HITS, 3.0, seed=9, distractors=2)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_sim_audio -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'simulation.audio'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/simulation/audio.py
"""Audio for synthetic clips: background noise plus a racket 'thwack' at every contact."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48_000
HIT_DURATION_S = 0.03
HIT_DECAY_S = 0.006
DISTRACTOR_SPACING_S = 0.15
DISTRACTOR_LEVEL = 0.25


def _hit_sound(rng: np.random.Generator, sample_rate: int, amplitude: float) -> np.ndarray:
    count = int(HIT_DURATION_S * sample_rate)
    t = np.arange(count) / sample_rate
    burst = rng.normal(size=count) * np.exp(-t / HIT_DECAY_S)
    sos = butter(4, [2000, 9000], btype="bandpass", fs=sample_rate, output="sos")
    sound = sosfilt(sos, burst)
    return (amplitude * sound / np.max(np.abs(sound))).astype(np.float32)


def render_audio(
    contact_times: list[float],
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 0,
    distractors: int = 0,
    offset_s: float = 0.0,
    hit_amplitude: float = 0.5,
    noise_amplitude: float = 0.01,
) -> tuple[np.ndarray, list[float]]:
    """Mono float32 track and the times of the distractor hits (a neighbouring court)."""
    rng = np.random.default_rng(seed)
    samples = (noise_amplitude * rng.normal(size=int(round(duration_s * sample_rate)))).astype(np.float32)

    def place(time: float, amplitude: float) -> None:
        start = int(round(time * sample_rate))
        sound = _hit_sound(rng, sample_rate, amplitude)
        if 0 <= start < len(samples):
            end = min(len(samples), start + len(sound))
            samples[start:end] += sound[: end - start]

    for time in contact_times:
        place(time + offset_s, hit_amplitude)
    heard = [time + offset_s for time in contact_times]
    distractor_times: list[float] = []
    attempts = 0
    while len(distractor_times) < distractors and attempts < 10_000:
        attempts += 1
        candidate = float(rng.uniform(0.0, duration_s - HIT_DURATION_S))
        if all(abs(candidate - other) >= DISTRACTOR_SPACING_S for other in heard + distractor_times):
            distractor_times.append(candidate)
    for time in sorted(distractor_times):
        place(time, hit_amplitude * DISTRACTOR_LEVEL)
    return np.clip(samples, -1.0, 1.0), sorted(distractor_times)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_sim_audio -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/simulation/audio.py worker/test_sim_audio.py
git commit -m "feat(sim): add synthetic rally audio with contact hits and distractors"
```

---

### Task 5: Frame renderer

**Files:**
- Create: `worker/simulation/render.py`
- Test: `worker/test_render.py`

**Interfaces:**
- Consumes: `Camera` (`project`, `rotation`, `tvec`, `focal_px`), `court_lines`, court constants.
- Produces: `render_frame(camera, image_size, shuttle=None, shuttle_previous=None, players=(), noise_sigma=2.0, seed=0) -> np.ndarray (H, W, 3) uint8 BGR`; colour constants `FLOOR_BGR`, `LINE_BGR`, `PLAYER_BGR`, `SHUTTLE_BGR`. Objects behind the camera are skipped.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_render.py
import unittest

import numpy as np

from geometry.camera import Camera
from simulation.render import render_frame

SIZE = (640, 360)
CAMERA = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 / 3, SIZE)


def pixel_at(image: np.ndarray, world) -> np.ndarray:
    x, y = CAMERA.project(np.array([world], dtype=float))[0]
    return image[int(round(y)), int(round(x))].astype(int)


class RenderFrameTests(unittest.TestCase):
    def test_frame_shape_and_type(self):
        image = render_frame(CAMERA, SIZE)
        self.assertEqual(image.shape, (360, 640, 3))
        self.assertEqual(image.dtype, np.uint8)

    # At 640x360 the far baseline is ~0.2 px wide (anti-aliased, not white), so these tests
    # use nearer lines that span whole pixels.
    def test_court_lines_are_white_and_the_floor_is_green(self):
        image = render_frame(CAMERA, SIZE, noise_sigma=0)
        self.assertTrue(np.all(pixel_at(image, (1.3, 0.76, 0.0)) > 170))
        blue, green, red = pixel_at(image, (1.3, 3.0, 0.0))
        self.assertGreater(green, red + 30)
        self.assertGreater(green, blue + 30)

    def test_the_net_tape_is_drawn(self):
        image = render_frame(CAMERA, SIZE, noise_sigma=0)
        self.assertTrue(np.all(pixel_at(image, (1.0, 6.7, 1.50)) > 200))

    def test_a_player_hides_the_line_behind_them(self):
        clear = render_frame(CAMERA, SIZE, noise_sigma=0)
        hidden = render_frame(CAMERA, SIZE, players=[(1.5, 3.0)], noise_sigma=0)
        self.assertTrue(np.all(pixel_at(clear, (1.3, 4.72, 0.0)) > 170))
        self.assertLess(int(pixel_at(hidden, (1.3, 4.72, 0.0)).sum()), 300)

    def test_the_shuttle_is_a_bright_blob_where_it_projects(self):
        without = render_frame(CAMERA, SIZE, noise_sigma=0)
        with_shuttle = render_frame(CAMERA, SIZE, shuttle=(2.59, 8.0, 2.0), shuttle_previous=(2.59, 7.8, 2.1), noise_sigma=0)
        self.assertTrue(np.all(pixel_at(with_shuttle, (2.59, 8.0, 2.0)) > 220))
        self.assertFalse(np.all(pixel_at(without, (2.59, 8.0, 2.0)) > 220))

    def test_objects_behind_the_camera_are_skipped(self):
        image = render_frame(CAMERA, SIZE, shuttle=(2.59, -10.0, 1.0), shuttle_previous=(2.59, -9.0, 1.0), players=[(2.59, -8.0)])
        self.assertEqual(image.shape, (360, 640, 3))

    def test_noise_is_seeded(self):
        a = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=3)
        b = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=3)
        c = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_render -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'simulation.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/simulation/render.py
"""Draw one synthetic frame of a singles court as a phone would see it."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from geometry.camera import Camera
from geometry.court_model import COLUMNS, COURT_LENGTH_M, NET_CENTRE_HEIGHT_M, NET_POST_HEIGHT_M, NET_Y_M, court_lines

FLOOR_BGR = (64, 112, 44)
LINE_BGR = (232, 232, 232)
WALL_TOP_BGR = np.array([70.0, 70.0, 75.0])
WALL_BOTTOM_BGR = np.array([120.0, 118.0, 115.0])
NET_BGR = (40, 40, 40)
TAPE_BGR = (240, 240, 240)
POST_BGR = (30, 30, 160)
PLAYER_BGR = (60, 45, 35)
SHUTTLE_BGR = (250, 250, 250)
LINE_WIDTH_M = 0.04
LINE_SEGMENT_M = 0.25
NET_DEPTH_M = 0.76
TAPE_DEPTH_M = 0.075
NET_OPACITY = 0.35
PLAYER_RADIUS_M = 0.25
PLAYER_HEIGHT_M = 1.75
SHUTTLE_SIZE_M = 0.065
_SHIFT = 4
_SCALE = 1 << _SHIFT
_MAX_PIXEL = 1e5


def _to_fixed(pixels: np.ndarray) -> np.ndarray | None:
    """Fixed-point int32 pixels for OpenCV's sub-pixel drawing; None if anything is behind or absurdly far."""
    if not np.isfinite(pixels).all() or np.abs(pixels).max() > _MAX_PIXEL:
        return None
    return np.round(pixels * _SCALE).astype(np.int32)


def _fill(image: np.ndarray, camera: Camera, corners, color, antialias: bool = True) -> None:
    fixed = _to_fixed(camera.project(np.asarray(corners, dtype=np.float64)))
    if fixed is not None:
        cv2.fillConvexPoly(image, fixed, color, lineType=cv2.LINE_AA if antialias else cv2.LINE_8, shift=_SHIFT)


def _depth(camera: Camera, point) -> float:
    return float((camera.rotation @ np.asarray(point, dtype=np.float64) + np.asarray(camera.tvec, dtype=np.float64).reshape(3))[2])


def _background(height: int, width: int) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, height)[:, None]
    rows = (WALL_TOP_BGR[None, :] * (1 - ramp) + WALL_BOTTOM_BGR[None, :] * ramp).astype(np.uint8)
    return np.repeat(rows[:, None, :], width, axis=1)


def _draw_floor(image: np.ndarray, camera: Camera) -> None:
    # 1 m tiles with a faint checker texture; no anti-aliasing so tiles meet without seams.
    for x0 in np.arange(-3.0, 8.0, 1.0):
        for y0 in np.arange(-6.0, COURT_LENGTH_M + 6.0, 1.0):
            shade = 4 * ((int(x0) + int(y0)) % 3 - 1)
            color = tuple(int(c + shade) for c in FLOOR_BGR)
            corners = [[x0, y0, 0.0], [x0 + 1, y0, 0.0], [x0 + 1, y0 + 1, 0.0], [x0, y0 + 1, 0.0]]
            _fill(image, camera, corners, color, antialias=False)


def _draw_lines(image: np.ndarray, camera: Camera) -> None:
    half = LINE_WIDTH_M / 2
    for _, start, end in court_lines():
        length = float(np.linalg.norm(end - start))
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0], 0.0]) * half
        steps = max(1, int(np.ceil(length / LINE_SEGMENT_M)))
        for i in range(steps):
            a = start + (end - start) * (i / steps)
            b = start + (end - start) * ((i + 1) / steps)
            _fill(image, camera, [a - normal, b - normal, b + normal, a + normal], LINE_BGR)


def _draw_net(image: np.ndarray, camera: Camera) -> None:
    left, centre, right = COLUMNS["dl"], COLUMNS["c"], COLUMNS["dr"]
    spans = [((left, NET_POST_HEIGHT_M), (centre, NET_CENTRE_HEIGHT_M)), ((centre, NET_CENTRE_HEIGHT_M), (right, NET_POST_HEIGHT_M))]
    mesh = image.copy()
    for (xa, za), (xb, zb) in spans:
        _fill(mesh, camera, [[xa, NET_Y_M, za - TAPE_DEPTH_M], [xb, NET_Y_M, zb - TAPE_DEPTH_M], [xb, NET_Y_M, zb - NET_DEPTH_M], [xa, NET_Y_M, za - NET_DEPTH_M]], NET_BGR)
    cv2.addWeighted(mesh, NET_OPACITY, image, 1 - NET_OPACITY, 0, dst=image)
    for (xa, za), (xb, zb) in spans:
        _fill(image, camera, [[xa, NET_Y_M, za], [xb, NET_Y_M, zb], [xb, NET_Y_M, zb - TAPE_DEPTH_M], [xa, NET_Y_M, za - TAPE_DEPTH_M]], TAPE_BGR)
    for x in (left, right):
        ends = _to_fixed(camera.project(np.array([[x, NET_Y_M, 0.0], [x, NET_Y_M, NET_POST_HEIGHT_M]])))
        if ends is not None:
            thickness = max(1, int(round(0.04 * camera.focal_px / max(_depth(camera, (x, NET_Y_M, 0.8)), 0.1))))
            cv2.line(image, tuple(int(v) for v in ends[0]), tuple(int(v) for v in ends[1]), POST_BGR, thickness, cv2.LINE_AA, _SHIFT)


def _draw_player(image: np.ndarray, camera: Camera, xy: tuple[float, float]) -> None:
    angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    ring = np.column_stack([xy[0] + PLAYER_RADIUS_M * np.cos(angles), xy[1] + PLAYER_RADIUS_M * np.sin(angles)])
    points = np.vstack([np.column_stack([ring, np.zeros(16)]), np.column_stack([ring, np.full(16, PLAYER_HEIGHT_M)])])
    fixed = _to_fixed(camera.project(points))
    if fixed is not None:
        cv2.fillConvexPoly(image, cv2.convexHull(fixed), PLAYER_BGR, lineType=cv2.LINE_AA, shift=_SHIFT)


def _draw_shuttle(image: np.ndarray, camera: Camera, position, previous) -> None:
    centre = _to_fixed(camera.project(np.asarray([position], dtype=np.float64)))
    depth = _depth(camera, position)
    if centre is None or depth <= 0:
        return
    radius = max(1.0, SHUTTLE_SIZE_M * camera.focal_px / depth / 2)
    if previous is not None:
        start = _to_fixed(camera.project(np.asarray([previous], dtype=np.float64)))
        if start is not None:
            thickness = max(2, int(round(2 * radius)))
            cv2.line(image, tuple(int(v) for v in start[0]), tuple(int(v) for v in centre[0]), SHUTTLE_BGR, thickness, cv2.LINE_AA, _SHIFT)
    cv2.circle(image, tuple(int(v) for v in centre[0]), int(round(radius * _SCALE)), SHUTTLE_BGR, -1, cv2.LINE_AA, _SHIFT)


def render_frame(
    camera: Camera,
    image_size: tuple[int, int],
    shuttle=None,
    shuttle_previous=None,
    players=(),
    noise_sigma: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """BGR frame: walls, floor, lines, then net/players/shuttle far-to-near, then sensor noise."""
    width, height = image_size
    image = _background(height, width).copy()
    _draw_floor(image, camera)
    _draw_lines(image, camera)
    drawables: list[tuple[float, Callable[[], None]]] = [
        (_depth(camera, (COLUMNS["c"], NET_Y_M, 1.0)), lambda: _draw_net(image, camera)),
    ]
    for xy in players:
        drawables.append((_depth(camera, (xy[0], xy[1], PLAYER_HEIGHT_M / 2)), lambda xy=xy: _draw_player(image, camera, xy)))
    if shuttle is not None:
        drawables.append((_depth(camera, shuttle), lambda: _draw_shuttle(image, camera, shuttle, shuttle_previous)))
    for _, draw in sorted(drawables, key=lambda item: -item[0]):
        draw()
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noisy = image.astype(np.int16) + np.round(rng.normal(0.0, noise_sigma, image.shape)).astype(np.int16)
        image = np.clip(noisy, 0, 255).astype(np.uint8)
    return image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_render -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/simulation/render.py worker/test_render.py
git commit -m "feat(sim): add synthetic court frame renderer"
```

---

### Task 6: Clip assembly, writer and CLI

**Files:**
- Create: `worker/simulation/clip.py`
- Test: `worker/test_clip.py`

**Interfaces:**
- Consumes: `Rally`, `canned_rally`, `player_positions`, `build_rally`, `Shot`; `tripod`, `stable_handheld`; `render_audio`, `SAMPLE_RATE`; `render_frame`; `Camera`.
- Produces: dataclass `SyntheticClip(fps, image_size, cameras, frames, shuttle, rally, audio, sample_rate, audio_offset_s, distractor_times)`; `make_clip(base_camera, image_size, rally=None, fps=60.0, handheld=True, seed=0, tail_s=0.5, audio_offset_s=0.0, distractors=0, noise_sigma=2.0) -> SyntheticClip` (frame `i` is time `i / fps`; `base_camera` must already be built for `image_size`); `truth_dict(clip) -> dict` (JSON-serialisable); `camera_from_truth(entry) -> Camera`; `write_clip(clip, directory) -> dict[str, Path]` writing `video.mp4` (mp4v), `audio.wav` (int16), `truth.json`; `main(argv) -> int` for `python -m simulation.clip`.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_clip.py
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

from geometry.camera import Camera
from simulation.clip import camera_from_truth, main, make_clip, truth_dict, write_clip
from simulation.rally import Shot, build_rally

SIZE = (320, 180)
CAMERA = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 / 6, SIZE)
SERVE_ONLY = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, None)])


class MakeClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=20.0, handheld=True, seed=1, distractors=1)

    def test_frame_count_and_size_follow_the_duration(self):
        expected = math.ceil((SERVE_ONLY.end_time + 0.5) * 20.0)
        self.assertEqual(len(self.clip.frames), expected)
        self.assertEqual(len(self.clip.cameras), expected)
        self.assertEqual(len(self.clip.shuttle), expected)
        self.assertEqual(self.clip.frames[0].shape, (180, 320, 3))

    def test_shuttle_truth_matches_the_rally_at_each_frame_time(self):
        self.assertIsNone(self.clip.shuttle[0])
        self.assertIsNone(self.clip.shuttle[-1])
        frame = 30  # 1.5 s, mid-flight
        np.testing.assert_allclose(self.clip.shuttle[frame], SERVE_ONLY.shuttle_at(frame / 20.0))

    def test_handheld_cameras_move_and_tripod_ones_do_not(self):
        self.assertFalse(np.allclose(self.clip.cameras[0].rvec, self.clip.cameras[10].rvec))
        still = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, handheld=False)
        self.assertTrue(all(camera is CAMERA for camera in still.cameras))

    def test_audio_covers_the_clip(self):
        duration = SERVE_ONLY.end_time + 0.5
        self.assertEqual(len(self.clip.audio), int(round(duration * self.clip.sample_rate)))
        self.assertEqual(len(self.clip.distractor_times), 1)


class TruthTests(unittest.TestCase):
    def test_truth_is_json_and_reconstructs_each_camera(self):
        clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, seed=2)
        truth = json.loads(json.dumps(truth_dict(clip)))
        self.assertEqual(truth["frameCount"], len(clip.frames))
        self.assertEqual(truth["contacts"][0]["kind"], "serve")
        self.assertIsNone(truth["shuttle"][0])
        point = np.array([[1.0, 9.0, 0.0]])
        rebuilt = camera_from_truth(truth["cameras"][7])
        np.testing.assert_allclose(rebuilt.project(point), clip.cameras[7].project(point), atol=1e-6)


class WriteClipTests(unittest.TestCase):
    def test_written_files_read_back(self):
        clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, seed=3)
        with tempfile.TemporaryDirectory() as temp:
            paths = write_clip(clip, Path(temp) / "clip")
            capture = cv2.VideoCapture(str(paths["video"]))
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (180, 320, 3))
                frames += 1
            capture.release()
            self.assertEqual(frames, len(clip.frames))
            rate, samples = wavfile.read(paths["audio"])
            self.assertEqual((rate, len(samples)), (clip.sample_rate, len(clip.audio)))
            self.assertEqual(json.loads(paths["truth"].read_text())["frameCount"], len(clip.frames))

    def test_cli_writes_a_canned_rally(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "cli"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--out", str(out), "--fps", "10", "--width", "160", "--height", "90", "--tripod"])
            self.assertEqual(code, 0)
            truth = json.loads((out / "truth.json").read_text())
            self.assertEqual(len(truth["contacts"]), 7)
            self.assertTrue((out / "video.mp4").is_file() and (out / "audio.wav").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_clip -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'simulation.clip'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/simulation/clip.py
"""Assemble a synthetic clip (frames, cameras, shuttle truth, audio) and write it to disk.

    python -m simulation.clip --out ../synthetic/clip-001 [--fps 60] [--tripod] [--seed 0]

Writes video.mp4 (OpenCV mp4v; readable by the worker, not by browsers), audio.wav and
truth.json. Frame i is time i / fps; every per-frame truth list is indexed the same way.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import wavfile

from geometry.camera import Camera
from simulation.audio import SAMPLE_RATE, render_audio
from simulation.camera_path import stable_handheld, tripod
from simulation.rally import Rally, canned_rally, player_positions
from simulation.render import render_frame


@dataclass
class SyntheticClip:
    fps: float
    image_size: tuple[int, int]
    cameras: list[Camera]
    frames: list[np.ndarray]
    shuttle: list[np.ndarray | None]
    rally: Rally
    audio: np.ndarray
    sample_rate: int
    audio_offset_s: float
    distractor_times: list[float]


def make_clip(
    base_camera: Camera,
    image_size: tuple[int, int],
    rally: Rally | None = None,
    fps: float = 60.0,
    handheld: bool = True,
    seed: int = 0,
    tail_s: float = 0.5,
    audio_offset_s: float = 0.0,
    distractors: int = 0,
    noise_sigma: float = 2.0,
) -> SyntheticClip:
    rally = rally or canned_rally()
    duration = rally.end_time + tail_s
    count = math.ceil(duration * fps)
    cameras = stable_handheld(base_camera, count, fps, seed=seed) if handheld else tripod(base_camera, count)
    exposure = 0.5 / fps
    frames: list[np.ndarray] = []
    shuttle: list[np.ndarray | None] = []
    for index in range(count):
        time = index / fps
        position = rally.shuttle_at(time)
        previous = rally.shuttle_at(max(0.0, time - exposure)) if position is not None else None
        players = list(player_positions(rally, time).values())
        frames.append(render_frame(cameras[index], image_size, position, previous, players, noise_sigma, seed + index))
        shuttle.append(position)
    audio, distractor_times = render_audio(
        [contact.time for contact in rally.contacts], duration, seed=seed, distractors=distractors, offset_s=audio_offset_s
    )
    return SyntheticClip(fps, image_size, cameras, frames, shuttle, rally, audio, SAMPLE_RATE, audio_offset_s, distractor_times)


def _camera_entry(camera: Camera) -> dict[str, Any]:
    return {
        "focalPx": float(camera.focal_px),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "rvec": [float(v) for v in np.asarray(camera.rvec).ravel()],
        "tvec": [float(v) for v in np.asarray(camera.tvec).ravel()],
        "k1": float(camera.k1),
    }


def camera_from_truth(entry: dict[str, Any]) -> Camera:
    return Camera(entry["focalPx"], entry["cx"], entry["cy"], np.array(entry["rvec"]), np.array(entry["tvec"]), entry["k1"])


def truth_dict(clip: SyntheticClip) -> dict[str, Any]:
    return {
        "fps": clip.fps,
        "imageSize": list(clip.image_size),
        "frameCount": len(clip.frames),
        "cameras": [_camera_entry(camera) for camera in clip.cameras],
        "shuttle": [None if position is None else [round(float(v), 5) for v in position] for position in clip.shuttle],
        "contacts": [
            {"timeMs": round(contact.time * 1000, 2), "hitter": contact.hitter, "kind": contact.kind, "position": [round(v, 5) for v in contact.position]}
            for contact in clip.rally.contacts
        ],
        "landing": [round(v, 5) for v in clip.rally.landing],
        "rallyEndMs": round(clip.rally.end_time * 1000, 2),
        "terminalVelocity": clip.rally.terminal_velocity,
        "audio": {
            "sampleRate": clip.sample_rate,
            "offsetMs": round(clip.audio_offset_s * 1000, 3),
            "distractorTimesMs": [round(t * 1000, 2) for t in clip.distractor_times],
        },
    }


def write_clip(clip: SyntheticClip, directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"video": directory / "video.mp4", "audio": directory / "audio.wav", "truth": directory / "truth.json"}
    writer = cv2.VideoWriter(str(paths["video"]), cv2.VideoWriter_fourcc(*"mp4v"), clip.fps, clip.image_size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an mp4v video writer")
    for frame in clip.frames:
        writer.write(frame)
    writer.release()
    wavfile.write(paths["audio"], clip.sample_rate, (np.clip(clip.audio, -1.0, 1.0) * 32767).astype(np.int16))
    paths["truth"].write_text(json.dumps(truth_dict(clip)), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a synthetic singles rally with ground truth.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tripod", action="store_true", help="still camera instead of a steadily held phone")
    parser.add_argument("--distractors", type=int, default=2, help="quieter hits from a neighbouring court")
    parser.add_argument("--audio-offset-ms", type=float, default=0.0)
    args = parser.parse_args(argv)
    size = (args.width, args.height)
    # Behind the near baseline, 3 m up: the recommended tier-A placement; focal ~ 26 mm-equivalent.
    camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 * args.width / 1920, size)
    clip = make_clip(
        camera, size, fps=args.fps, handheld=not args.tripod, seed=args.seed,
        distractors=args.distractors, audio_offset_s=args.audio_offset_ms / 1000,
    )
    paths = write_clip(clip, args.out)
    print(f"wrote {len(clip.frames)} frames, {len(clip.rally.contacts)} contacts -> {paths['video'].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_clip -v`
Expected: PASS (7 tests)

Then render a full-size clip and look at a frame to sanity-check the picture (not a test): `cd worker && python -m simulation.clip --out ../synthetic-preview --fps 60` and open a mid-rally frame.

- [ ] **Step 5: Commit**

```bash
git add worker/simulation/clip.py worker/test_clip.py
git commit -m "feat(sim): assemble, write and render synthetic clips from the CLI"
```

---

## Self-Review

**Spec coverage (Step 1).** Physics with drag, launch solving and descending interception → Task 1. Rally builder with net-clearance refusal and a canned realistic rally → Task 2. Tripod + stable landscape handheld paths → Task 3. Audio with hits, distractors and A/V offset → Task 4. Renderer (floor, 40 mm lines, net/tape/posts, occluding players, motion-blurred shuttle, walls, noise, depth order) → Task 5. Frames + per-frame cameras + shuttle truth + contacts + landing + audio written as mp4v/wav/json without ffmpeg → Task 6.

**Placeholder scan.** None.

**Type consistency.** `Rally.shuttle_at`, `player_positions`, `render_frame(camera, image_size, shuttle, shuttle_previous, players, noise_sigma, seed)`, `render_audio(..., distractors, offset_s)` and `stable_handheld(camera, frames, fps, seed)` are called with the signatures their tasks define.

**Review Focus coverage.** (1) Task 2 refusal tests; (2) Task 5 `test_objects_behind_the_camera_are_skipped`; (3) Task 6 frame-time alignment test; (4) Task 6 write/read-back tests; (5) Tasks 4–5 seed tests.
