# Foundation: Evaluation Harness + Camera Calibration Solver — Implementation Plan


**Goal:** Replace the planar corner-click court mapping with a real camera-calibration solver (focal length, pose, lens term, error in cm, capability tier) and add the evaluation harness that measures every later phase against hand-labelled clips.

**Architecture:** A pure-Python `geometry` package (BWF singles court model → pinhole camera → robust solver → capability tier) and an offline `evaluation` package (event/shot metrics, golden-clip schema, report CLI). The RunPod worker gains an additive `camera` summary in its result; the server schema accepts it. Per-frame handheld tracking (Phase 1b) is a separate plan that builds on `solve_camera` and `Camera`.

**Tech Stack:** Python 3.11+ (`unittest`, numpy ≥1.26 and 2.x, OpenCV 4.10, SciPy 1.13), TypeScript/zod/vitest for the server schema.

**Spec:** `docs/design/specs/2026-10-01-badminton-understanding-design.md` (sections 3, 4-L1, 5, 6, 7 Phase 0 and 1a).

## Global Constraints

- Singles only; court frame: x across from the left singles sideline (0) to the right (5.18 m), y from the near baseline (0) to the far baseline (13.40 m), z up.
- Net posts 1.55 m high on the doubles sidelines (0.46 m outside the singles lines); net tape at centre 1.524 m; short service line 1.98 m from the net; doubles long service line 0.76 m inside the back boundary.
- Code must run on `numpy==1.26.4` (pinned in `worker/requirements.txt`) and numpy 2.x.
- Tests use `unittest` (the worker's existing style), run from the `worker/` directory: `python -m unittest <module> -v`.
- Result-schema changes are additive and optional only; existing contracts (RunPod job, queue, DeepSeek call) are unchanged.
- Tier thresholds (`TIER_*`, `VALIDATED_*`, `APPROX_*`) are provisional constants, tuned later against the golden set.
- `worker/handler.py`, `worker/Dockerfile` and `docs/cv-worker-contract.md` already carry the user's uncommitted edits: edit them, but do **not** stage or commit them without the user's approval.

## Review Focus

1. Fewer than four usable keypoints (near corners out of frame, three taps) → `solve_camera` returns `None`, the adapter returns `None`, the handler behaves exactly as before. Tested in Tasks 4 and 8.
2. Degenerate input — collinear points, four taps on the same pixel, only off-floor points → no exception, no false "validated". Tested in Task 4.
3. NaN/Inf pixel coordinates → ignored, never crash the optimiser. Tested in Task 4.
4. Portrait video (width < height) → same accuracy as landscape. Tested in Task 4 via the portrait scene.
5. Wildly wrong taps → flagged as outliers or tier `unavailable`/`approximate`, never `validated`. Tested in Task 4.

---

### Task 1: Court model

**Files:**
- Create: `worker/geometry/__init__.py`
- Create: `worker/geometry/court_model.py`
- Test: `worker/test_court_model.py`

**Interfaces:**
- Produces: `KEYPOINTS: dict[str, np.ndarray]` (name → `[x, y, z]` metres, 35 entries), `FLOOR_KEYPOINT_NAMES: tuple[str, ...]` (30 floor intersections), `CORNER_LABEL_TO_KEYPOINT: dict[str, str]`, `court_lines() -> list[tuple[str, np.ndarray, np.ndarray]]`, constants `COURT_LENGTH_M`, `SINGLES_WIDTH_M`, `NET_Y_M`, `COLUMNS`, `ROWS`. Keypoint names are `"{row}_{column}"` with rows `back0 long0 short0 short1 long1 back1` and columns `dl sl c sr dr`, plus `post_left_base post_left_top post_right_base post_right_top net_centre_top`.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_court_model.py
import unittest

import numpy as np

from geometry.court_model import (
    CORNER_LABEL_TO_KEYPOINT,
    FLOOR_KEYPOINT_NAMES,
    KEYPOINTS,
    NET_Y_M,
    court_lines,
)


def on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    segment, offset = end - start, point - start
    if np.linalg.norm(np.cross(segment, offset)) > 1e-9:
        return False
    t = float(np.dot(offset, segment) / np.dot(segment, segment))
    return -1e-9 <= t <= 1 + 1e-9


class CourtModelTests(unittest.TestCase):
    def test_bwf_court_dimensions(self):
        self.assertAlmostEqual(KEYPOINTS["back1_dr"][1] - KEYPOINTS["back0_dl"][1], 13.40)
        self.assertAlmostEqual(KEYPOINTS["back0_dr"][0] - KEYPOINTS["back0_dl"][0], 6.10)
        self.assertAlmostEqual(KEYPOINTS["back0_sr"][0] - KEYPOINTS["back0_sl"][0], 5.18)

    def test_service_lines(self):
        self.assertAlmostEqual(KEYPOINTS["short0_c"][1], 4.72)
        self.assertAlmostEqual(KEYPOINTS["short1_c"][1], 8.68)
        self.assertAlmostEqual(KEYPOINTS["short1_c"][1] - KEYPOINTS["short0_c"][1], 2 * 1.98)
        self.assertAlmostEqual(KEYPOINTS["long0_c"][1], 0.76)
        self.assertAlmostEqual(KEYPOINTS["long1_c"][1], 12.64)

    def test_centre_line_is_mid_court(self):
        self.assertAlmostEqual(KEYPOINTS["back0_c"][0], 2.59)

    def test_net_geometry(self):
        self.assertAlmostEqual(KEYPOINTS["post_left_top"][2], 1.55)
        self.assertAlmostEqual(KEYPOINTS["post_right_top"][2], 1.55)
        self.assertAlmostEqual(KEYPOINTS["net_centre_top"][2], 1.524)
        for name in ("post_left_base", "post_left_top", "post_right_base", "post_right_top", "net_centre_top"):
            self.assertAlmostEqual(KEYPOINTS[name][1], NET_Y_M)
        self.assertAlmostEqual(KEYPOINTS["post_left_base"][0], -0.46)
        self.assertAlmostEqual(KEYPOINTS["post_right_base"][0], 5.64)

    def test_keypoint_counts(self):
        self.assertEqual(len(FLOOR_KEYPOINT_NAMES), 30)
        self.assertEqual(len(KEYPOINTS), 35)
        self.assertTrue(all(KEYPOINTS[name][2] == 0.0 for name in FLOOR_KEYPOINT_NAMES))

    def test_every_floor_keypoint_is_a_line_intersection(self):
        lines = court_lines()
        for name in FLOOR_KEYPOINT_NAMES:
            count = sum(1 for _, start, end in lines if on_segment(KEYPOINTS[name], start, end))
            self.assertGreaterEqual(count, 2, name)

    def test_ui_corner_labels_are_the_singles_corners(self):
        self.assertEqual(
            {label: tuple(KEYPOINTS[name][:2]) for label, name in CORNER_LABEL_TO_KEYPOINT.items()},
            {"nearLeft": (0.0, 0.0), "nearRight": (5.18, 0.0), "farRight": (5.18, 13.40), "farLeft": (0.0, 13.40)},
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_court_model -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/__init__.py
"""Camera and court geometry for Optiqen analysis."""
```

```python
# worker/geometry/court_model.py
"""BWF singles court in a metric world frame.

Frame: x runs across the court from the left singles sideline (0) to the right
singles sideline (5.18); y runs along the court from the near baseline (0) to
the far baseline (13.40); z is up from the floor. The doubles sidelines sit
0.46 m outside the singles sidelines and stay visible during singles play, so
they are part of the model.
"""
from __future__ import annotations

import numpy as np

COURT_LENGTH_M = 13.40
SINGLES_WIDTH_M = 5.18
DOUBLES_MARGIN_M = 0.46
NET_Y_M = COURT_LENGTH_M / 2
NET_POST_HEIGHT_M = 1.55
NET_CENTRE_HEIGHT_M = 1.524
SHORT_SERVICE_M = 1.98
LONG_SERVICE_INSET_M = 0.76

COLUMNS: dict[str, float] = {
    "dl": -DOUBLES_MARGIN_M,
    "sl": 0.0,
    "c": SINGLES_WIDTH_M / 2,
    "sr": SINGLES_WIDTH_M,
    "dr": SINGLES_WIDTH_M + DOUBLES_MARGIN_M,
}
ROWS: dict[str, float] = {
    "back0": 0.0,
    "long0": LONG_SERVICE_INSET_M,
    "short0": NET_Y_M - SHORT_SERVICE_M,
    "short1": NET_Y_M + SHORT_SERVICE_M,
    "long1": COURT_LENGTH_M - LONG_SERVICE_INSET_M,
    "back1": COURT_LENGTH_M,
}


def _build_keypoints() -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for row_name, y in ROWS.items():
        for column_name, x in COLUMNS.items():
            points[f"{row_name}_{column_name}"] = np.array([x, y, 0.0])
    left, right = COLUMNS["dl"], COLUMNS["dr"]
    points["post_left_base"] = np.array([left, NET_Y_M, 0.0])
    points["post_left_top"] = np.array([left, NET_Y_M, NET_POST_HEIGHT_M])
    points["post_right_base"] = np.array([right, NET_Y_M, 0.0])
    points["post_right_top"] = np.array([right, NET_Y_M, NET_POST_HEIGHT_M])
    points["net_centre_top"] = np.array([COLUMNS["c"], NET_Y_M, NET_CENTRE_HEIGHT_M])
    return points


KEYPOINTS: dict[str, np.ndarray] = _build_keypoints()
FLOOR_KEYPOINT_NAMES: tuple[str, ...] = tuple(
    name for name, point in KEYPOINTS.items() if point[2] == 0.0 and not name.startswith("post_")
)

# The four singles corners the existing calibration UI asks the player to tap.
CORNER_LABEL_TO_KEYPOINT: dict[str, str] = {
    "nearLeft": "back0_sl",
    "nearRight": "back0_sr",
    "farRight": "back1_sr",
    "farLeft": "back1_sl",
}


def court_lines() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Painted floor lines as (name, start, end) in world metres."""
    left, right = COLUMNS["dl"], COLUMNS["dr"]
    lines: list[tuple[str, np.ndarray, np.ndarray]] = []
    for row_name, y in ROWS.items():
        lines.append((row_name, np.array([left, y, 0.0]), np.array([right, y, 0.0])))
    for column_name, x in COLUMNS.items():
        if column_name == "c":
            continue
        lines.append((f"side_{column_name}", np.array([x, 0.0, 0.0]), np.array([x, COURT_LENGTH_M, 0.0])))
    centre = COLUMNS["c"]
    lines.append(("centre0", np.array([centre, 0.0, 0.0]), np.array([centre, ROWS["short0"], 0.0])))
    lines.append(("centre1", np.array([centre, ROWS["short1"], 0.0]), np.array([centre, COURT_LENGTH_M, 0.0])))
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_court_model -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/geometry/__init__.py worker/geometry/court_model.py worker/test_court_model.py
git commit -m "feat(worker): add BWF singles court model with 35 named 3D keypoints"
```

---

### Task 2: Pinhole camera

**Files:**
- Create: `worker/geometry/camera.py`
- Test: `worker/test_camera.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Camera(focal_px: float, cx: float, cy: float, rvec: np.ndarray, tvec: np.ndarray, k1: float = 0.0)` frozen dataclass with `.rotation`, `.centre`, `Camera.look_at(position, target, focal_px, image_size, k1=0.0)`, `.project(points: (N,3)) -> (N,2)` (NaN behind the camera), `.pixel_to_plane(pixels: (N,2), z=0.0) -> (N,2)` (world x, y; NaN if the ray misses the plane). `rvec`/`tvec` map world → camera (OpenCV: x right, y down, z forward).

- [ ] **Step 1: Write the failing test**

```python
# worker/test_camera.py
import unittest

import numpy as np

from geometry.camera import Camera

SIZE = (1920, 1080)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, SIZE)

    def test_look_at_target_lands_on_the_principal_point(self):
        pixel = self.camera.project(np.array([[2.59, 7.0, 0.0]]))[0]
        np.testing.assert_allclose(pixel, [960.0, 540.0], atol=1e-6)

    def test_centre_is_the_camera_position(self):
        np.testing.assert_allclose(self.camera.centre, [2.59, -3.5, 3.0], atol=1e-9)

    def test_world_right_is_image_right(self):
        left = self.camera.project(np.array([[0.0, 7.0, 0.0]]))[0]
        right = self.camera.project(np.array([[5.18, 7.0, 0.0]]))[0]
        self.assertLess(left[0], right[0])

    def test_floor_round_trip(self):
        floor = np.array([[0.0, 0.0, 0.0], [5.18, 13.4, 0.0], [2.59, 6.7, 0.0], [1.0, 4.0, 0.0]])
        pixels = self.camera.project(floor)
        np.testing.assert_allclose(self.camera.pixel_to_plane(pixels, 0.0), floor[:, :2], atol=1e-6)

    def test_round_trip_with_lens_distortion(self):
        camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, SIZE, k1=-0.08)
        floor = np.array([[0.0, 13.4, 0.0], [5.18, 13.4, 0.0], [0.0, 3.0, 0.0], [5.18, 3.0, 0.0]])
        pixels = camera.project(floor)
        np.testing.assert_allclose(camera.pixel_to_plane(pixels, 0.0), floor[:, :2], atol=1e-6)

    def test_elevated_plane_round_trip(self):
        point = np.array([[1.0, 5.0, 1.5]])
        pixels = self.camera.project(point)
        np.testing.assert_allclose(self.camera.pixel_to_plane(pixels, 1.5), point[:, :2], atol=1e-6)

    def test_point_behind_camera_is_nan(self):
        self.assertTrue(np.isnan(self.camera.project(np.array([[2.59, -10.0, 0.0]]))).all())

    def test_ray_above_the_horizon_misses_the_floor(self):
        self.assertTrue(np.isnan(self.camera.pixel_to_plane(np.array([[960.0, 0.0]]), 0.0)).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_camera -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'geometry.camera'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/camera.py
"""Pinhole camera with one radial-distortion term.

World frame is the court frame from court_model (z up). `rvec`/`tvec` map world
points into the camera frame (OpenCV convention: x right, y down, z forward).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    focal_px: float
    cx: float
    cy: float
    rvec: np.ndarray
    tvec: np.ndarray
    k1: float = 0.0

    @property
    def rotation(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64).reshape(3, 1))[0]

    @property
    def centre(self) -> np.ndarray:
        return -self.rotation.T @ np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @classmethod
    def look_at(
        cls,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        focal_px: float,
        image_size: tuple[int, int],
        k1: float = 0.0,
    ) -> "Camera":
        position_v = np.asarray(position, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - position_v
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        rvec = cv2.Rodrigues(rotation)[0].ravel()
        tvec = -rotation @ position_v
        return cls(focal_px, image_size[0] / 2, image_size[1] / 2, rvec, tvec, k1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2); NaN where the point is behind the camera."""
        world = np.atleast_2d(np.asarray(points, dtype=np.float64))
        cam = world @ self.rotation.T + np.asarray(self.tvec, dtype=np.float64).reshape(3)
        out = np.full((len(world), 2), np.nan)
        front = cam[:, 2] > 1e-6
        x = cam[front, 0] / cam[front, 2]
        y = cam[front, 1] / cam[front, 2]
        scale = 1.0 + self.k1 * (x * x + y * y)
        out[front, 0] = self.focal_px * x * scale + self.cx
        out[front, 1] = self.focal_px * y * scale + self.cy
        return out

    def pixel_to_plane(self, pixels: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Pixels (N, 2) -> world (x, y) on the horizontal plane at height z; NaN if the ray misses it."""
        px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        xd = (px[:, 0] - self.cx) / self.focal_px
        yd = (px[:, 1] - self.cy) / self.focal_px
        x, y = xd.copy(), yd.copy()
        for _ in range(12):
            scale = 1.0 + self.k1 * (x * x + y * y)
            x, y = xd / scale, yd / scale
        rays = np.stack([x, y, np.ones_like(x)], axis=1) @ self.rotation
        centre = self.centre
        out = np.full((len(px), 2), np.nan)
        dz = rays[:, 2]
        valid = np.abs(dz) > 1e-9
        s = np.full(len(px), np.nan)
        s[valid] = (z - centre[2]) / dz[valid]
        hit = valid & (s > 0)
        out[hit] = centre[:2] + s[hit, None] * rays[hit, :2]
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_camera -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/geometry/camera.py worker/test_camera.py
git commit -m "feat(worker): add pinhole camera with projection and plane back-projection"
```

---

### Task 3: Synthetic scenes

**Files:**
- Create: `worker/geometry/synthetic.py`
- Test: `worker/test_synthetic.py`

**Interfaces:**
- Consumes: `Camera.look_at`, `Camera.project`, `KEYPOINTS`, `COLUMNS`, `NET_Y_M`.
- Produces: `LANDSCAPE=(1920,1080)`, `PORTRAIT=(1080,1920)`, `standard_scenes() -> dict[str, tuple[Camera, tuple[int,int]]]` with keys `behind_elevated behind_low side_elevated corner_elevated behind_portrait`, and `observe(camera, image_size, names=None, noise_px=0.0, seed=0) -> dict[str, tuple[float, float]]` (only keypoints that land inside the frame).

- [ ] **Step 1: Write the failing test**

```python
# worker/test_synthetic.py
import unittest

import numpy as np

from geometry.court_model import FLOOR_KEYPOINT_NAMES, KEYPOINTS
from geometry.synthetic import observe, standard_scenes


class SyntheticSceneTests(unittest.TestCase):
    def test_scene_names(self):
        self.assertEqual(
            set(standard_scenes()),
            {"behind_elevated", "behind_low", "side_elevated", "corner_elevated", "behind_portrait"},
        )

    def test_every_scene_sees_enough_floor_keypoints(self):
        for name, (camera, size) in standard_scenes().items():
            floor = [key for key in observe(camera, size) if key in FLOOR_KEYPOINT_NAMES]
            self.assertGreaterEqual(len(floor), 8, name)

    def test_noise_free_observations_match_projection(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size)
        for name, pixel in observed.items():
            expected = camera.project(KEYPOINTS[name][None, :])[0]
            np.testing.assert_allclose(pixel, expected, atol=1e-9)

    def test_observations_stay_inside_the_image(self):
        for camera, size in standard_scenes().values():
            for x, y in observe(camera, size, noise_px=3.0, seed=4).values():
                self.assertTrue(0 <= x < size[0] and 0 <= y < size[1])

    def test_noise_is_deterministic_and_has_the_requested_scale(self):
        camera, size = standard_scenes()["behind_elevated"]
        clean = observe(camera, size)
        first = observe(camera, size, noise_px=1.0, seed=7)
        again = observe(camera, size, noise_px=1.0, seed=7)
        self.assertEqual(first, again)
        diffs = np.array([np.subtract(first[name], clean[name]) for name in first if name in clean])
        rms = float(np.sqrt(np.mean(diffs**2)))
        self.assertGreater(rms, 0.5)
        self.assertLess(rms, 2.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_synthetic -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'geometry.synthetic'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/synthetic.py
"""Synthetic cameras and keypoint observations for tests and evaluation."""
from __future__ import annotations

import numpy as np

from geometry.camera import Camera
from geometry.court_model import COLUMNS, KEYPOINTS, NET_Y_M

LANDSCAPE = (1920, 1080)
PORTRAIT = (1080, 1920)
CENTRE_X = COLUMNS["c"]
FOCAL_PX = 1300.0


def standard_scenes() -> dict[str, tuple[Camera, tuple[int, int]]]:
    """Camera placements a phone user plausibly holds, tripod or hand."""
    aim = (CENTRE_X, NET_Y_M, 0.0)
    return {
        "behind_elevated": (Camera.look_at((CENTRE_X, -3.5, 3.0), (CENTRE_X, 7.0, 0.0), FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "behind_low": (Camera.look_at((CENTRE_X + 0.4, -2.5, 1.5), (CENTRE_X, 7.0, 0.0), FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "side_elevated": (Camera.look_at((-4.5, NET_Y_M, 3.5), aim, FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "corner_elevated": (Camera.look_at((-3.0, -3.0, 3.5), aim, FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "behind_portrait": (Camera.look_at((CENTRE_X, -4.0, 3.2), (CENTRE_X, 6.0, 0.0), FOCAL_PX, PORTRAIT), PORTRAIT),
    }


def observe(
    camera: Camera,
    image_size: tuple[int, int],
    names: list[str] | None = None,
    noise_px: float = 0.0,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Pixel observations of court keypoints that land inside the image."""
    rng = np.random.default_rng(seed)
    selected = names if names is not None else list(KEYPOINTS)
    pixels = camera.project(np.array([KEYPOINTS[name] for name in selected]))
    width, height = image_size
    observed: dict[str, tuple[float, float]] = {}
    for name, pixel in zip(selected, pixels):
        if not np.isfinite(pixel).all():
            continue
        noisy = pixel + rng.normal(0.0, noise_px, size=2) if noise_px > 0 else pixel
        if 0 <= noisy[0] < width and 0 <= noisy[1] < height:
            observed[name] = (float(noisy[0]), float(noisy[1]))
    return observed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_synthetic -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/geometry/synthetic.py worker/test_synthetic.py
git commit -m "feat(worker): add synthetic camera scenes and keypoint observations"
```

---

### Task 4: Calibration solver

**Files:**
- Create: `worker/geometry/calibrate.py`
- Test: `worker/test_calibrate.py`

**Interfaces:**
- Consumes: `Camera`, `KEYPOINTS`, `standard_scenes`, `observe`.
- Produces: `solve_camera(observations: dict[str, tuple[float, float]], image_size: tuple[int, int], focal_prior_px: float | None = None) -> CameraSolution | None` and frozen dataclass `CameraSolution(camera: Camera, tier: str, rms_px: float, floor_rms_cm: float, loo_floor_cm: float | None, redundancy: int, inliers: tuple[str, ...], outliers: tuple[str, ...])`. `tier` is `"validated" | "approximate" | "unavailable"`. `redundancy` = inlier count − 4. `loo_floor_cm` is the leave-one-out floor error (None with fewer than 5 floor inliers). Needs ≥4 non-collinear floor keypoints; returns `None` otherwise.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_calibrate.py
import unittest

import numpy as np

from geometry.calibrate import solve_camera
from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, KEYPOINTS, NET_Y_M, SINGLES_WIDTH_M
from geometry.synthetic import LANDSCAPE, observe, standard_scenes

GRID = np.array(
    [[x, y, 0.0] for x in np.linspace(0, SINGLES_WIDTH_M, 6) for y in np.linspace(0, COURT_LENGTH_M, 12)]
)


def grid_error_cm(true_camera: Camera, solved_camera: Camera) -> np.ndarray:
    estimated = solved_camera.pixel_to_plane(true_camera.project(GRID), 0.0)
    return np.linalg.norm(estimated - GRID[:, :2], axis=1) * 100.0


class CalibrationSolverTests(unittest.TestCase):
    def test_recovers_pose_and_focal_without_noise(self):
        for name, (camera, size) in standard_scenes().items():
            solution = solve_camera(observe(camera, size), size)
            self.assertIsNotNone(solution, name)
            self.assertLess(grid_error_cm(camera, solution.camera).max(), 0.5, name)
            self.assertLess(abs(solution.camera.focal_px / camera.focal_px - 1), 0.005, name)

    def test_recovers_focal_and_floor_with_pixel_noise(self):
        for name, (camera, size) in standard_scenes().items():
            solution = solve_camera(observe(camera, size, noise_px=1.0, seed=3), size)
            self.assertIsNotNone(solution, name)
            errors = grid_error_cm(camera, solution.camera)
            self.assertLess(np.median(errors), 3.0, name)
            self.assertLess(errors.max(), 15.0, name)
            self.assertLess(abs(solution.camera.focal_px / camera.focal_px - 1), 0.03, name)
            self.assertIn(solution.tier, ("validated", "approximate"), name)

    def test_four_floor_keypoints_give_an_approximate_solution(self):
        camera, size = standard_scenes()["behind_elevated"]
        names = ["long0_sl", "long0_sr", "long1_sl", "long1_sr"]
        observed = observe(camera, size, names=names)
        self.assertEqual(len(observed), 4)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        self.assertEqual(solution.tier, "approximate")
        self.assertEqual(solution.redundancy, 0)
        self.assertIsNone(solution.loo_floor_cm)
        self.assertLess(grid_error_cm(camera, solution.camera).max(), 5.0)

    def test_partial_court_far_half_still_solves(self):
        camera, size = standard_scenes()["behind_elevated"]
        far = {k: v for k, v in observe(camera, size, noise_px=1.0, seed=5).items() if KEYPOINTS[k][1] >= NET_Y_M}
        solution = solve_camera(far, size)
        self.assertIsNotNone(solution)
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)
        self.assertLess(abs(solution.camera.focal_px / camera.focal_px - 1), 0.03)

    def test_rejects_outlier_taps(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size, noise_px=1.0, seed=5)
        bad = ["long0_sr", "short0_dr"]
        for name in bad:
            self.assertIn(name, observed)
            observed[name] = (observed[name][0] + 70.0, observed[name][1] - 50.0)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        self.assertEqual(set(solution.outliers), set(bad))
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)

    def test_recovers_lens_distortion(self):
        camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, LANDSCAPE, k1=-0.08)
        solution = solve_camera(observe(camera, LANDSCAPE, noise_px=0.7, seed=2), LANDSCAPE)
        self.assertIsNotNone(solution)
        self.assertLess(abs(solution.camera.k1 + 0.08), 0.03)
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)

    def test_wrong_focal_prior_still_converges(self):
        camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1000.0, LANDSCAPE)
        solution = solve_camera(observe(camera, LANDSCAPE, noise_px=1.0, seed=2), LANDSCAPE)
        self.assertIsNotNone(solution)
        self.assertLess(abs(solution.camera.focal_px / 1000.0 - 1), 0.05)

    def test_gross_noise_is_never_validated(self):
        camera, size = standard_scenes()["behind_elevated"]
        solution = solve_camera(observe(camera, size, noise_px=12.0, seed=1), size)
        self.assertTrue(solution is None or solution.tier != "validated")

    def test_fewer_than_four_keypoints_returns_none(self):
        camera, size = standard_scenes()["behind_elevated"]
        three = observe(camera, size, names=["long0_sl", "long0_sr", "long1_sl"])
        self.assertIsNone(solve_camera(three, size))

    def test_collinear_floor_points_return_none(self):
        camera, size = standard_scenes()["behind_elevated"]
        column = {k: v for k, v in observe(camera, size).items() if k.endswith("_sl")}
        self.assertGreaterEqual(len(column), 4)
        self.assertIsNone(solve_camera(column, size))

    def test_off_floor_points_alone_return_none(self):
        camera, size = standard_scenes()["behind_elevated"]
        off_floor = {k: v for k, v in observe(camera, size).items() if KEYPOINTS[k][2] > 0}
        self.assertIsNone(solve_camera(off_floor, size))

    def test_unknown_names_are_ignored(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size, noise_px=0.5, seed=1)
        observed["not_a_keypoint"] = (10.0, 10.0)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        self.assertNotIn("not_a_keypoint", solution.inliers + solution.outliers)

    def test_non_finite_pixels_are_ignored(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size, noise_px=0.5, seed=1)
        names = [name for name in observed if KEYPOINTS[name][2] == 0.0][:2]
        observed[names[0]] = (float("nan"), 100.0)
        observed[names[1]] = (float("inf"), 100.0)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        for name in names:
            self.assertNotIn(name, solution.inliers + solution.outliers)
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)

    def test_identical_pixels_do_not_crash_or_validate(self):
        same = {name: (500.0, 500.0) for name in ("back0_sl", "back0_sr", "back1_sl", "back1_sr")}
        solution = solve_camera(same, LANDSCAPE)
        self.assertTrue(solution is None or solution.tier == "unavailable")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_calibrate -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'geometry.calibrate'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/calibrate.py
"""Single-image camera calibration against the BWF singles court model.

Given pixel observations of named court keypoints, recover the pinhole camera
(focal length, pose, one radial term). Floor keypoints give a homography, whose
decomposition seeds a robust nonlinear refinement over all keypoints. Net-post
and net-tape points are off the floor plane and pin the focal length when
visible. The caller learns how far to trust the result from `tier`, the
in-sample `rms_px`, and the leave-one-out floor error.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from geometry.camera import Camera
from geometry.court_model import KEYPOINTS

TYPICAL_FOCAL_OVER_LONG_SIDE = 0.72  # 26 mm-equivalent phone main lens
OUTLIER_PX = 8.0
MIN_KEYPOINTS = 4
MIN_K1_POINTS = 8
VALIDATED_MAX_RMS_PX = 3.0
VALIDATED_MAX_LOO_CM = 15.0
APPROX_MAX_RMS_PX = 8.0
APPROX_MAX_LOO_CM = 40.0
VALIDATED_MIN_REDUNDANCY = 2
_PRIOR_FOCAL_PX_PER_LOG = 6.0
_PRIOR_K1_PX_PER_UNIT = 30.0


@dataclass(frozen=True)
class CameraSolution:
    camera: Camera
    tier: str  # "validated" | "approximate" | "unavailable"
    rms_px: float
    floor_rms_cm: float
    loo_floor_cm: float | None
    redundancy: int
    inliers: tuple[str, ...]
    outliers: tuple[str, ...]


def _non_collinear(points_xy: np.ndarray) -> bool:
    centred = points_xy - points_xy.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return bool(singular[0] > 0 and singular[-1] / singular[0] > 0.02)


def _focal_from_homography(homography: np.ndarray, cx: float, cy: float) -> float | None:
    g = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ homography
    h1, h2 = g[:, 0], g[:, 1]
    denominator = h1[2] * h2[2]
    if abs(denominator) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / denominator
        if f2 > 0:
            return float(np.sqrt(f2))
    denominator = h2[2] ** 2 - h1[2] ** 2
    if abs(denominator) > 1e-12:
        f2 = ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / denominator
        if f2 > 0:
            return float(np.sqrt(f2))
    return None


def _pose_from_homography(homography: np.ndarray, focal: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    m = np.linalg.inv(k) @ homography
    lam = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    if m[2, 2] < 0:
        lam = -lam
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    u, _, vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return cv2.Rodrigues(rotation)[0].ravel(), t


def _unpack(params: np.ndarray, cx: float, cy: float, fit_k1: bool) -> Camera:
    return Camera(
        float(np.exp(params[6])), cx, cy, params[:3].copy(), params[3:6].copy(), float(params[7]) if fit_k1 else 0.0
    )


def _pack(camera: Camera, fit_k1: bool) -> np.ndarray:
    base = [*np.asarray(camera.rvec).ravel(), *np.asarray(camera.tvec).ravel(), np.log(camera.focal_px)]
    return np.array(base + ([camera.k1] if fit_k1 else []), dtype=np.float64)


def _refine(
    start: np.ndarray,
    world: np.ndarray,
    pixels: np.ndarray,
    cx: float,
    cy: float,
    focal_typical: float,
    fit_k1: bool,
) -> tuple[np.ndarray, float]:
    def residuals(params: np.ndarray) -> np.ndarray:
        projected = _unpack(params, cx, cy, fit_k1).project(world)
        flat = (projected - pixels).ravel()
        flat[~np.isfinite(flat)] = 1e3
        terms = [flat, [_PRIOR_FOCAL_PX_PER_LOG * (params[6] - np.log(focal_typical))]]
        if fit_k1:
            terms.append([_PRIOR_K1_PX_PER_UNIT * params[7]])
        return np.concatenate(terms)

    result = least_squares(residuals, start, loss="soft_l1", f_scale=2.0, method="trf", max_nfev=200)
    return result.x, float(result.cost)


def _fit(
    names: list[str],
    observations: dict[str, tuple[float, float]],
    cx: float,
    cy: float,
    focal_typical: float,
) -> Camera | None:
    world = np.array([KEYPOINTS[name] for name in names])
    pixels = np.array([observations[name] for name in names], dtype=np.float64)
    floor = world[:, 2] == 0.0
    if floor.sum() < MIN_KEYPOINTS or not _non_collinear(world[floor, :2]):
        return None
    method = cv2.RANSAC if floor.sum() >= 5 else 0
    homography, _ = cv2.findHomography(world[floor, :2].astype(np.float64), pixels[floor], method, 5.0)
    if homography is None:
        return None
    seeds = [focal_typical * scale for scale in (0.7, 1.0, 1.4)]
    from_homography = _focal_from_homography(homography, cx, cy)
    if from_homography is not None and 0.25 * focal_typical < from_homography < 4.0 * focal_typical:
        seeds.append(from_homography)
    fit_k1 = len(names) >= MIN_K1_POINTS
    best: tuple[np.ndarray, float] | None = None
    for focal in seeds:
        rvec, tvec = _pose_from_homography(homography, focal, cx, cy)
        start = _pack(Camera(focal, cx, cy, rvec, tvec), fit_k1)
        params, cost = _refine(start, world, pixels, cx, cy, focal_typical, fit_k1)
        if best is None or cost < best[1]:
            best = (params, cost)
    assert best is not None
    return _unpack(best[0], cx, cy, fit_k1)


def _refit_warm(
    names: list[str],
    observations: dict[str, tuple[float, float]],
    camera: Camera,
    cx: float,
    cy: float,
    focal_typical: float,
) -> Camera | None:
    world = np.array([KEYPOINTS[name] for name in names])
    floor = world[:, 2] == 0.0
    if floor.sum() < MIN_KEYPOINTS or not _non_collinear(world[floor, :2]):
        return None
    pixels = np.array([observations[name] for name in names], dtype=np.float64)
    fit_k1 = len(names) >= MIN_K1_POINTS
    params, _ = _refine(_pack(camera, fit_k1), world, pixels, cx, cy, focal_typical, fit_k1)
    return _unpack(params, cx, cy, fit_k1)


def _floor_error_cm(camera: Camera, names: list[str], observations: dict[str, tuple[float, float]]) -> np.ndarray:
    pixels = np.array([observations[name] for name in names], dtype=np.float64)
    truth = np.array([KEYPOINTS[name][:2] for name in names])
    return np.linalg.norm(camera.pixel_to_plane(pixels, 0.0) - truth, axis=1) * 100.0


def _residual_px(camera: Camera, names: list[str], observations: dict[str, tuple[float, float]]) -> np.ndarray:
    projected = camera.project(np.array([KEYPOINTS[name] for name in names]))
    observed = np.array([observations[name] for name in names], dtype=np.float64)
    distance = np.linalg.norm(projected - observed, axis=1)
    distance[~np.isfinite(distance)] = 1e3
    return distance


def _leave_one_out_cm(
    camera: Camera,
    inliers: list[str],
    floor_names: list[str],
    observations: dict[str, tuple[float, float]],
    cx: float,
    cy: float,
    focal_typical: float,
) -> float | None:
    if len(floor_names) < MIN_KEYPOINTS + 1:
        return None
    errors: list[float] = []
    for held_out in floor_names:
        rest = [name for name in inliers if name != held_out]
        refit = _refit_warm(rest, observations, camera, cx, cy, focal_typical)
        if refit is None:
            continue
        errors.append(float(_floor_error_cm(refit, [held_out], observations)[0]))
    if not errors:
        return None
    return float(np.sqrt(np.mean(np.square(errors))))


def _tier(rms_px: float, loo_floor_cm: float | None, redundancy: int) -> str:
    validated = (
        redundancy >= VALIDATED_MIN_REDUNDANCY
        and loo_floor_cm is not None
        and rms_px <= VALIDATED_MAX_RMS_PX
        and loo_floor_cm <= VALIDATED_MAX_LOO_CM
    )
    if validated:
        return "validated"
    if rms_px <= APPROX_MAX_RMS_PX and (loo_floor_cm is None or loo_floor_cm <= APPROX_MAX_LOO_CM):
        return "approximate"
    return "unavailable"


def solve_camera(
    observations: dict[str, tuple[float, float]],
    image_size: tuple[int, int],
    focal_prior_px: float | None = None,
) -> CameraSolution | None:
    """Return the best camera for the observed keypoints, or None when under-determined."""
    width, height = image_size
    cx, cy = width / 2, height / 2
    focal_typical = focal_prior_px or TYPICAL_FOCAL_OVER_LONG_SIDE * max(width, height)
    names = [name for name, pixel in observations.items() if name in KEYPOINTS and np.isfinite(pixel).all()]
    if len(names) < MIN_KEYPOINTS:
        return None
    camera = _fit(names, observations, cx, cy, focal_typical)
    if camera is None:
        return None
    residual = _residual_px(camera, names, observations)
    inliers = [name for name, value in zip(names, residual) if value <= OUTLIER_PX]
    outliers = [name for name, value in zip(names, residual) if value > OUTLIER_PX]
    if outliers and len(inliers) >= MIN_KEYPOINTS:
        refit = _fit(inliers, observations, cx, cy, focal_typical)
        if refit is not None:
            camera = refit
        else:
            inliers, outliers = names, []
    else:
        inliers, outliers = names, []
    floor_names = [name for name in inliers if KEYPOINTS[name][2] == 0.0]
    floor_rms_cm = float(np.sqrt(np.mean(_floor_error_cm(camera, floor_names, observations) ** 2)))
    rms_px = float(np.sqrt(np.mean(np.square(_residual_px(camera, inliers, observations)))))
    loo = _leave_one_out_cm(camera, inliers, floor_names, observations, cx, cy, focal_typical)
    redundancy = len(inliers) - MIN_KEYPOINTS
    return CameraSolution(
        camera=camera,
        tier=_tier(rms_px, loo, redundancy),
        rms_px=rms_px,
        floor_rms_cm=floor_rms_cm,
        loo_floor_cm=loo,
        redundancy=redundancy,
        inliers=tuple(inliers),
        outliers=tuple(outliers),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_calibrate -v`
Expected: PASS (14 tests, a few seconds). If a numeric threshold fails by a hair, print the measured value and loosen only that threshold to ~2× the measurement; never loosen the tier or outlier assertions.

- [ ] **Step 5: Commit**

```bash
git add worker/geometry/calibrate.py worker/test_calibrate.py
git commit -m "feat(worker): add robust single-image camera calibration solver"
```

---

### Task 5: Capture-geometry tier

**Files:**
- Create: `worker/geometry/quality.py`
- Test: `worker/test_quality.py`

**Interfaces:**
- Consumes: `Camera` (`.centre`, `.project`), court constants.
- Produces: `assess_geometry(camera: Camera, image_size: tuple[int,int]) -> GeometryQuality` and frozen dataclass `GeometryQuality(tier: str, elevation_deg: float, visible_fraction: float, distance_m: float, reasons: tuple[str, ...])` with `tier` in `"A" | "B" | "C"`; `court_visibility(camera, image_size) -> float`. Constants `TIER_A_MIN_ELEVATION_DEG=12.0`, `TIER_A_MIN_VISIBLE=0.85`, `TIER_B_MIN_ELEVATION_DEG=5.0`, `TIER_B_MIN_VISIBLE=0.40`.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_quality.py
import unittest

from geometry.camera import Camera
from geometry.quality import assess_geometry, court_visibility
from geometry.synthetic import LANDSCAPE, standard_scenes


class GeometryTierTests(unittest.TestCase):
    def tier(self, name: str) -> str:
        camera, size = standard_scenes()[name]
        return assess_geometry(camera, size).tier

    def test_elevated_views_with_the_whole_court_are_tier_a(self):
        for name in ("behind_elevated", "corner_elevated", "behind_portrait"):
            self.assertEqual(self.tier(name), "A", name)

    def test_low_camera_is_tier_b_and_says_to_raise_the_phone(self):
        camera, size = standard_scenes()["behind_low"]
        quality = assess_geometry(camera, size)
        self.assertEqual(quality.tier, "B")
        self.assertTrue(any("Raise the phone" in reason for reason in quality.reasons))

    def test_side_view_missing_part_of_the_court_is_tier_b(self):
        # side_elevated has ~83% of the court in frame, just under the tier A bar.
        camera, size = standard_scenes()["side_elevated"]
        quality = assess_geometry(camera, size)
        self.assertEqual(quality.tier, "B")
        self.assertTrue(any("out of frame" in reason for reason in quality.reasons))

    def test_close_camera_seeing_little_court_is_tier_c(self):
        camera = Camera.look_at((2.59, 1.0, 2.0), (2.59, 3.0, 0.0), 1300.0, LANDSCAPE)
        quality = assess_geometry(camera, LANDSCAPE)
        self.assertEqual(quality.tier, "C")
        self.assertLess(quality.visible_fraction, 0.40)

    def test_floor_level_camera_is_tier_c(self):
        camera = Camera.look_at((2.59, -3.0, 0.3), (2.59, 7.0, 0.3), 1300.0, LANDSCAPE)
        quality = assess_geometry(camera, LANDSCAPE)
        self.assertEqual(quality.tier, "C")
        self.assertLess(quality.elevation_deg, 5.0)

    def test_visibility_is_a_fraction(self):
        for camera, size in standard_scenes().values():
            self.assertTrue(0.0 <= court_visibility(camera, size) <= 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_quality -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'geometry.quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/quality.py
"""How much of the game a given camera placement can support."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, NET_Y_M, SINGLES_WIDTH_M

TIER_A_MIN_ELEVATION_DEG = 12.0
TIER_A_MIN_VISIBLE = 0.85
TIER_B_MIN_ELEVATION_DEG = 5.0
TIER_B_MIN_VISIBLE = 0.40


@dataclass(frozen=True)
class GeometryQuality:
    tier: str  # "A" | "B" | "C"
    elevation_deg: float
    visible_fraction: float
    distance_m: float
    reasons: tuple[str, ...]


def court_visibility(camera: Camera, image_size: tuple[int, int]) -> float:
    """Fraction of a regular grid over the singles court that lands inside the frame."""
    width, height = image_size
    xs = np.linspace(0.0, SINGLES_WIDTH_M, 11)
    ys = np.linspace(0.0, COURT_LENGTH_M, 21)
    grid = np.array([[x, y, 0.0] for y in ys for x in xs])
    pixels = camera.project(grid)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return float(inside.mean())


def assess_geometry(camera: Camera, image_size: tuple[int, int]) -> GeometryQuality:
    court_centre = np.array([SINGLES_WIDTH_M / 2, NET_Y_M, 0.0])
    offset = camera.centre - court_centre
    elevation = math.degrees(math.atan2(float(offset[2]), float(np.linalg.norm(offset[:2]))))
    distance = float(np.linalg.norm(offset))
    visible = court_visibility(camera, image_size)
    reasons: list[str] = []
    if elevation < TIER_B_MIN_ELEVATION_DEG:
        reasons.append("Camera is almost at floor level; court depth cannot be measured reliably. Raise the phone.")
    elif elevation < TIER_A_MIN_ELEVATION_DEG:
        reasons.append("Camera is low; depth estimates carry wide uncertainty. Raise the phone for full 3D analysis.")
    if visible < TIER_B_MIN_VISIBLE:
        reasons.append("Less than 40% of the court is in frame. Step back or widen the view.")
    elif visible < TIER_A_MIN_VISIBLE:
        reasons.append("Part of the court is out of frame. Step back until both baselines are visible for full analysis.")
    if elevation >= TIER_A_MIN_ELEVATION_DEG and visible >= TIER_A_MIN_VISIBLE:
        tier = "A"
    elif elevation >= TIER_B_MIN_ELEVATION_DEG and visible >= TIER_B_MIN_VISIBLE:
        tier = "B"
    else:
        tier = "C"
    return GeometryQuality(tier, elevation, visible, distance, tuple(reasons))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_quality -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/geometry/quality.py worker/test_quality.py
git commit -m "feat(worker): add capture-geometry capability tier (A/B/C)"
```

---

### Task 6: Evaluation metrics

**Files:**
- Create: `worker/evaluation/__init__.py`
- Create: `worker/evaluation/metrics.py`
- Test: `worker/test_eval_metrics.py`

**Interfaces:**
- Produces: `match_events(predicted_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> EventMatch` with frozen dataclass `EventMatch(true_positives, false_positives, false_negatives, precision, recall, f1, median_abs_error_ms: float | None, pairs: tuple[tuple[int, int], ...])` where `pairs` are `(predicted_index, truth_index)`; one-to-one, nearest first. `macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] | None = None) -> dict` returning `{"macroF1": float, "perLabel": {label: {"precision","recall","f1","support","predicted"}}}`; a label is scored only when it has support or predictions; `ValueError` when lengths differ.

- [ ] **Step 1: Write the failing test**

```python
# worker/test_eval_metrics.py
import unittest

from evaluation.metrics import macro_f1, match_events


class MatchEventsTests(unittest.TestCase):
    def test_perfect_match(self):
        match = match_events([100, 500], [100, 500], 50)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (2, 0, 0))
        self.assertEqual(match.f1, 1.0)
        self.assertEqual(match.median_abs_error_ms, 0.0)

    def test_tolerance_is_inclusive_and_misses_count(self):
        match = match_events([130, 900], [100, 500], 30)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (1, 1, 1))
        self.assertEqual(match.pairs, ((0, 0),))

    def test_one_prediction_cannot_match_two_truths_and_nearest_wins(self):
        match = match_events([110, 118], [100], 50)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (1, 1, 0))
        self.assertEqual(match.pairs, ((0, 0),))
        self.assertEqual(match.median_abs_error_ms, 10.0)

    def test_median_error_uses_matched_pairs_only(self):
        match = match_events([100, 520, 900], [110, 500], 50)
        self.assertEqual(match.median_abs_error_ms, 15.0)

    def test_nothing_to_score_is_zero_not_an_error(self):
        match = match_events([], [], 50)
        self.assertEqual((match.precision, match.recall, match.f1), (0.0, 0.0, 0.0))
        self.assertIsNone(match.median_abs_error_ms)


class MacroF1Tests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = macro_f1(["clear", "drop"], ["clear", "drop"])
        self.assertEqual(result["macroF1"], 1.0)

    def test_labels_never_seen_or_predicted_are_not_scored(self):
        result = macro_f1(["clear", "clear"], ["clear", "clear"], ("clear", "drop", "smash"))
        self.assertEqual(result["macroF1"], 1.0)
        self.assertEqual(result["perLabel"]["drop"]["support"], 0)

    def test_wrong_and_missing_labels_lower_the_score(self):
        y_true = ["clear", "smash", "drop"]
        y_pred = ["clear", "drive", "missed"]
        result = macro_f1(y_true, y_pred, ("clear", "smash", "drop", "drive"))
        # clear 1.0, smash 0.0, drop 0.0, drive 0.0 (predicted once, never true)
        self.assertAlmostEqual(result["macroF1"], 0.25)
        self.assertEqual(result["perLabel"]["drive"]["predicted"], 1)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            macro_f1(["clear"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd worker && python -m unittest test_eval_metrics -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/evaluation/__init__.py
"""Offline evaluation of worker results against hand-labelled clips."""
```

```python
# worker/evaluation/metrics.py
"""Event-timing and classification metrics used by every capability gate."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class EventMatch:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    median_abs_error_ms: float | None
    pairs: tuple[tuple[int, int], ...]  # (predicted index, truth index)


def match_events(predicted_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> EventMatch:
    """One-to-one matching, closest pairs first, within an inclusive tolerance."""
    candidates = sorted(
        (abs(p - t), pi, ti)
        for pi, p in enumerate(predicted_ms)
        for ti, t in enumerate(truth_ms)
        if abs(p - t) <= tolerance_ms
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    pairs: list[tuple[int, int]] = []
    errors: list[float] = []
    for distance, pi, ti in candidates:
        if pi in used_predicted or ti in used_truth:
            continue
        used_predicted.add(pi)
        used_truth.add(ti)
        pairs.append((pi, ti))
        errors.append(distance)
    true_positives = len(pairs)
    false_positives = len(predicted_ms) - true_positives
    false_negatives = len(truth_ms) - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventMatch(
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
        float(median(errors)) if errors else None,
        tuple(sorted(pairs)),
    )


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] | None = None) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    classes = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    per_label: dict[str, dict[str, float]] = {}
    for label in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "predicted": tp + fp,
        }
    scored = [row["f1"] for row in per_label.values() if row["support"] > 0 or row["predicted"] > 0]
    return {"macroF1": sum(scored) / len(scored) if scored else 0.0, "perLabel": per_label}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd worker && python -m unittest test_eval_metrics -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/evaluation/__init__.py worker/evaluation/metrics.py worker/test_eval_metrics.py
git commit -m "feat(worker): add event-matching and macro-F1 evaluation metrics"
```

---

### Task 7: Golden-clip schema, report CLI and labelling guide

**Files:**
- Create: `worker/evaluation/golden.py`
- Create: `worker/evaluation/evaluate.py`
- Create: `docs/golden-set.md`
- Test: `worker/test_golden.py`, `worker/test_evaluate.py`

**Interfaces:**
- Consumes: `match_events`, `macro_f1`, `solve_camera`, `assess_geometry`, `KEYPOINTS`, `observe`/`standard_scenes` (tests).
- Produces: `golden.SCHEMA_VERSION = 1`, `golden.GOLDEN_SHOT_LABELS` (`serve clear drop net lift drive push smash block other`), `validate_golden(doc) -> list[str]` (empty = valid), `load_golden_dir(directory: Path) -> list[dict]` (raises `ValueError` listing every problem); `evaluate.evaluate_clip(golden, result) -> dict`, `evaluate.evaluate_directory(golden_dir, results_dir) -> dict`, `evaluate.main(argv) -> int`. A golden clip is JSON: `schemaVersion, clipId, sourceFps, imageSize [w,h], courtKeypoints [{name,x,y}] (pixels), contacts [{timeMs}], shots [{timeMs,label,landing:{x,y}|null}] (landing in court metres), rallies [{startMs,endMs,winner}]`. A worker result is the job output JSON: `events[{type,timeMs}]`, `shots[{timeMs,label,verified}]`.

- [ ] **Step 1: Write the failing tests**

```python
# worker/test_golden.py
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from evaluation.golden import GOLDEN_SHOT_LABELS, load_golden_dir, validate_golden

VALID = {
    "schemaVersion": 1,
    "clipId": "clip-001",
    "sourceFps": 60,
    "imageSize": [1920, 1080],
    "courtKeypoints": [{"name": "back1_sl", "x": 612.0, "y": 301.5}],
    "contacts": [{"timeMs": 1200}, {"timeMs": 2100}],
    "shots": [
        {"timeMs": 1200, "label": "serve", "landing": {"x": 2.4, "y": 9.8}},
        {"timeMs": 2100, "label": "clear", "landing": None},
    ],
    "rallies": [{"startMs": 800, "endMs": 4300, "winner": "near"}],
}


def corrupted(**changes):
    doc = copy.deepcopy(VALID)
    doc.update(changes)
    return doc


class ValidateGoldenTests(unittest.TestCase):
    def test_valid_document_has_no_errors(self):
        self.assertEqual(validate_golden(VALID), [])

    def test_labels_include_the_worker_shot_classes(self):
        for label in ("smash", "clear", "drop", "net", "lift", "drive", "push", "serve"):
            self.assertIn(label, GOLDEN_SHOT_LABELS)

    def test_reports_each_kind_of_problem(self):
        cases = {
            "schemaVersion": corrupted(schemaVersion=2),
            "clipId": corrupted(clipId=" "),
            "sourceFps": corrupted(sourceFps=0),
            "imageSize": corrupted(imageSize=[1920]),
            "keypoint": corrupted(courtKeypoints=[{"name": "nope", "x": 1, "y": 1}]),
            "outside": corrupted(courtKeypoints=[{"name": "back1_sl", "x": 5000, "y": 1}]),
            "contact": corrupted(contacts=[{"timeMs": -5}]),
            "label": corrupted(shots=[{"timeMs": 10, "label": "bogus", "landing": None}]),
            "landing": corrupted(shots=[{"timeMs": 10, "label": "clear", "landing": {"x": 90, "y": 1}}]),
            "before endms": corrupted(rallies=[{"startMs": 500, "endMs": 400, "winner": "near"}]),
            "winner": corrupted(rallies=[{"startMs": 1, "endMs": 400, "winner": "left"}]),
        }
        for needle, doc in cases.items():
            errors = validate_golden(doc)
            self.assertTrue(any(needle.lower() in error.lower() for error in errors), (needle, errors))

    def test_non_object_and_missing_lists_are_rejected(self):
        self.assertTrue(validate_golden([]))
        broken = copy.deepcopy(VALID)
        del broken["contacts"]
        self.assertTrue(any("contacts" in error for error in validate_golden(broken)))

    def test_bool_and_nan_are_not_numbers(self):
        self.assertTrue(validate_golden(corrupted(sourceFps=True)))
        self.assertTrue(validate_golden(corrupted(contacts=[{"timeMs": float("nan")}])))


class LoadGoldenDirTests(unittest.TestCase):
    def test_loads_valid_files_and_reports_all_bad_ones(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "a.json").write_text(json.dumps(VALID))
            self.assertEqual([doc["clipId"] for doc in load_golden_dir(directory)], ["clip-001"])
            (directory / "b.json").write_text(json.dumps(corrupted(sourceFps=-1)))
            (directory / "c.json").write_text("{not json")
            with self.assertRaises(ValueError) as raised:
                load_golden_dir(directory)
            message = str(raised.exception)
            self.assertIn("b.json", message)
            self.assertIn("c.json", message)


class GoldenGuideTests(unittest.TestCase):
    def test_the_json_example_in_the_guide_is_valid(self):
        guide = Path(__file__).resolve().parent.parent / "docs" / "golden-set.md"
        match = re.search(r"```json\n(.*?)\n```", guide.read_text(encoding="utf-8"), re.S)
        self.assertIsNotNone(match)
        self.assertEqual(validate_golden(json.loads(match.group(1))), [])


if __name__ == "__main__":
    unittest.main()
```

```python
# worker/test_evaluate.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import evaluate_clip, evaluate_directory, main
from geometry.synthetic import observe, standard_scenes


def golden_clip(clip_id: str = "clip-001") -> dict:
    camera, size = standard_scenes()["behind_elevated"]
    keypoints = [{"name": name, "x": x, "y": y} for name, (x, y) in observe(camera, size, noise_px=0.5, seed=1).items()]
    return {
        "schemaVersion": 1,
        "clipId": clip_id,
        "sourceFps": 60,
        "imageSize": list(size),
        "courtKeypoints": keypoints,
        "contacts": [{"timeMs": 1000}, {"timeMs": 2000}, {"timeMs": 3000}],
        "shots": [
            {"timeMs": 1000, "label": "clear", "landing": None},
            {"timeMs": 2000, "label": "smash", "landing": None},
            {"timeMs": 3000, "label": "drop", "landing": None},
        ],
        "rallies": [],
    }


RESULT = {
    "events": [
        {"type": "contact", "timeMs": 1010},
        {"type": "contact", "timeMs": 2050},
        {"type": "contact", "timeMs": 3500},
        {"type": "split_step", "timeMs": 900},
    ],
    "shots": [
        {"timeMs": 1010, "label": "clear", "verified": True},
        {"timeMs": 2050, "label": "drive", "verified": True},
        {"timeMs": 3500, "label": "drop", "verified": True},
    ],
}


class EvaluateClipTests(unittest.TestCase):
    def test_contact_metrics_at_both_tolerances(self):
        report = evaluate_clip(golden_clip(), RESULT)
        tight = report["contacts"]["tol33ms"]
        loose = report["contacts"]["tol100ms"]
        self.assertEqual((tight["truePositives"], tight["falsePositives"], tight["falseNegatives"]), (1, 2, 2))
        self.assertEqual((loose["truePositives"], loose["falsePositives"], loose["falseNegatives"]), (2, 1, 1))
        self.assertEqual(loose["medianAbsErrorMs"], 30.0)

    def test_shot_macro_f1_counts_missed_and_wrong_labels(self):
        shots = evaluate_clip(golden_clip(), RESULT)["shots"]
        self.assertEqual(shots["matched"], 2)
        self.assertEqual(shots["truthCount"], 3)
        self.assertAlmostEqual(shots["macroF1"], 0.25)

    def test_unverified_prediction_is_scored_as_wrong(self):
        result = {
            "events": [],
            "shots": [{"timeMs": 1000, "label": "clear", "verified": False}],
        }
        shots = evaluate_clip(golden_clip(), result)["shots"]
        self.assertEqual(shots["perLabel"]["clear"]["f1"], 0.0)

    def test_calibration_is_scored_from_golden_keypoints(self):
        calibration = evaluate_clip(golden_clip(), RESULT)["calibration"]
        self.assertIn(calibration["tier"], ("validated", "approximate"))
        self.assertIn(calibration["geometryTier"], ("A", "B", "C"))
        self.assertGreater(calibration["redundancy"], 0)

    def test_clip_without_enough_keypoints_has_no_calibration(self):
        clip = golden_clip()
        clip["courtKeypoints"] = clip["courtKeypoints"][:2]
        self.assertIsNone(evaluate_clip(clip, RESULT)["calibration"])


class EvaluateDirectoryTests(unittest.TestCase):
    def test_report_summarises_clips_and_flags_missing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "golden").mkdir()
            (root / "results").mkdir()
            (root / "golden" / "a.json").write_text(json.dumps(golden_clip("a")))
            (root / "golden" / "b.json").write_text(json.dumps(golden_clip("b")))
            (root / "results" / "a.json").write_text(json.dumps(RESULT))
            report = evaluate_directory(root / "golden", root / "results")
            self.assertEqual(report["summary"]["clips"], 2)
            self.assertEqual(report["summary"]["clipsWithResults"], 1)
            by_id = {clip["clipId"]: clip for clip in report["clips"]}
            self.assertIn("error", by_id["b"])
            self.assertAlmostEqual(report["summary"]["meanContactF1At100ms"], by_id["a"]["contacts"]["tol100ms"]["f1"])
            self.assertEqual(sum(report["summary"]["calibrationTiers"].values()), 1)

    def test_cli_writes_the_report_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "golden").mkdir()
            (root / "results").mkdir()
            (root / "golden" / "a.json").write_text(json.dumps(golden_clip("a")))
            (root / "results" / "a.json").write_text(json.dumps(RESULT))
            out = root / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--golden", str(root / "golden"), "--results", str(root / "results"), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["summary"]["clips"], 1)

    def test_cli_returns_nonzero_on_invalid_golden_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "golden").mkdir()
            (root / "results").mkdir()
            (root / "golden" / "bad.json").write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["--golden", str(root / "golden"), "--results", str(root / "results")])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd worker && python -m unittest test_golden test_evaluate -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'evaluation.golden'`

- [ ] **Step 3: Write minimal implementation**

```python
# worker/evaluation/golden.py
"""Schema and loader for hand-labelled golden clips (see docs/golden-set.md)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, KEYPOINTS, SINGLES_WIDTH_M

SCHEMA_VERSION = 1
GOLDEN_SHOT_LABELS = ("serve", "clear", "drop", "net", "lift", "drive", "push", "smash", "block", "other")
WINNERS = ("near", "far", "unknown")
LANDING_MARGIN_M = 1.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_golden(doc: Any) -> list[str]:
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    errors: list[str] = []
    if doc.get("schemaVersion") != SCHEMA_VERSION:
        errors.append(f"schemaVersion must be {SCHEMA_VERSION}")
    clip_id = doc.get("clipId")
    if not isinstance(clip_id, str) or not clip_id.strip():
        errors.append("clipId must be a non-empty string")
    if not _is_number(doc.get("sourceFps")) or doc["sourceFps"] <= 0:
        errors.append("sourceFps must be a positive number")
    size = doc.get("imageSize")
    size_ok = (
        isinstance(size, list)
        and len(size) == 2
        and all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in size)
    )
    if not size_ok:
        errors.append("imageSize must be [width, height] positive integers")
    for index, item in enumerate(doc.get("courtKeypoints", [])):
        where = f"courtKeypoints[{index}]"
        if not isinstance(item, dict) or item.get("name") not in KEYPOINTS:
            errors.append(f"{where}: keypoint name must be one of the court model names")
        elif not (_is_number(item.get("x")) and _is_number(item.get("y"))):
            errors.append(f"{where}: x and y must be numbers")
        elif size_ok and not (0 <= item["x"] < size[0] and 0 <= item["y"] < size[1]):
            errors.append(f"{where}: pixel is outside the image")
    for key in ("contacts", "shots"):
        if not isinstance(doc.get(key), list):
            errors.append(f"{key} must be a list")
    for index, item in enumerate(doc.get("contacts") or []):
        if not isinstance(item, dict) or not _is_number(item.get("timeMs")) or item["timeMs"] < 0:
            errors.append(f"contacts[{index}]: timeMs must be a non-negative number")
    for index, item in enumerate(doc.get("shots") or []):
        where = f"shots[{index}]"
        if not isinstance(item, dict) or not _is_number(item.get("timeMs")) or item["timeMs"] < 0:
            errors.append(f"{where}: timeMs must be a non-negative number")
            continue
        if item.get("label") not in GOLDEN_SHOT_LABELS:
            errors.append(f"{where}: label must be one of {', '.join(GOLDEN_SHOT_LABELS)}")
        landing = item.get("landing")
        if landing is not None:
            in_court = (
                isinstance(landing, dict)
                and _is_number(landing.get("x"))
                and _is_number(landing.get("y"))
                and -DOUBLES_MARGIN_M - LANDING_MARGIN_M <= landing["x"] <= SINGLES_WIDTH_M + DOUBLES_MARGIN_M + LANDING_MARGIN_M
                and -LANDING_MARGIN_M <= landing["y"] <= COURT_LENGTH_M + LANDING_MARGIN_M
            )
            if not in_court:
                errors.append(f"{where}: landing must be null or court metres near the court")
    for index, item in enumerate(doc.get("rallies", [])):
        where = f"rallies[{index}]"
        if not isinstance(item, dict) or not (_is_number(item.get("startMs")) and _is_number(item.get("endMs"))):
            errors.append(f"{where}: startMs and endMs must be numbers")
            continue
        if item["startMs"] >= item["endMs"]:
            errors.append(f"{where}: startMs must be before endMs")
        if item.get("winner") not in WINNERS:
            errors.append(f"{where}: winner must be one of {', '.join(WINNERS)}")
    return errors


def load_golden_dir(directory: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: invalid JSON ({exc.msg})")
            continue
        errors = validate_golden(doc)
        if errors:
            problems.extend(f"{path.name}: {error}" for error in errors)
        else:
            documents.append(doc)
    if problems:
        raise ValueError("\n".join(problems))
    return documents
```

```python
# worker/evaluation/evaluate.py
"""Compare worker results with hand-labelled golden clips.

    python -m evaluation.evaluate --golden golden/ --results results/ --out report.json

`results/<clipId>.json` is the worker job output for that clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.golden import GOLDEN_SHOT_LABELS, load_golden_dir
from evaluation.metrics import macro_f1, match_events
from geometry.calibrate import solve_camera
from geometry.quality import assess_geometry

CONTACT_TOLERANCES_MS = (33, 100)
SHOT_MATCH_TOLERANCE_MS = 100


def evaluate_calibration(golden: dict[str, Any]) -> dict[str, Any] | None:
    size = (golden["imageSize"][0], golden["imageSize"][1])
    observations = {item["name"]: (item["x"], item["y"]) for item in golden.get("courtKeypoints", [])}
    solution = solve_camera(observations, size)
    if solution is None:
        return None
    quality = assess_geometry(solution.camera, size)
    return {
        "tier": solution.tier,
        "rmsPx": round(solution.rms_px, 3),
        "floorRmsCm": round(solution.floor_rms_cm, 2),
        "looFloorCm": None if solution.loo_floor_cm is None else round(solution.loo_floor_cm, 2),
        "redundancy": solution.redundancy,
        "outliers": list(solution.outliers),
        "geometryTier": quality.tier,
    }


def _contact_report(predicted: list[float], truth: list[float], tolerance_ms: float) -> dict[str, Any]:
    match = match_events(predicted, truth, tolerance_ms)
    return {
        "truePositives": match.true_positives,
        "falsePositives": match.false_positives,
        "falseNegatives": match.false_negatives,
        "precision": round(match.precision, 4),
        "recall": round(match.recall, 4),
        "f1": round(match.f1, 4),
        "medianAbsErrorMs": match.median_abs_error_ms,
    }


def evaluate_clip(golden: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    predicted_contacts = [event["timeMs"] for event in result.get("events", []) if event.get("type") == "contact"]
    truth_contacts = [contact["timeMs"] for contact in golden["contacts"]]
    contacts = {f"tol{tol}ms": _contact_report(predicted_contacts, truth_contacts, tol) for tol in CONTACT_TOLERANCES_MS}

    predicted_shots = result.get("shots", [])
    truth_shots = golden["shots"]
    match = match_events(
        [shot["timeMs"] for shot in predicted_shots], [shot["timeMs"] for shot in truth_shots], SHOT_MATCH_TOLERANCE_MS
    )
    matched = {truth_index: predicted_index for predicted_index, truth_index in match.pairs}
    y_true = [shot["label"] for shot in truth_shots]
    y_pred: list[str] = []
    for truth_index in range(len(truth_shots)):
        predicted_index = matched.get(truth_index)
        if predicted_index is None:
            y_pred.append("missed")
            continue
        shot = predicted_shots[predicted_index]
        y_pred.append(shot["label"] if shot.get("verified") else "unverified")
    scored = macro_f1(y_true, y_pred, GOLDEN_SHOT_LABELS)
    return {
        "clipId": golden["clipId"],
        "calibration": evaluate_calibration(golden),
        "contacts": contacts,
        "shots": {
            "matched": len(match.pairs),
            "truthCount": len(truth_shots),
            "macroF1": round(scored["macroF1"], 4),
            "perLabel": scored["perLabel"],
        },
    }


def evaluate_directory(golden_dir: Path, results_dir: Path) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    for golden in load_golden_dir(golden_dir):
        result_path = Path(results_dir) / f"{golden['clipId']}.json"
        if not result_path.is_file():
            clips.append({"clipId": golden["clipId"], "error": f"no result file {result_path.name}"})
            continue
        clips.append(evaluate_clip(golden, json.loads(result_path.read_text(encoding="utf-8"))))
    scored = [clip for clip in clips if "error" not in clip]
    tiers: dict[str, int] = {}
    for clip in scored:
        if clip["calibration"] is not None:
            tier = clip["calibration"]["tier"]
            tiers[tier] = tiers.get(tier, 0) + 1

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "clips": clips,
        "summary": {
            "clips": len(clips),
            "clipsWithResults": len(scored),
            "meanContactF1At33ms": mean([clip["contacts"]["tol33ms"]["f1"] for clip in scored]),
            "meanContactF1At100ms": mean([clip["contacts"]["tol100ms"]["f1"] for clip in scored]),
            "meanShotMacroF1": mean([clip["shots"]["macroF1"] for clip in scored]),
            "calibrationTiers": tiers,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate worker results against golden clips.")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_directory(args.golden, args.results)
    except ValueError as exc:
        print(f"Invalid golden set:\n{exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

````markdown
<!-- docs/golden-set.md -->
# Golden set: hand-labelled singles clips

Every capability gate in `docs/evaluation-protocol.md` is measured against these
clips. Build the set before trusting any model change. Target: **about 20
singles rallies**, 10–60 s each, **≥ 60 fps**, covering:

- tripod and **handheld** footage, portrait and landscape;
- behind-baseline elevated, low, side-on and corner placements;
- at least three phones, two venues, different lighting and floor colours;
- beginner, club and strong players; rallies of different lengths.

## Files

```
golden/<clipId>.json     hand labels (this schema)
results/<clipId>.json    the worker's job output for the same clip
```

Keep the source videos outside git (they are large and private); the labels and
results are small and belong in the repo or the team drive.

## Labelling a clip

1. **Court keypoints** — on one clear frame, click as many named court
   intersections as are visible (`back0_sl` = near baseline × left singles
   sideline; rows `back0 long0 short0 short1 long1 back1`, columns
   `dl sl c sr dr`; net: `post_left_base post_left_top post_right_base
   post_right_top net_centre_top`). Pixel coordinates in the original video
   resolution. More points give a tighter calibration check.
2. **Contacts** — the millisecond timestamp of every racket–shuttle contact,
   stepped frame by frame.
3. **Shots** — for each contact: the stroke label (`serve clear drop net lift
   drive push smash block other`) and, when visible, where the shuttle landed in
   court metres (`x` across from the left singles line, `y` from the near
   baseline) or `null`.
4. **Rallies** — start/end time and the winner as seen from the camera
   (`near`, `far`, or `unknown`).

## Format

```json
{
  "schemaVersion": 1,
  "clipId": "clip-001",
  "sourceFps": 60,
  "imageSize": [1920, 1080],
  "courtKeypoints": [
    { "name": "back1_sl", "x": 612.0, "y": 301.5 },
    { "name": "back1_sr", "x": 1310.0, "y": 298.0 }
  ],
  "contacts": [{ "timeMs": 1200 }, { "timeMs": 2100 }],
  "shots": [
    { "timeMs": 1200, "label": "serve", "landing": { "x": 2.4, "y": 9.8 } },
    { "timeMs": 2100, "label": "clear", "landing": null }
  ],
  "rallies": [{ "startMs": 800, "endMs": 4300, "winner": "near" }]
}
```

`worker/test_golden.py` validates this example, so it cannot drift from the schema.

## Running the baseline

1. Run each clip through the current worker (local `python worker/handler.py`
   test job or RunPod) and save the job output as `results/<clipId>.json`.
2. From `worker/`:

```bash
python -m evaluation.evaluate --golden ../golden --results ../results --out ../baseline-report.json
```

The report gives, per clip and in aggregate: contact precision/recall/F1 at ±33
and ±100 ms, shot macro-F1 (missed and unverified predictions count as wrong),
and calibration tier, pixel RMS and leave-one-out floor error. Commit the
baseline report before changing any model so every later phase can show lift.
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd worker && python -m unittest test_golden test_evaluate -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add worker/evaluation/golden.py worker/evaluation/evaluate.py worker/test_golden.py worker/test_evaluate.py docs/golden-set.md
git commit -m "feat(worker): add golden-clip schema, evaluation report CLI and labelling guide"
```

---

### Task 8: Worker integration and server schema

**Files:**
- Create: `worker/geometry/calibration_adapter.py`
- Modify: `worker/handler.py` (imports ~line 24; handler body ~lines 394-408 and ~line 484)
- Modify: `worker/requirements.txt`
- Modify: `worker/Dockerfile` (the `COPY worker/handler.py ...` line)
- Modify: `server/analysisSchema.ts:45`
- Modify: `docs/cv-worker-contract.md`
- Test: `worker/test_calibration_adapter.py`, `server/analysisCamera.test.ts`

**Interfaces:**
- Consumes: `solve_camera`, `assess_geometry`, `CORNER_LABEL_TO_KEYPOINT`, `KEYPOINTS`, `Camera.look_at`.
- Produces: `corners_to_observations(corners: list[dict], image_size: tuple[int, int]) -> dict[str, tuple[float, float]]` (UI percent coordinates → pixels keyed by court keypoint; unknown labels and non-numeric values skipped) and `summarise_calibration(corners: list[dict], image_size: tuple[int, int]) -> dict | None` returning `{cameraTier, rmsPx, focalPx, k1, looFloorCm, geometryTier, elevationDeg, visibleFraction, reasons}` (JSON-serialisable), `None` when the corners cannot be solved. The worker result gains `"camera": <that dict or None>`; the server schema gains `camera` (nullable, optional).

- [ ] **Step 1: Write the failing tests**

```python
# worker/test_calibration_adapter.py
import json
import unittest

from geometry.calibration_adapter import corners_to_observations, summarise_calibration
from geometry.camera import Camera
from geometry.court_model import CORNER_LABEL_TO_KEYPOINT, KEYPOINTS
from geometry.synthetic import LANDSCAPE

# High and far enough back that all four singles corners are inside the frame.
WIDE = Camera.look_at((2.59, -5.5, 6.0), (2.59, 6.7, 0.0), 1000.0, LANDSCAPE)


def percent_corners(camera: Camera, size: tuple[int, int]) -> list[dict]:
    corners = []
    for label, name in CORNER_LABEL_TO_KEYPOINT.items():
        x, y = camera.project(KEYPOINTS[name][None, :])[0]
        corners.append({"label": label, "x": x / size[0] * 100, "y": y / size[1] * 100})
    return corners


class CornersToObservationsTests(unittest.TestCase):
    def test_percent_coordinates_become_pixels_keyed_by_keypoint(self):
        observed = corners_to_observations([{"label": "nearLeft", "x": 50, "y": 25}], (1920, 1080))
        self.assertEqual(observed, {"back0_sl": (960.0, 270.0)})

    def test_unknown_labels_and_bad_values_are_skipped(self):
        corners = [
            {"label": "centre", "x": 10, "y": 10},
            {"label": "nearLeft", "x": "bad", "y": 10},
            {"label": "nearRight", "x": None, "y": 10},
            {"label": "farLeft", "x": float("nan"), "y": 10},
            {"label": "farRight", "x": 20, "y": 30},
        ]
        self.assertEqual(corners_to_observations(corners, (1000, 1000)), {"back1_sr": (200.0, 300.0)})


class SummariseCalibrationTests(unittest.TestCase):
    def test_four_corners_give_a_json_ready_camera_summary(self):
        corners = percent_corners(WIDE, LANDSCAPE)
        for corner in corners:
            self.assertTrue(0 <= corner["x"] <= 100 and 0 <= corner["y"] <= 100, corner)
        summary = summarise_calibration(corners, LANDSCAPE)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["cameraTier"], "approximate")
        self.assertLess(abs(summary["focalPx"] / 1000.0 - 1), 0.1)
        self.assertIn(summary["geometryTier"], ("A", "B", "C"))
        self.assertIsNone(summary["looFloorCm"])
        self.assertIsInstance(summary["reasons"], list)
        json.dumps(summary)

    def test_three_corners_are_not_solvable(self):
        corners = percent_corners(WIDE, LANDSCAPE)[:3]
        self.assertIsNone(summarise_calibration(corners, LANDSCAPE))

    def test_empty_corners_are_not_solvable(self):
        self.assertIsNone(summarise_calibration([], LANDSCAPE))


if __name__ == "__main__":
    unittest.main()
```

```ts
// server/analysisCamera.test.ts
import { describe, expect, it } from "vitest";
import { analysisResultSchema } from "./analysisSchema";

const base = {
  processingVersion: "optiqen-badminton-worker-test",
  calibration: {
    corners: [
      { label: "nearLeft", x: 10, y: 80 },
      { label: "nearRight", x: 90, y: 80 },
      { label: "farRight", x: 70, y: 20 },
    ],
    confidence: "provisional",
    supportsCourtMapping: false,
    guidance: "test",
  },
  quality: { usableFrameRatio: 0.9, poseTrackConfidence: 0.9, shuttleTrackConfidence: 0.5 },
  metrics: [],
  shotDistribution: {},
};

const camera = {
  cameraTier: "approximate",
  rmsPx: 0,
  focalPx: 1012.4,
  k1: 0,
  looFloorCm: null,
  geometryTier: "A",
  elevationDeg: 24.4,
  visibleFraction: 1,
  reasons: [],
};

describe("analysis result camera summary", () => {
  it("keeps a worker camera summary through validation", () => {
    const parsed = analysisResultSchema.parse({ ...base, camera });
    expect(parsed.camera).toEqual(camera);
  });

  it("accepts a result with a null or missing camera summary", () => {
    expect(analysisResultSchema.parse({ ...base, camera: null }).camera).toBeNull();
    expect(analysisResultSchema.parse(base).camera).toBeUndefined();
  });

  it("rejects a geometry tier outside A, B and C", () => {
    expect(analysisResultSchema.safeParse({ ...base, camera: { ...camera, geometryTier: "D" } }).success).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd worker && python -m unittest test_calibration_adapter -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'geometry.calibration_adapter'`

Run: `npx vitest run server/analysisCamera.test.ts`
Expected: FAIL — `expected undefined to deeply equal {...}` (zod strips the unknown `camera` key).

- [ ] **Step 3: Write minimal implementation**

```python
# worker/geometry/calibration_adapter.py
"""Bridge between the UI corner taps and the camera solver."""
from __future__ import annotations

import math
from typing import Any

from geometry.calibrate import solve_camera
from geometry.court_model import CORNER_LABEL_TO_KEYPOINT
from geometry.quality import assess_geometry


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def corners_to_observations(corners: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, tuple[float, float]]:
    """UI corners are percent-of-frame; return pixel observations keyed by court keypoint."""
    width, height = image_size
    observed: dict[str, tuple[float, float]] = {}
    for corner in corners:
        name = CORNER_LABEL_TO_KEYPOINT.get(corner.get("label"))
        x, y = corner.get("x"), corner.get("y")
        if name is None or not (_is_number(x) and _is_number(y)):
            continue
        observed[name] = (float(x) * width / 100, float(y) * height / 100)
    return observed


def summarise_calibration(corners: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any] | None:
    """Camera facts to attach to the accepted calibration; None when the corners cannot be solved."""
    solution = solve_camera(corners_to_observations(corners, image_size), image_size)
    if solution is None:
        return None
    quality = assess_geometry(solution.camera, image_size)
    return {
        "cameraTier": solution.tier,
        "rmsPx": round(solution.rms_px, 3),
        "focalPx": round(solution.camera.focal_px, 1),
        "k1": round(solution.camera.k1, 4),
        "looFloorCm": None if solution.loo_floor_cm is None else round(solution.loo_floor_cm, 2),
        "geometryTier": quality.tier,
        "elevationDeg": round(quality.elevation_deg, 1),
        "visibleFraction": round(quality.visible_fraction, 3),
        "reasons": list(quality.reasons),
    }
```

Edit `worker/requirements.txt` — add after the `numpy` line:

```text
scipy==1.13.1
```

Edit `worker/Dockerfile` — replace the line `COPY worker/handler.py worker/badminton.py /opt/netoval/` with:

```dockerfile
COPY worker/handler.py worker/badminton.py /opt/netoval/
COPY worker/geometry/ /opt/netoval/geometry/
```

Edit `worker/handler.py`, three places (use the Edit tool; the snippets below are exact):

1. After the `from badminton import (...)` block, add:

```python
from geometry.calibration_adapter import summarise_calibration
```

2. Replace

```python
            homography: np.ndarray | None = None
            accepted_calibration = data["calibration"]
```

with

```python
            homography: np.ndarray | None = None
            camera_summary: dict[str, Any] | None = None
            accepted_calibration = data["calibration"]
```

and replace

```python
                    if homography is None:
                        warnings.append(accepted_calibration["guidance"])
```

with

```python
                    if homography is None:
                        warnings.append(accepted_calibration["guidance"])
                    camera_summary = summarise_calibration(data["calibration"]["corners"], (width, height))
                    if camera_summary is not None and camera_summary["geometryTier"] != "A":
                        warnings.extend(camera_summary["reasons"])
```

3. In the final `return {...}`, replace `"courtType": court_type,` with:

```python
            "courtType": court_type,
            "camera": camera_summary,
```

Edit `server/analysisSchema.ts` — after the line `  courtType: z.enum(["singles", "doubles"]).optional(),` that follows the `calibration` object (the top-level one, currently line 45), add:

```ts
  camera: z.object({
    cameraTier: z.enum(["validated", "approximate", "unavailable"]),
    rmsPx: z.number().finite().nonnegative(),
    focalPx: z.number().finite().positive(),
    k1: z.number().finite(),
    looFloorCm: z.number().finite().nonnegative().nullable(),
    geometryTier: z.enum(["A", "B", "C"]),
    elevationDeg: z.number().finite(),
    visibleFraction: z.number().min(0).max(1),
    reasons: z.array(z.string().min(1).max(300)).max(8),
  }).nullable().optional(),
```

Edit `docs/cv-worker-contract.md` — in the "Job status and result" table, add this row after the "Court registration" row:

```markdown
| Camera solution | `camera`: tier (`validated`/`approximate`/`unavailable`), pixel RMS, focal length, lens term, leave-one-out floor error, capture-geometry tier (`A`/`B`/`C`) with reasons | `null` when fewer than four court points could be solved. Tier `A` is required for full 3D claims; `B` and `C` limit claims per `docs/design/specs/2026-10-01-badminton-understanding-design.md` §5. |
```

- [ ] **Step 4: Run tests and checks to verify they pass**

Run: `cd worker && python -m unittest test_calibration_adapter -v`
Expected: PASS (5 tests). If `test_four_corners_give_a_json_ready_camera_summary` fails on the in-frame assertion, move `WIDE` further back/higher until all four corners are inside the frame; the camera, not the assertion, is wrong.

Run: `npx vitest run server/analysisCamera.test.ts`
Expected: PASS (3 tests)

Run: `python -c "import ast; ast.parse(open('worker/handler.py', encoding='utf-8').read())"`
Expected: no output (handler still parses; the worker cannot be imported locally without `runpod`/`ultralytics`).

Run the full worker suite and the server suite:
`cd worker && python -m unittest discover -p "test_*.py" -v` → all PASS (existing `test_badminton.py` plus the new files).
`npx vitest run` and `npx tsc --noEmit` → no new failures or type errors.

- [ ] **Step 5: Commit**

The new files, `worker/requirements.txt` and `server/analysisSchema.ts` are clean to commit. `worker/handler.py`, `worker/Dockerfile` and `docs/cv-worker-contract.md` contain the user's earlier uncommitted edits, so leave them unstaged and tell the user they are ready for review.

```bash
git add worker/geometry/calibration_adapter.py worker/test_calibration_adapter.py worker/requirements.txt server/analysisSchema.ts server/analysisCamera.test.ts docs/design
git commit -m "feat: report camera solution and capture tier in worker results"
```

---

### Task 9: Browser labeller for golden clips

**Files:**
- Create: `tools/labeller/labeller-core.js`
- Create: `tools/labeller/index.html`
- Modify: `vitest.config.ts` (add `tools/**/*.test.ts` to `test.include`)
- Modify: `docs/golden-set.md` (add a "The labeller" section before "## Labelling a clip")
- Test: `tools/labeller/labeller-core.test.ts`

**Interfaces:**
- Consumes: the golden schema from Task 7 (`validate_golden`, `GOLDEN_SHOT_LABELS`) and the court keypoint names from Task 1 — the test checks both against the Python source of truth so the tool cannot drift.
- Produces: `window.LabellerCore` (also `globalThis.LabellerCore` under vitest) with `SHOT_LABELS`, `KEYPOINT_NAMES`, `emptyState(clipId, fps, imageSize)`, `addContact(state, timeMs, label?) -> index`, `setLabel(state, index, label)`, `removeContact(state, index)`, `nearestContactIndex(state, timeMs, toleranceMs?) -> index | -1`, `startRally(state, timeMs)`, `endRally(state, timeMs)`, `setWinner(state, winner)`, `setKeypoint(state, name, x, y)`, `removeKeypoint(state, name)`, `importWorkerResult(state, result) -> addedCount`, `buildGolden(state) -> {doc, warnings}`, `stateFromGolden(doc)`, `frameIndexAt(sec, fps)`, `frameTimeMs(index, fps)`, `frameCentreSeconds(index, fps)`, `stepSeconds(currentSec, frames, fps, durationSec)`, `fpsFromMediaTimes(times)`. State is `{clipId, fps, imageSize:[w,h], keypoints:{name:{x,y}}, contacts:[{timeMs,label}], rallies:[{startMs,endMs|null,winner}]}`. `tools/labeller/index.html` is a zero-install page: open it in Chrome or Edge, no server.

- [ ] **Step 1: Write the failing test and register the folder with vitest**

In `vitest.config.ts`, replace the `include` line with:

```ts
    include: ["server/**/*.test.ts", "server/**/*.spec.ts", "client/src/**/*.test.ts", "client/src/**/*.test.tsx", "tools/**/*.test.ts"],
```

```ts
// tools/labeller/labeller-core.test.ts
// @ts-nocheck
import { execFileSync } from "node:child_process";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

const workerDir = path.resolve(import.meta.dirname, "..", "..", "worker");
const python = process.env.PYTHON ?? "python";

function runPython(code: string, input?: string): string {
  return execFileSync(python, ["-c", code], { cwd: workerDir, input, encoding: "utf8" });
}

let core: any;

beforeAll(async () => {
  await import("./labeller-core.js");
  core = (globalThis as any).LabellerCore;
});

function freshState() {
  return core.emptyState("clip-001", 60, [1920, 1080]);
}

describe("labeller core: sources of truth", () => {
  it("uses the Python court model's keypoint names in the same order", () => {
    const names = JSON.parse(runPython("import json; from geometry.court_model import KEYPOINTS; print(json.dumps(list(KEYPOINTS)))"));
    expect(core.KEYPOINT_NAMES).toEqual(names);
  });

  it("uses the Python golden shot labels in the same order", () => {
    const labels = JSON.parse(runPython("import json; from evaluation.golden import GOLDEN_SHOT_LABELS; print(json.dumps(list(GOLDEN_SHOT_LABELS)))"));
    expect(core.SHOT_LABELS).toEqual(labels);
  });
});

describe("labeller core: contacts", () => {
  it("keeps contacts sorted, defaults the label to other, and merges taps within 20 ms", () => {
    const state = freshState();
    expect(core.addContact(state, 2000)).toBe(0);
    expect(core.addContact(state, 1000)).toBe(0);
    expect(state.contacts.map((c: any) => c.timeMs)).toEqual([1000, 2000]);
    expect(core.addContact(state, 1010)).toBe(0);
    expect(state.contacts).toHaveLength(2);
    expect(state.contacts[0].label).toBe("other");
  });

  it("rejects unknown labels and missing contacts", () => {
    const state = freshState();
    core.addContact(state, 1000);
    expect(() => core.setLabel(state, 0, "bogus")).toThrow("Unknown shot label");
    expect(() => core.setLabel(state, 5, "clear")).toThrow("No contact");
    core.setLabel(state, 0, "smash");
    expect(state.contacts[0].label).toBe("smash");
  });

  it("removes contacts and finds the nearest one within a tolerance", () => {
    const state = freshState();
    core.addContact(state, 1000);
    core.addContact(state, 2000);
    expect(core.nearestContactIndex(state, 1900, 150)).toBe(1);
    expect(core.nearestContactIndex(state, 1500, 100)).toBe(-1);
    core.removeContact(state, 0);
    expect(state.contacts.map((c: any) => c.timeMs)).toEqual([2000]);
  });
});

describe("labeller core: rallies", () => {
  it("ends the open rally after its start and applies the winner to the latest rally", () => {
    const state = freshState();
    core.startRally(state, 1000);
    expect(() => core.endRally(state, 1000)).toThrow("after it starts");
    core.endRally(state, 4000);
    core.setWinner(state, "near");
    core.startRally(state, 6000);
    core.endRally(state, 9000);
    core.setWinner(state, "far");
    expect(state.rallies).toEqual([
      { startMs: 1000, endMs: 4000, winner: "near" },
      { startMs: 6000, endMs: 9000, winner: "far" },
    ]);
  });

  it("restarting an open rally moves its start instead of adding another", () => {
    const state = freshState();
    core.startRally(state, 1000);
    core.startRally(state, 1500);
    expect(state.rallies).toHaveLength(1);
    expect(state.rallies[0].startMs).toBe(1500);
  });

  it("rejects an end with no open rally, a bad winner, and a winner with no rally", () => {
    const state = freshState();
    expect(() => core.endRally(state, 100)).toThrow("No rally is open");
    expect(() => core.setWinner(state, "near")).toThrow("No rally");
    core.startRally(state, 100);
    expect(() => core.setWinner(state, "left")).toThrow("Unknown winner");
  });

  it("leaves an open rally out of the export and says so", () => {
    const state = freshState();
    core.startRally(state, 1000);
    const { doc, warnings } = core.buildGolden(state);
    expect(doc.rallies).toEqual([]);
    expect(warnings.some((w: string) => w.includes("no end"))).toBe(true);
  });
});

describe("labeller core: court keypoints", () => {
  it("stores rounded pixel positions and replaces a re-clicked point", () => {
    const state = freshState();
    core.setKeypoint(state, "back1_sl", 612.123, 301.567);
    core.setKeypoint(state, "back1_sl", 700, 300);
    expect(state.keypoints).toEqual({ back1_sl: { x: 700, y: 300 } });
    core.removeKeypoint(state, "back1_sl");
    expect(state.keypoints).toEqual({});
  });

  it("rejects names outside the court model", () => {
    expect(() => core.setKeypoint(freshState(), "nope", 1, 1)).toThrow("Unknown court keypoint");
  });

  it("rejects non-finite times and pixel positions instead of storing them", () => {
    const state = freshState();
    expect(() => core.addContact(state, NaN)).toThrow("finite");
    expect(() => core.startRally(state, Infinity)).toThrow("finite");
    expect(() => core.setKeypoint(state, "back1_sl", NaN, 5)).toThrow("finite");
    expect(() => core.setKeypoint(state, "back1_sl", 5, Infinity)).toThrow("finite");
    expect(state.contacts).toEqual([]);
    expect(state.rallies).toEqual([]);
    expect(state.keypoints).toEqual({});
  });
});

describe("labeller core: video time", () => {
  it("maps video time to frame indexes whether the time is a frame start or centre", () => {
    expect(core.frameIndexAt(core.frameCentreSeconds(7, 60), 60)).toBe(7);
    expect(core.frameIndexAt(7 / 60, 60)).toBe(7);
    expect(core.frameIndexAt(7 / 60 - 0.001, 60)).toBe(7);
    expect(core.frameIndexAt(-1, 60)).toBe(0);
  });

  it("converts a frame index to milliseconds", () => {
    expect(core.frameTimeMs(60, 60)).toBe(1000);
    expect(core.frameTimeMs(1, 29.97)).toBe(33);
  });

  it("steps whole frames and clamps to the clip", () => {
    expect(core.stepSeconds(core.frameCentreSeconds(10, 60), 1, 60, 10)).toBeCloseTo(core.frameCentreSeconds(11, 60), 9);
    expect(core.stepSeconds(0.01, -5, 60, 10)).toBeCloseTo(core.frameCentreSeconds(0, 60), 9);
    expect(core.stepSeconds(9.99, 100, 60, 10)).toBeCloseTo(core.frameCentreSeconds(599, 60), 9);
  });

  it("detects fps from media timestamps even with a dropped frame", () => {
    const times = [0, 1, 2, 3, 5, 6, 7, 8].map(n => n / 60);
    expect(core.fpsFromMediaTimes(times)).toBe(60);
    expect(core.fpsFromMediaTimes([0, 1 / 30])).toBeNull();
    expect(core.fpsFromMediaTimes([])).toBeNull();
  });
});

describe("labeller core: worker import and export", () => {
  it("imports contacts, uses verified shot labels only, and never duplicates", () => {
    const state = freshState();
    core.addContact(state, 1005, "serve");
    const result = {
      events: [
        { type: "contact", timeMs: 1000 },
        { type: "contact", timeMs: 2000 },
        { type: "contact", timeMs: 3000 },
        { type: "split_step", timeMs: 1500 },
      ],
      shots: [
        { timeMs: 2010, label: "clear", verified: true },
        { timeMs: 3010, label: "smash", verified: false },
      ],
    };
    expect(core.importWorkerResult(state, result)).toBe(2);
    expect(state.contacts).toEqual([
      { timeMs: 1005, label: "serve" },
      { timeMs: 2000, label: "clear" },
      { timeMs: 3000, label: "other" },
    ]);
  });

  it("exports a golden document the Python validator accepts", () => {
    const state = freshState();
    core.setKeypoint(state, "back1_sl", 612, 301.5);
    core.setKeypoint(state, "post_left_top", 400, 200);
    core.addContact(state, 1200, "serve");
    core.addContact(state, 2100, "clear");
    core.startRally(state, 800);
    core.endRally(state, 4300);
    core.setWinner(state, "near");
    const { doc, warnings } = core.buildGolden(state);
    expect(warnings).toEqual([]);
    expect(doc.schemaVersion).toBe(1);
    expect(doc.shots).toEqual([
      { timeMs: 1200, label: "serve", landing: null },
      { timeMs: 2100, label: "clear", landing: null },
    ]);
    const errors = JSON.parse(
      runPython("import json,sys; from evaluation.golden import validate_golden; print(json.dumps(validate_golden(json.load(sys.stdin))))", JSON.stringify(doc)),
    );
    expect(errors).toEqual([]);
  });

  it("round-trips through the golden format", () => {
    const state = freshState();
    core.setKeypoint(state, "back0_sr", 1310, 298);
    core.addContact(state, 1200, "drop");
    core.addContact(state, 2100, "net");
    core.startRally(state, 800);
    core.endRally(state, 4300);
    core.setWinner(state, "far");
    const { doc } = core.buildGolden(state);
    expect(core.stateFromGolden(doc)).toEqual(state);
  });

  it("warns about contacts still labelled other", () => {
    const state = freshState();
    core.addContact(state, 1000);
    const { warnings } = core.buildGolden(state);
    expect(warnings.some((w: string) => w.includes('"other"'))).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tools/labeller/labeller-core.test.ts`
Expected: FAIL — `Failed to resolve import "./labeller-core.js"`

- [ ] **Step 3: Write the implementation**

```js
// tools/labeller/labeller-core.js
// Pure labelling logic for the golden-set labeller. No DOM access, so it is unit-tested
// under vitest and loaded as a classic script by index.html (works from file://).
(function (global, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.LabellerCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Keep these in sync with worker/geometry/court_model.py and worker/evaluation/golden.py;
  // labeller-core.test.ts fails if they drift.
  const SHOT_LABELS = ["serve", "clear", "drop", "net", "lift", "drive", "push", "smash", "block", "other"];
  const ROWS = ["back0", "long0", "short0", "short1", "long1", "back1"];
  const COLUMNS = ["dl", "sl", "c", "sr", "dr"];
  const KEYPOINT_NAMES = [
    ...ROWS.flatMap(row => COLUMNS.map(column => `${row}_${column}`)),
    "post_left_base",
    "post_left_top",
    "post_right_base",
    "post_right_top",
    "net_centre_top",
  ];
  const WINNERS = ["near", "far", "unknown"];
  const MERGE_WINDOW_MS = 20;

  const round2 = n => Math.round(n * 100) / 100;

  // A collapsed window or hidden tab gives the page zero-size layout, so coordinate maths
  // can yield NaN/Infinity. JSON.stringify would turn those into null in the export.
  function finite(value, what) {
    if (!Number.isFinite(value)) throw new Error(`${what} must be a finite number`);
    return value;
  }

  function emptyState(clipId, fps, imageSize) {
    return { clipId, fps, imageSize: [imageSize[0], imageSize[1]], keypoints: {}, contacts: [], rallies: [] };
  }

  function addContact(state, timeMs, label) {
    const time = Math.round(finite(timeMs, "Contact time"));
    const existing = state.contacts.findIndex(contact => Math.abs(contact.timeMs - time) <= MERGE_WINDOW_MS);
    if (existing >= 0) return existing;
    state.contacts.push({ timeMs: time, label: label || "other" });
    state.contacts.sort((a, b) => a.timeMs - b.timeMs);
    return state.contacts.findIndex(contact => contact.timeMs === time);
  }

  function setLabel(state, index, label) {
    if (!SHOT_LABELS.includes(label)) throw new Error(`Unknown shot label: ${label}`);
    if (!state.contacts[index]) throw new Error(`No contact at index ${index}`);
    state.contacts[index].label = label;
  }

  function removeContact(state, index) {
    state.contacts.splice(index, 1);
  }

  function nearestContactIndex(state, timeMs, toleranceMs = Infinity) {
    let best = -1;
    let bestDistance = Infinity;
    state.contacts.forEach((contact, index) => {
      const distance = Math.abs(contact.timeMs - timeMs);
      if (distance <= toleranceMs && distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  }

  function startRally(state, timeMs) {
    const time = Math.round(finite(timeMs, "Rally start"));
    const open = state.rallies.find(rally => rally.endMs === null);
    if (open) {
      open.startMs = time;
      return;
    }
    state.rallies.push({ startMs: time, endMs: null, winner: "unknown" });
  }

  function endRally(state, timeMs) {
    const open = state.rallies.find(rally => rally.endMs === null);
    if (!open) throw new Error("No rally is open. Press R to start one first.");
    const time = Math.round(finite(timeMs, "Rally end"));
    if (time <= open.startMs) throw new Error("A rally must end after it starts.");
    open.endMs = time;
  }

  function setWinner(state, winner) {
    if (!WINNERS.includes(winner)) throw new Error(`Unknown winner: ${winner}`);
    if (state.rallies.length === 0) throw new Error("No rally to set a winner on.");
    state.rallies[state.rallies.length - 1].winner = winner;
  }

  function setKeypoint(state, name, x, y) {
    if (!KEYPOINT_NAMES.includes(name)) throw new Error(`Unknown court keypoint: ${name}`);
    state.keypoints[name] = { x: round2(finite(x, "Pixel x")), y: round2(finite(y, "Pixel y")) };
  }

  function removeKeypoint(state, name) {
    delete state.keypoints[name];
  }

  function importWorkerResult(state, result) {
    const shots = result.shots || [];
    let added = 0;
    for (const event of result.events || []) {
      if (event.type !== "contact" || typeof event.timeMs !== "number") continue;
      const shot = shots.find(candidate => Math.abs(candidate.timeMs - event.timeMs) <= 100);
      const label = shot && shot.verified && SHOT_LABELS.includes(shot.label) ? shot.label : "other";
      const before = state.contacts.length;
      addContact(state, event.timeMs, label);
      if (state.contacts.length > before) added += 1;
    }
    return added;
  }

  function buildGolden(state) {
    const warnings = [];
    const rallies = [];
    for (const rally of state.rallies) {
      if (rally.endMs === null) {
        warnings.push(`Rally starting at ${rally.startMs} ms has no end and was left out of the export.`);
        continue;
      }
      rallies.push({ startMs: rally.startMs, endMs: rally.endMs, winner: rally.winner });
    }
    const unlabelled = state.contacts.filter(contact => contact.label === "other").length;
    if (unlabelled > 0) warnings.push(`${unlabelled} contact(s) are still labelled "other".`);
    const doc = {
      schemaVersion: 1,
      clipId: state.clipId,
      sourceFps: state.fps,
      imageSize: [state.imageSize[0], state.imageSize[1]],
      courtKeypoints: Object.entries(state.keypoints).map(([name, point]) => ({ name, x: point.x, y: point.y })),
      contacts: state.contacts.map(contact => ({ timeMs: contact.timeMs })),
      shots: state.contacts.map(contact => ({ timeMs: contact.timeMs, label: contact.label, landing: null })),
      rallies,
    };
    return { doc, warnings };
  }

  function stateFromGolden(doc) {
    const state = emptyState(doc.clipId, doc.sourceFps, doc.imageSize);
    for (const item of doc.courtKeypoints || []) state.keypoints[item.name] = { x: item.x, y: item.y };
    const labelAt = new Map((doc.shots || []).map(shot => [shot.timeMs, shot.label]));
    for (const contact of doc.contacts || []) {
      state.contacts.push({ timeMs: contact.timeMs, label: labelAt.get(contact.timeMs) || "other" });
    }
    state.contacts.sort((a, b) => a.timeMs - b.timeMs);
    for (const rally of doc.rallies || []) {
      state.rallies.push({ startMs: rally.startMs, endMs: rally.endMs, winner: rally.winner });
    }
    return state;
  }

  // Video time <-> frames. A time anywhere within ~a quarter frame of a frame start or its
  // centre maps to that frame, which tolerates timestamp jitter in phone footage.
  function frameIndexAt(seconds, fps) {
    return Math.max(0, Math.floor(seconds * fps + 0.25));
  }

  function frameTimeMs(index, fps) {
    return Math.round((index * 1000) / fps);
  }

  function frameCentreSeconds(index, fps) {
    return (index + 0.5) / fps;
  }

  function stepSeconds(currentSeconds, frames, fps, durationSeconds) {
    const last = Math.max(0, Math.ceil(durationSeconds * fps) - 1);
    const index = Math.min(last, Math.max(0, frameIndexAt(currentSeconds, fps) + frames));
    return frameCentreSeconds(index, fps);
  }

  function fpsFromMediaTimes(times) {
    const deltas = [];
    for (let i = 1; i < times.length; i += 1) {
      const delta = times[i] - times[i - 1];
      if (delta > 0) deltas.push(delta);
    }
    if (deltas.length < 2) return null;
    deltas.sort((a, b) => a - b);
    return Math.round(1000 / deltas[Math.floor(deltas.length / 2)]) / 1000;
  }

  return {
    SHOT_LABELS,
    KEYPOINT_NAMES,
    emptyState,
    addContact,
    setLabel,
    removeContact,
    nearestContactIndex,
    startRally,
    endRally,
    setWinner,
    setKeypoint,
    removeKeypoint,
    importWorkerResult,
    buildGolden,
    stateFromGolden,
    frameIndexAt,
    frameTimeMs,
    frameCentreSeconds,
    stepSeconds,
    fpsFromMediaTimes,
  };
});
```

```html
<!-- tools/labeller/index.html -->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden Set Labeller</title>
<style>
  :root { --bg:#f6f7f9; --panel:#fff; --ink:#15181d; --muted:#5d6672; --line:#d9dde3; --accent:#1f6feb; --ok:#1a7f37; --warn:#9a6700; --mark:#ffcc00; }
  @media (prefers-color-scheme: dark) { :root { --bg:#0f1216; --panel:#171b21; --ink:#e8ebef; --muted:#9aa4b0; --line:#2a3039; --accent:#5aa2ff; --ok:#3fb950; --warn:#d29922; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui, sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
  header label.field { display:flex; gap:6px; align-items:center; color:var(--muted); }
  input, select, button, .btn { font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:5px 9px; }
  .btn, button { cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  input[type=number] { width:90px; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; padding:16px; }
  @media (max-width: 960px) { main { grid-template-columns:1fr; } }
  #wrap { position:relative; display:inline-block; max-width:100%; background:#000; line-height:0; }
  video { max-width:100%; max-height:70vh; display:block; }
  canvas { position:absolute; left:0; top:0; cursor:crosshair; }
  #readout { margin:8px 0; color:var(--muted); font-variant-numeric:tabular-nums; }
  #status { color:var(--muted); }
  #status.error { color:#cf222e; }
  #warnings { color:var(--warn); margin:6px 0; }
  aside section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:12px; }
  h2 { margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  td, th { padding:3px 4px; text-align:left; border-bottom:1px solid var(--line); }
  tr.selected { background:color-mix(in srgb, var(--accent) 16%, transparent); }
  tr[data-index] { cursor:pointer; }
  ul { list-style:none; margin:0; padding:0; }
  li { display:flex; gap:8px; align-items:center; padding:2px 0; }
  .hint { color:var(--muted); margin:6px 0 0; font-size:12px; }
  kbd { background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:0 5px; font:12px ui-monospace, monospace; }
  .x { margin-left:auto; padding:0 7px; }
</style>
</head>
<body>
<header>
  <label class="btn">Open video <input id="video-file" type="file" accept="video/*" hidden></label>
  <label class="btn">Load labels <input id="labels-file" type="file" accept=".json,application/json" hidden></label>
  <label class="btn">Load worker result <input id="result-file" type="file" accept=".json,application/json" hidden></label>
  <label class="field">Clip ID <input id="clip-id" size="16" placeholder="clip-001"></label>
  <label class="field">FPS <input id="fps" type="number" min="1" step="0.001" value="30"></label>
  <button id="detect-fps" title="Play a moment of the clip to measure its frame rate">Detect FPS</button>
  <button id="save" class="primary">Save labels</button>
  <span id="status">Open a video to begin.</span>
</header>
<main>
  <section>
    <div id="wrap"><video id="video" muted playsinline></video><canvas id="overlay"></canvas></div>
    <div id="readout">frame <b id="frame-no">0</b> · <b id="time-ms">0</b> ms · speed <b id="speed">1</b>×</div>
    <div id="warnings"></div>
  </section>
  <aside>
    <section>
      <h2>Court points</h2>
      <select id="kp-select"></select>
      <p class="hint">Pick a name, then click that point on the paused frame. Use a frame where the court is clearly visible.</p>
      <ul id="kp-list"></ul>
    </section>
    <section>
      <h2>Contacts</h2>
      <table><thead><tr><th>#</th><th>ms</th><th>frame</th><th>shot</th><th></th></tr></thead><tbody id="contacts"></tbody></table>
    </section>
    <section>
      <h2>Rallies</h2>
      <ul id="rallies"></ul>
    </section>
    <section>
      <h2>Keys</h2>
      <p class="hint"><kbd>Space</kbd> play/pause · <kbd>,</kbd> <kbd>.</kbd> step 1 frame · <kbd>&lt;</kbd> <kbd>&gt;</kbd> step 10 · <kbd>[</kbd> <kbd>]</kbd> slower/faster<br>
      <kbd>C</kbd> mark contact · <kbd>1</kbd>–<kbd>9</kbd>,<kbd>0</kbd> shot: serve clear drop net lift drive push smash block other<br>
      <kbd>R</kbd> start/end rally · <kbd>N</kbd> <kbd>F</kbd> <kbd>U</kbd> winner near/far/unknown · <kbd>Del</kbd> delete contact</p>
    </section>
  </aside>
</main>
<script src="labeller-core.js"></script>
<script>
(() => {
  "use strict";
  const C = window.LabellerCore;
  const $ = id => document.getElementById(id);
  const video = $("video");
  const overlay = $("overlay");
  const ctx = overlay.getContext("2d");
  let state = C.emptyState("", 30, [0, 0]);
  let selectedContact = -1;

  const fps = () => {
    const value = Number($("fps").value);
    return Number.isFinite(value) && value > 0 ? value : 30;
  };
  const storageKey = () => `optiqen-labeller:${state.clipId}`;
  const currentMs = () => C.frameTimeMs(C.frameIndexAt(video.currentTime, fps()), fps());

  function say(message, isError = false) {
    $("status").textContent = message;
    $("status").className = isError ? "error" : "";
  }

  function persist() {
    state.clipId = $("clip-id").value.trim();
    state.fps = fps();
    if (!state.clipId) return;
    try { localStorage.setItem(storageKey(), JSON.stringify(state)); } catch { /* storage unavailable: manual save still works */ }
  }

  function seekToMs(ms) {
    video.pause();
    video.currentTime = C.frameCentreSeconds(C.frameIndexAt(ms / 1000, fps()), fps());
  }

  function step(frames) {
    video.pause();
    video.currentTime = C.stepSeconds(video.currentTime, frames, fps(), video.duration || 0);
  }

  function renderKeypoints() {
    const select = $("kp-select");
    if (!select.options.length) {
      for (const name of C.KEYPOINT_NAMES) select.add(new Option(name, name));
    }
    for (const option of select.options) option.textContent = (state.keypoints[option.value] ? "✓ " : "") + option.value;
    $("kp-list").innerHTML = Object.keys(state.keypoints)
      .map(name => `<li><span>${name}</span><button class="x" data-remove-kp="${name}" title="Remove">×</button></li>`).join("");
  }

  function renderContacts() {
    $("contacts").innerHTML = state.contacts.map((contact, index) => {
      const options = C.SHOT_LABELS.map(label => `<option${label === contact.label ? " selected" : ""}>${label}</option>`).join("");
      const frame = C.frameIndexAt(contact.timeMs / 1000, fps());
      return `<tr data-index="${index}" class="${index === selectedContact ? "selected" : ""}"><td>${index + 1}</td><td>${contact.timeMs}</td><td>${frame}</td>` +
        `<td><select data-label="${index}">${options}</select></td><td><button data-remove="${index}" title="Delete">×</button></td></tr>`;
    }).join("");
  }

  function renderRallies() {
    $("rallies").innerHTML = state.rallies.map((rally, index) => {
      const options = ["near", "far", "unknown"].map(w => `<option${w === rally.winner ? " selected" : ""}>${w}</option>`).join("");
      return `<li><span>${rally.startMs}–${rally.endMs === null ? "open" : rally.endMs} ms</span><select data-winner="${index}">${options}</select></li>`;
    }).join("");
  }

  function drawOverlay() {
    // Size the canvas from the video's exact (fractional) box so click positions map to
    // video pixels without a rounding offset.
    const box = video.getBoundingClientRect();
    overlay.width = Math.round(box.width);
    overlay.height = Math.round(box.height);
    overlay.style.width = `${box.width}px`;
    overlay.style.height = `${box.height}px`;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!video.videoWidth) return;
    const scale = overlay.width / video.videoWidth;
    ctx.font = "12px system-ui, sans-serif";
    for (const [name, point] of Object.entries(state.keypoints)) {
      const x = point.x * scale, y = point.y * scale;
      ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();
      ctx.fillStyle = "#ffcc00"; ctx.fillText(name, x + 9, y - 8);
    }
  }

  function render() {
    renderKeypoints(); renderContacts(); renderRallies(); drawOverlay();
    $("warnings").innerHTML = C.buildGolden(state).warnings.map(w => `<div>⚠ ${w}</div>`).join("");
    persist();
  }

  function updateReadout() {
    $("frame-no").textContent = C.frameIndexAt(video.currentTime, fps());
    $("time-ms").textContent = currentMs();
    $("speed").textContent = video.playbackRate;
  }

  function download(name, text) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  // Playback is the only way a browser exposes frame timestamps. Chrome refuses to play
  // video in a hidden tab, so a failed detection must leave nothing behind that could
  // pause or rewind the clip later: the pending frame callback is always cancelled.
  let detectHandle = null;
  function stopDetect() {
    if (detectHandle !== null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(detectHandle);
    detectHandle = null;
  }

  function detectFps() {
    if (!("requestVideoFrameCallback" in video)) { say("This browser cannot detect FPS — enter it manually.", true); return; }
    if (!video.videoWidth) { say("Open a video first.", true); return; }
    stopDetect();
    const stamps = [];
    const onFrame = (_now, meta) => {
      stamps.push(meta.mediaTime);
      if (stamps.length < 24) { detectHandle = video.requestVideoFrameCallback(onFrame); return; }
      detectHandle = null;
      video.pause();
      const detected = C.fpsFromMediaTimes(stamps);
      if (detected) { $("fps").value = detected; state.fps = detected; say(`Detected ${detected} fps — check it matches the phone's setting.`); }
      else say("Could not detect FPS — enter it manually.", true);
      video.currentTime = 0;
      render();
    };
    detectHandle = video.requestVideoFrameCallback(onFrame);
    video.currentTime = 0;
    video.play().catch(() => {
      stopDetect();
      say("Could not play the clip to detect FPS (keep this tab in front and press Detect FPS) — or enter it manually.", true);
    });
  }

  $("video-file").addEventListener("change", event => {
    const file = event.target.files[0];
    if (!file) return;
    video.src = URL.createObjectURL(file);
    $("clip-id").value = file.name.replace(/\.[^.]+$/, "");
    video.addEventListener("loadedmetadata", () => {
      state = C.emptyState($("clip-id").value, fps(), [video.videoWidth, video.videoHeight]);
      try {
        const saved = JSON.parse(localStorage.getItem(storageKey()) || "null");
        if (saved && saved.clipId === state.clipId) { state = saved; state.imageSize = [video.videoWidth, video.videoHeight]; $("fps").value = state.fps; say(`Restored autosaved labels for ${state.clipId}.`); }
      } catch { /* ignore a corrupt autosave */ }
      selectedContact = -1;
      render();
      if (!state.contacts.length && !Object.keys(state.keypoints).length) detectFps();
    }, { once: true });
  });

  $("labels-file").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const doc = JSON.parse(await file.text());
      if (doc.schemaVersion !== 1) throw new Error("not a schemaVersion 1 labels file");
      state = C.stateFromGolden(doc);
      $("clip-id").value = state.clipId;
      $("fps").value = state.fps;
      selectedContact = -1;
      say(`Loaded ${state.contacts.length} contacts for ${state.clipId}.`);
      render();
    } catch (error) { say(`Could not load labels: ${error.message}`, true); }
    event.target.value = "";
  });

  $("result-file").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const added = C.importWorkerResult(state, JSON.parse(await file.text()));
      say(`Imported ${added} contact(s) from the worker result — review each one before saving.`);
      render();
    } catch (error) { say(`Could not read the worker result: ${error.message}`, true); }
    event.target.value = "";
  });

  $("save").addEventListener("click", () => {
    const clipId = $("clip-id").value.trim();
    if (!clipId) { say("Enter a clip ID first.", true); return; }
    persist();
    const { doc, warnings } = C.buildGolden(state);
    download(`${clipId}.json`, JSON.stringify(doc, null, 2));
    say(warnings.length ? `Saved with ${warnings.length} warning(s) — see below.` : `Saved ${clipId}.json`);
  });

  $("clip-id").addEventListener("change", persist);
  $("detect-fps").addEventListener("click", detectFps);
  $("fps").addEventListener("change", () => { state.fps = fps(); render(); });
  overlay.addEventListener("click", event => {
    if (!video.videoWidth) return;
    const rect = overlay.getBoundingClientRect();
    if (!rect.width || !rect.height) { say("The video has no visible size yet — make the window larger and try again.", true); return; }
    try {
      C.setKeypoint(state, $("kp-select").value, ((event.clientX - rect.left) * video.videoWidth) / rect.width, ((event.clientY - rect.top) * video.videoHeight) / rect.height);
    } catch (error) { say(error.message, true); return; }
    const next = C.KEYPOINT_NAMES.find(candidate => !state.keypoints[candidate]);
    if (next) $("kp-select").value = next;
    render();
  });

  $("kp-list").addEventListener("click", event => {
    const name = event.target.dataset.removeKp;
    if (name) { C.removeKeypoint(state, name); render(); }
  });
  $("contacts").addEventListener("click", event => {
    const remove = event.target.dataset.remove;
    if (remove !== undefined) { C.removeContact(state, Number(remove)); selectedContact = -1; render(); return; }
    if (event.target.dataset.label !== undefined) return;
    const row = event.target.closest("tr[data-index]");
    if (row) { selectedContact = Number(row.dataset.index); seekToMs(state.contacts[selectedContact].timeMs); render(); }
  });
  $("contacts").addEventListener("change", event => {
    const index = event.target.dataset.label;
    if (index !== undefined) { C.setLabel(state, Number(index), event.target.value); render(); }
  });
  $("rallies").addEventListener("change", event => {
    const index = event.target.dataset.winner;
    if (index !== undefined) { state.rallies[Number(index)].winner = event.target.value; render(); }
  });

  document.addEventListener("keydown", event => {
    if (event.target.closest("input, select, textarea") || event.ctrlKey || event.metaKey || event.altKey) return;
    const key = event.key;
    let handled = true;
    try {
      if (key === " ") { if (video.paused) video.play(); else video.pause(); }
      else if (key === ",") step(-1);
      else if (key === ".") step(1);
      else if (key === "<") step(-10);
      else if (key === ">") step(10);
      else if (key === "[") video.playbackRate = Math.max(0.1, video.playbackRate / 2);
      else if (key === "]") video.playbackRate = Math.min(2, video.playbackRate * 2);
      else if (key === "c" || key === "C") { video.pause(); selectedContact = C.addContact(state, currentMs()); render(); }
      else if (/^[0-9]$/.test(key)) {
        if (selectedContact < 0) throw new Error("Select or mark a contact first (C).");
        C.setLabel(state, selectedContact, key === "0" ? "other" : C.SHOT_LABELS[Number(key) - 1]); render();
      }
      else if (key === "r" || key === "R") {
        if (state.rallies.some(rally => rally.endMs === null)) C.endRally(state, currentMs()); else C.startRally(state, currentMs());
        render();
      }
      else if (key === "n" || key === "N") { C.setWinner(state, "near"); render(); }
      else if (key === "f" || key === "F") { C.setWinner(state, "far"); render(); }
      else if (key === "u" || key === "U") { C.setWinner(state, "unknown"); render(); }
      else if (key === "Delete" || key === "Backspace") {
        if (selectedContact >= 0) { C.removeContact(state, selectedContact); selectedContact = -1; render(); }
      }
      else handled = false;
    } catch (error) { say(error.message, true); }
    if (handled) event.preventDefault();
    updateReadout();
  });

  video.addEventListener("seeked", updateReadout);
  video.addEventListener("loadeddata", () => { drawOverlay(); updateReadout(); });
  window.addEventListener("resize", drawOverlay);
  setInterval(updateReadout, 100);
  render();
})();
</script>
</body>
</html>
```

Add to `docs/golden-set.md`, immediately before the line `## Labelling a clip`:

```markdown
## The labeller

`tools/labeller/index.html` is a zero-install browser tool that writes this schema
directly — open the file in Chrome or Edge (no server needed). Open the clip, check
the detected FPS against the phone's setting (keep the tab in front while it
detects; press **Detect FPS** to retry, or type the value), then:

- step with `,` and `.` (one frame) or `<` and `>` (ten); play at 0.25× with `[`;
- press `C` at each contact, then `1`–`9`/`0` for the shot type
  (serve clear drop net lift drive push smash block other);
- press `R` at a rally's start and again at its end, then `N`/`F`/`U` for the
  winner (near/far/unknown, as seen from the camera);
- pick a court-point name and click it on a clear, paused frame;
- **Save labels** downloads `<clipId>.json`. Work autosaves in the browser, and
  **Load labels** resumes a saved file.

To pre-label, run the clip through the worker first and use **Load worker
result**: its detected contacts appear as `other`/verified-shot labels for you to
correct, which is much faster than starting from nothing. Shuttle landing points
are left `null` in this version.
```

- [ ] **Step 4: Run tests and open the tool to verify**

Run: `npx vitest run tools/labeller/labeller-core.test.ts`
Expected: PASS (20 tests; needs the Task 1 and Task 7 Python modules on disk and `python` on PATH, or set `PYTHON`).

Run: `cd worker && python -m unittest test_golden -v`
Expected: PASS (the guide's JSON example is still the first `json` block).

Manual check (the HTML is the one part unit tests cannot see): open `tools/labeller/index.html` in Chrome or Edge, open `uploads/Hendry_Clip.mp4`, confirm FPS is detected as 30, press `.` and see the frame counter advance by one, press `C` then `8` and see a `smash` row appear, press `R` twice (a second apart in video time) then `N`, click a court point and see the marker, and press **Save labels**. Then validate the download:

```bash
cd worker && python -c "import json,sys; from evaluation.golden import validate_golden; print(validate_golden(json.load(open(sys.argv[1]))))" ~/Downloads/Hendry_Clip.json
```

Expected: `[]` (plus warnings in the page if some contacts are still `other`).

- [ ] **Step 5: Commit**

```bash
git add tools/labeller/labeller-core.js tools/labeller/index.html tools/labeller/labeller-core.test.ts vitest.config.ts docs/golden-set.md
git commit -m "feat(tools): add browser labeller for golden-set clips"
```

---

## Self-Review

**Spec coverage.** §3 diagnosis → fixed in Tasks 1–4 and 8 (camera model replaces planar-only mapping; near-corners-out-of-frame case handled by solving from any ≥4 floor keypoints). §4-L1 court model, intrinsics, pose, error in cm → Tasks 1, 2, 4 (per-frame handheld tracking is Phase 1b, its own plan, and reuses `Camera`/`solve_camera`). §5 tiers → Task 5 and surfaced in Task 8. §6 golden set and evaluation → Tasks 6–7. §7 Phase 0 and 1a gates → Task 7 produces the baseline report; Task 4 tests demonstrate the < 10 cm floor-error target on synthetic data (real-footage gate is measured with the golden set). Phases 1b–9 are intentionally out of scope for this plan.

**Placeholder scan.** No TBD/TODO; every code step shows full code. One intentional guarded loosening rule in Task 4 Step 4 (numeric thresholds only, never tier/outlier assertions).

**Type consistency.** `solve_camera` → `CameraSolution` fields (`tier`, `rms_px`, `floor_rms_cm`, `loo_floor_cm`, `redundancy`, `inliers`, `outliers`) are used identically in Tasks 7 and 8; `assess_geometry` → `GeometryQuality` fields (`tier`, `elevation_deg`, `visible_fraction`, `distance_m`, `reasons`) match; the adapter's output keys match the zod schema keys exactly (`cameraTier rmsPx focalPx k1 looFloorCm geometryTier elevationDeg visibleFraction reasons`).

**Review Focus coverage.** (1) Tasks 4 and 8 (`test_fewer_than_four_keypoints_returns_none`, `test_three_corners_are_not_solvable`); (2) Task 4 (`collinear`, `off_floor`, `identical_pixels`); (3) Task 4 (`non_finite_pixels`) and Task 8 (`bad_values_are_skipped`); (4) Task 4 scenes include `behind_portrait`; (5) Task 4 (`rejects_outlier_taps`, `gross_noise_is_never_validated`).

---

## Post-review changes (2026-10-03)

The whole-branch review found defects in this plan's own code. The repository, not the code blocks above, is the reference for later phases:

- `geometry/calibrate.py`: the pose sign is chosen from the observed points' depth (the court origin may be behind the camera), and cameras below the floor or with an inlier behind them are tier `unavailable` (mirrored labels fit a reflected camera almost exactly).
- `geometry/calibration_adapter.py`: no capture advice from a camera the solver rejected (one `UNSOLVED_REASON` instead), and no summary for doubles courts in v1.
- `evaluation/evaluate.py`: unmatched verified shots count as false claims; summary numbers pool counts over clips; each clip reports `workerCamera` and names its calibration `source`; calibration is solved from one frame.
- Golden keypoints and the labeller carry an optional `timeMs` (the frame a point was clicked on).
