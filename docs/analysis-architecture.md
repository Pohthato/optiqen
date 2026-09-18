# Netoval Analysis Architecture

Netoval treats the court floor as a calibrated plane and keeps two distinct coordinate systems. A projective transform maps pixel locations to 2D court meters for feet, recovery positions, coverage, and shot-origin summaries. A separate camera-pose estimate is required for true 3D reasoning; planar homography alone must not be presented as a complete reconstruction of shuttle height, racket depth, or joint depth.

| Product surface | Method | Validation rule |
| --- | --- | --- |
| Court-space movement | Robust homography from visible court-line correspondences | Display reprojection error and a calibration-confidence state. |
| Partial court view | User selects three or more semantically named landmarks; infer a missing rectangular landmark only when line-direction consistency is sufficient | Require user confirmation of inferred geometry and permit adjustments. |
| Camera viewpoint | Estimate intrinsics and extrinsics against a regulation court model when enough line/keypoint evidence exists | Label the result as calibrated, approximate, or unavailable rather than overstating certainty. |
| Skeleton biomechanics | Temporal pose model with ankle/hip/shoulder keypoints, uncertainty scores, and smoothing | Exclude low-confidence frames from aggregate coaching conclusions. |
| Shuttle and racket movement | Tiny-object detector plus temporal tracker, linked to pose and court coordinates | Render discontinuities and confidence instead of inventing invisible trajectories. |
| Contact and rally events | Speed collapse/reversal of the shuttle plus racket/wrist proximity; rallies split on gaps | Coach with timestamped events only; sub-60 fps clips are labelled as low-confidence capture. |
| Stroke and side classification | Verified shot only when contact plus post-impact track clears a gate; `landingSide` from the observed cloud beyond the net centre-line | Never interpolate an unobserved shuttle bounce; present labels as classified strokes, not ground truth. |
| Racket swing and footwork | `racket_swing` events and image-relative speed/amplitude metrics from confident pose/racket tracks | Surface as relative measures with a clear unit (`frame diagonals per second`, `normalized frame diagonal`). |

The processing system should use an asynchronous GPU-capable worker service. The web application owns uploads, analysis records, calibration state, progress, results, and coaching interaction. The worker service owns heavyweight frame decoding, pose inference, shuttle/racket tracking, court-line detection, bundle adjustment, and annotated-video rendering. This boundary prevents the product interface from claiming real-time or 3D precision that its available hardware cannot support.

## Calibration interaction

The first frame becomes a calibration canvas. The player must submit three known court landmarks, while a fourth visible point is optional. Each corner carries a semantic label rather than relying on click order alone. With three points, the system derives a provisional missing-corner hint from court topology and delegates line-direction, vanishing-point, homography, and reprojection validation to the worker. The interface lets the player continue with provisional geometry, upgrades to validated geometry when a fourth point or sufficient court-line evidence is available, and falls back to image-space-only coaching when exact court mapping is not reliable.

## Sources

The two implementation references are retained here for the build team. Recent sports-field research distinguishes planar homography from camera calibration using a 3D field model, while practical sports tracking examples demonstrate that stable source-to-court correspondences enable pixel-to-field movement transformation.

1. [Gutiérrez-Pérez & Agudo, Sports Field Registration by Leveraging Geometric Properties](https://arxiv.org/html/2404.08401v1)
2. [Roboflow, Camera Calibration in Sports with Keypoints](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)
