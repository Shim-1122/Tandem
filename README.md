# Homunculus Shadowing

Camera-based arm and hand joint tracking so the Homunculus pneumatic arm can shadow a person's motion: shoulder (3 DOF), elbow, and wrist (3 DOF) angles plus hand gestures from a single webcam.

Pipeline: camera frame -> MediaPipe hand and pose landmarks (3D, meters) -> One Euro smoothing -> joint angles and gestures -> normalized 0..1 joint commands with limits and slew-rate control.

## What is here

| Module | Purpose |
| --- | --- |
| `vtrack/tracker.py` | MediaPipe Tasks wrapper (hand + pose, VIDEO mode), model download, hand-to-arm matching by position |
| `vtrack/kinematics.py` | Seven joint angles per arm: shoulder 3 DOF, elbow 1 DOF, wrist 3 DOF, with validity flags |
| `vtrack/synthetic.py` | Forward kinematics (angles to landmarks) for tests and for driving a simulated arm |
| `vtrack/geometry.py` | Lower level angle math: finger flexion, torso frame |
| `vtrack/filters.py` | One Euro filter and a landmark smoother that resets after dropouts |
| `vtrack/gestures.py` | Rule-based gestures (open palm, fist, point, peace, thumbs up) and a debouncer |
| `vtrack/mapping.py` | Angle to 0..1 command mapping and a slew-rate limiter |
| `scripts/live.py` | Webcam demo showing all seven angles, with optional CSV recording |

## Run

```
pip install -r requirements.txt
python -m pytest                      # no camera needed
python scripts/live.py --side right --record recordings/run1.csv
```

Models (about 8 MB hand, 6 MB pose lite) download into `models/` on first run. Stand 1.5 to 2.5 m
from the camera, facing it, with the tracked arm and torso in view. Press `c` in the window to print
the hand/pose frame agreement check (see Verify first).

## Joint angles

Zero is the neutral pose: arm hanging, elbow bent 90 degrees, forearm pointing forward, thumb up.
Left and right arms use the same signs (the left arm is mirrored internally). Angles are in degrees.

| Joint | Angle | Plane | Positive direction |
| --- | --- | --- | --- |
| Shoulder | `shoulder_flexion` | sagittal | arm raised forward (extension negative); 180 is overhead |
| Shoulder | `shoulder_abduction` | coronal | arm raised outward (adduction negative), range -90 to 90 |
| Shoulder | `shoulder_rotation` | transverse | internal rotation (forearm swings toward the midline) |
| Elbow | `elbow_flexion` | sagittal | bending; 0 is straight |
| Wrist | `wrist_flexion` | | fingers toward the palm side (extension negative) |
| Wrist | `wrist_deviation` | | radial, toward the thumb side (ulnar negative) |
| Forearm | `forearm_rotation` | | pronation, palm turning down (supination negative) |

The shoulder is three rotations in order: sagittal about the lateral axis, then coronal about the
forward axis, then transverse about the arm's long axis (intrinsic Z-X-Y, pitch, roll, yaw). The
wrist is a Y-X-Z sequence of the hand relative to the forearm. Pronation and supination are
reported with the wrist because the hand's orientation is what shows them.

## What the camera cannot tell you (flags in the output)

- **Elbow nearly straight (under 20 degrees):** shoulder rotation and forearm rotation cannot be separated, because the
  elbow bend plane is undefined. Shoulder rotation is held at its last good value, `shoulder_rotation_valid` is
  false, and the hand's extra twist shows up as forearm rotation. Their sum stays correct.
- **Arm held horizontally out to the side (abduction within 5 degrees of 90):** the sagittal and transverse axes line up
  (gimbal lock). Rotation is held and flexion takes the difference, so the rotation matrix is still right but the
  two angles are not individually meaningful there.
- **No hand detected:** wrist angles use the pose's own pinky and index points if available; otherwise `wrist_valid` is false.

## Verify first (not yet done on a real camera)

1. **Axes.** Face the camera, hold the hand still in view, and press `c`. The torso forward vector should have a clearly
   negative z (toward the camera); if not, the world axes are mirrored and every sagittal sign is flipped.
   Then check frame agreement: the solver assumes hand and pose world landmarks share axis directions. If `long_axis_deg` or `palm_normal_deg` is more than about 15, hand orientation cannot be trusted with
   the hand model; use `--pose-only-wrist`.
2. **Sign check.** Do each motion in the table and confirm the displayed sign.
3. **Error against ground truth.** Compare against a goniometer or a leader arm before any control.

Expect the largest errors in shoulder rotation and forearm rotation: they come from small differences between
noisy landmarks, and a few centimeters of elbow or wrist error is several degrees over a 25 to 30 cm segment.

## Other conventions and limits

- Angles use MediaPipe world landmarks (meters), not image coordinates.
- `--side` is the person's own arm. A hand is matched to that arm by image position, not by MediaPipe's left/right label.
- One RGB camera gives estimated depth, so angles are most reliable when the arm moves across the camera view.
- The command mapping stops at 0..1 per joint. Mapping to air muscle pressure needs a calibration curve per muscle.
- The shoulder angle order (Z-X-Y) should match the physical Homunculus shoulder once its axes are decided.
- Tested: 50 unit tests including round trips through forward kinematics for both arms and random body orientations.
  Not tested: a real camera, real hand and pose landmark accuracy.
