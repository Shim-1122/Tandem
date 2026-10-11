# Homunculus Shadowing

Camera-based arm and hand joint tracking so the Homunculus pneumatic arm can shadow a person's motion: shoulder (3 DOF), elbow, and wrist (3 DOF) angles plus hand gestures from a single webcam.

Pipeline: camera frame -> MediaPipe hand and pose landmarks (3D, meters) -> One Euro smoothing -> joint angles and gestures -> normalized 0..1 joint commands with limits and slew-rate control.

## What is here

| Module | Purpose |
| --- | --- |
| `vtrack/tracker.py` | MediaPipe Tasks wrapper (hand + pose, VIDEO mode), model download, hand-to-arm matching by position |
| `vtrack/kinematics.py` | Seven joint angles per arm: shoulder 3 DOF, elbow 1 DOF, wrist 3 DOF, with validity flags |
| `vtrack/synthetic.py` | Forward kinematics (angles to landmarks and segment frames) for tests, the 3D panel, and a simulated arm |
| `vtrack/draw.py` | Landmark wireframes and the 3D panels (arm rebuilt from solved angles, or the hand in POV mode) |
| `vtrack/elbow.py` | Elbow prediction from shoulder, wrist and learned arm lengths when the elbow is out of view |
| `vtrack/fingers.py` | Finger kinematics: 21 finger and thumb joint angles from hand landmarks, and the reverse (forward kinematics) |
| `vtrack/handpose.py` | POV mode: hand position and orientation relative to the camera (PnP), zeroing |
| `vtrack/geometry.py` | Lower level angle math: finger flexion, torso frame |
| `vtrack/filters.py` | One Euro landmark and joint-angle smoothing, smoothing presets, jitter meter |
| `vtrack/gestures.py` | Rule-based gestures (open palm, fist, point, peace, thumbs up) and a debouncer |
| `vtrack/mapping.py` | Angle to 0..1 command mapping and a slew-rate limiter |
| `scripts/live.py` | Live demo: wireframe on the video, 3D kinematics panel, angles, CSV recording |
| `scripts/eval_stability.py` | Jitter and lag of each smoothing preset on simulated noisy landmarks |

## Run

```
pip install -r requirements.txt
python -m pytest                                        # no camera needed
python scripts/live.py --side right                     # standing, hips in view
python scripts/live.py --side right --torso desk        # seated at a desk, hips hidden
python scripts/live.py --side both --torso desk         # both arms at once
python scripts/live.py --mode pov --hand right --fov 70 # head-mounted (POV) camera, hands only
```

Add `--record recordings/run1.csv` to any of these to save every frame. Models (about 8 MB hand, 6 MB
pose lite) download into `models/` on first run.

The window shows the video with the landmark wireframe (cyan body, yellow hands) on the left and a 3D
panel on the right. In body mode the panel rebuilds your arm from the solved joint angles through forward
kinematics, with red/green/blue axes for the upper arm, forearm and hand frames. It shows what the solver
believes, so if it moves like your arm the kinematics are right, and if it disagrees with the video the
angles are wrong.

With `--side both`, each arm has its own solver and hand, the readout shows LEFT and RIGHT columns, the
3D panel draws both arms (cyan right, pink left), and a recording gets `L_` and `R_` columns. Both arms
share one torso frame, so they must both be in view with the shoulders.

The tracker only shows an arm's angles when its shoulder and wrist are actually visible (a hidden elbow is
predicted, see Elbow prediction); otherwise it says which joint is out of view, because MediaPipe still reports
a guessed position for hidden joints.
Angles far outside a normal human range are flagged as a tracking problem.

Keys (click the video window first): `q` quit, `w` toggle wireframe, `c` print checks (axes, arm lengths,
hand agreement) to the terminal, `f` cycle the finger table, `z` calibrate (see below).

## Modes

**Standing (`--torso auto`, default).** Stand 1.5 to 2.5 m from the camera, facing it, with the tracked arm
and torso in view, hips included. Torso "up" comes from shoulders and hips.

**Desk (`--torso desk`).** No hips needed. Shoulders and the tracked arm must be in view. Sit upright, let the
tracked upper arm hang straight down beside your torso with the elbow bent 90 degrees, forearm forward and
thumb up (with `--side both`, both arms like that), and press `z`. That pose becomes zero, and the calibration removes the camera's tilt (a laptop
screen leaned back 25 degrees would otherwise add 25 degrees of shoulder flexion). Press `z` again whenever
you move the camera or your chair. If the upper arm is not hanging straight down when you press `z`, every
flexion reading is offset by that amount. Before calibrating, the camera's vertical is used, so keep the
camera level.

**POV (`--mode pov`).** For a camera on your head or goggles. Your shoulders are not visible from there, so
the arm joint angles cannot be measured; instead it tracks:
- hand position relative to the camera in cm (right, down, away), from the hand landmarks and the camera's
  field of view (`--fov`; laptop webcams are about 65 to 78, wide action cameras 110 or more),
- hand motion and rotation relative to a zero pose you set with `z` (dx, dy, dz; twist, pitch, yaw),
- flexion at every finger joint (MCP, PIP, DIP) and the gesture.

That relative hand motion is what a robot's end effector would follow during teleoperation. Two limits:
depth assumes an average-size hand (a bigger hand reads farther away), and because the camera moves with
your head, head motion looks like hand motion unless the goggles also report head orientation (an IMU).
MediaPipe's left/right label assumes a mirrored image; if POV mode follows the wrong hand, add `--swap-hands`.

## Elbow prediction

When the elbow leaves the frame but the shoulder and wrist are still visible, the elbow is predicted.
Upper arm and forearm lengths are learned while all three joints are clearly visible (until then they are
estimated from shoulder width using average adult proportions), so the elbow must lie on a known circle
around the shoulder-wrist line. Where on that circle comes from MediaPipe's own guess for the hidden elbow,
continuity with the previous frame, and, only as a tie-breaker, the elbow hanging low. The readout says
"elbow out of view: predicted" and the 3D panel draws that arm in a darker shade.

Let the tracker see your whole arm for a second or two first so it learns your arm lengths (press `c` to see
them). In simulation, with MediaPipe's hidden-elbow guess off by 4 cm per axis, the predicted elbow was off
by 2.6 cm (median) versus 6.2 cm for the raw guess; with the elbow hidden for a second while bending, it
stayed within 0.4 cm. The weakest case is rotating the arm about the shoulder-wrist line while the elbow is
hidden: that motion does not change the shoulder or wrist, so only MediaPipe's guess can show it.

## Stability

Three things reduce arm jitter:
- `--pose-model full` (default) or `heavy`: more accurate pose landmarks than `lite`, at a lower frame rate.
- Landmark smoothing that is strongest along depth, the noisiest coordinate from a single camera.
- Smoothing of the joint angles themselves (One Euro filters, which smooth hard when still and lighten up
  when you move).

Pick the trade-off with `--smoothing`. In simulation (`python scripts/eval_stability.py`; pose landmark noise
0.6 cm sideways and 2 cm in depth, an assumption), mean over the seven angles:

| Setting | Jitter holding still (deg) | Error while moving (deg) |
| --- | --- | --- |
| no smoothing | 4.04 | 4.60 |
| previous version | 1.51 | 3.64 |
| `responsive` | 1.47 | 2.86 |
| `balanced` (default) | 1.18 | 3.20 |
| `stable` | 0.91 | 5.34 |

The `+-` number next to each angle is its standard deviation over the last second: hold still to read the
real jitter on your setup. Arm angles will stay noisier than finger angles, because they depend on depth
differences across the whole body while finger angles come from the hand model's close-up landmarks.

## Finger kinematics

Press `f` to show the finger joint table (in POV mode it is always on). Angles are measured in the palm's own
frame, so they do not change when the whole hand moves, and a left hand is mirrored so the signs match.

| Digit | Angles (deg) |
| --- | --- |
| Index, middle, ring, pinky | `mcp_flex` knuckle flexion (toward the palm +), `mcp_abd` knuckle side bend (toward the thumb +), `pip`, `dip` |
| Thumb | `cmc_palmar` (out of the palm plane), `cmc_radial` (away from the index), `cmc_rotation` (opposition +; zero = bending across the palm toward the pinky), `mcp`, `ip` |

The 3D panels draw the hand rebuilt from these solved angles (in POV mode over the raw landmarks in grey), so
any mismatch between the model and the measurement is visible. The thumb's rotation is only measurable while
the thumb is bent; while it is straight the last value is held. Recordings get 21 finger columns per hand.

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
   Then hold the wrist straight (hand in line with the forearm) and check `hand_vs_forearm_deg`: under about 20 means the
   hand model and pose model agree on axes. Much larger means the wrist angles from the hand model cannot be trusted;
   use `--pose-only-wrist`. `long_axis_deg` and `palm_normal_deg` compare against the pose model's rough finger points
   and are expected to be larger.
2. **Sign check.** Do each motion in the table and confirm the displayed sign.
3. **Error against ground truth.** Compare against a goniometer or a leader arm before any control.

Expect the largest errors in shoulder rotation and forearm rotation: they come from small differences between
noisy landmarks, and a few centimeters of elbow or wrist error is several degrees over a 25 to 30 cm segment.

## Other conventions and limits

- Angles use MediaPipe world landmarks (meters), not image coordinates.
- `--side` is the person's own arm. In body mode a hand is matched to that arm by image position, not by MediaPipe's left/right label.
- One RGB camera gives estimated depth, so angles are most reliable when the arm moves across the camera view.
- The command mapping stops at 0..1 per joint. Mapping to air muscle pressure needs a calibration curve per muscle.
- The shoulder angle order (Z-X-Y) should match the physical Homunculus shoulder once its axes are decided.
- Tested: 85 unit tests (Python 3.9 and 3.13), including kinematics round trips for both arms, desk calibration
  with a tilted camera and missing hips, and PnP hand position recovery; the live loop was run headless in all
  three modes. Not tested here: real camera accuracy. Use the checks above on your own setup.
