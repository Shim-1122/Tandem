# Homunculus Shadowing

Camera-based full-body tracking built as the control input for Homunculus, a pneumatic humanoid arm, so it can shadow a person's motion: shoulder (3 DOF), elbow, and wrist (3 DOF) angles, 21 finger joints and 16 hand gestures, 10 DOF per leg and 3 in the core, the head's 6 degrees of freedom, face recognition and emotion with per-person calibration, from a single webcam.

> **Status: the Homunculus arm is not built yet.** It is a planned build. This repository is the vision side only: it tracks a person and produces joint angles and 0..1 joint commands ready for the arm, but nothing here has driven real hardware yet.

Pipeline: camera thread (newest frame only) -> one thread per model (MediaPipe hand, pose and face landmarks; OpenCV face detection, emotion and recognition) -> One Euro smoothing -> joint angles, gestures, head pose, emotions -> normalized 0..1 joint commands with limits and slew-rate control.

## What is here

| Module | Purpose |
| --- | --- |
| `vtrack/app.py` | The live application: runs the models in threads, combines their results, draws the screen, records CSV |
| `vtrack/pipeline.py` | Low-latency capture and model threads (newest frame only), CPU priority for hands, rate and latency meters |
| `vtrack/tracker.py` | MediaPipe Tasks wrappers (hand, pose, face mesh; VIDEO mode), model download, hand-to-arm matching by position |
| `vtrack/head.py` | Head pose with 6 DOF (position and yaw, pitch, roll) from 4 anchor points on the face |
| `vtrack/face.py` | Face detection, emotion (six cardinal emotions plus neutral, in percent) and face recognition |
| `vtrack/emotion_cal.py` | Per-person emotion calibration: guided capture, fitted correction, held-out accuracy |
| `vtrack/legs.py` | Legs (hip 4, knee 1, ankle 3, foot 2 DOF each) and core (3 DOF) from the pose landmarks |
| `vtrack/kinematics.py` | Seven joint angles per arm: shoulder 3 DOF, elbow 1 DOF, wrist 3 DOF, with validity flags |
| `vtrack/synthetic.py` | Forward kinematics (angles to landmarks and segment frames) for tests, the 3D panel, and a simulated arm |
| `vtrack/draw.py` | Landmark wireframes and the 3D panels (arm rebuilt from solved angles, or the hand in POV mode) |
| `vtrack/elbow.py` | Elbow prediction from shoulder, wrist and learned arm lengths when the elbow is out of view |
| `vtrack/fingers.py` | Finger kinematics: 21 finger and thumb joint angles from hand landmarks, and the reverse (forward kinematics) |
| `vtrack/handpose.py` | POV mode: hand position and orientation relative to the camera (PnP), zeroing |
| `vtrack/geometry.py` | Lower level angle math: finger flexion, torso frame |
| `vtrack/filters.py` | One Euro landmark and joint-angle smoothing, smoothing presets, jitter meter |
| `vtrack/gestures.py` | 16 rule-based hand gestures and a time-based filter that removes flicker |
| `vtrack/mapping.py` | Angle to 0..1 command mapping and a slew-rate limiter |
| `scripts/live.py` | Starts the live app from the command line (options listed with `--help`) |
| `scripts/eval_stability.py` | Jitter and lag of each smoothing preset on simulated noisy landmarks |

## Run

```
pip install -r requirements.txt
python -m pytest                                        # no camera needed
python scripts/live.py                                  # everything: both arms and hands, legs, core, head, face
python scripts/live.py --torso desk                     # seated at a desk, hips hidden
python scripts/live.py --side right                     # one arm only
python scripts/live.py --mode pov --hand right --fov 70 # head-mounted (POV) camera, hands only
python scripts/live.py --name shyam                     # then press e to enroll your face as shyam
```

Add `--record recordings/run1.csv` to any of these to save every update. Models download into `models/` on
first run: hand 8 MB, pose 9 MB (full), face mesh 4 MB, face detector 0.2 MB, emotion 16 MB, and the face
recognition model (38 MB) only when you first enroll your face. On slower computers add `--no-emotion` or
`--no-face` (see Latency).

The window has three parts:
- **Left: the video** (mirrored in body mode, like a mirror) with the landmark wireframe (cyan body, yellow
  hands), the head's four anchor points and its axes (red forward, green up, blue right), and a compact
  readout in the top left corner: one table with every joint angle (both arms and the core on the left, both
  legs on the right, L and R columns), gestures, and a status line with each model's rate and latency.
- **Top right: the face panel.** Your face, who it is (after you enroll), the six cardinal emotions with
  percentages, and the head's 6 DOF (position and yaw, pitch, roll, plus motion since you zeroed it with `h`).
- **Bottom right: the 3D panel.** In body mode it rebuilds your whole body from the solved joint angles through
  forward kinematics: head (turned relative to the torso, with its four anchor points), both arms with red/
  green/blue axes for the upper arm, forearm and hand frames, both hands with every finger, the core, the
  pelvis and both legs with their feet. Anything not in view is drawn grey in a neutral pose (for example the
  legs while you sit at a desk). It shows what the solver believes, so if it moves like you the kinematics
  are right, and if it disagrees with the video the angles are wrong. In POV mode it shows the hand.

Both arms are tracked by default (`--side left` or `--side right` for one); each arm has its own solver and
hand, the 3D panel draws them cyan (right) and pink (left), and a recording gets `L_` and `R_` columns. Both
arms share one torso frame, so the shoulders must be in view. Legs, core and head are always tracked when in
view. Flags after a value in the readout: `!` outside the normal human range (a tracking problem), `h` held
(shoulder rotation with a nearly straight elbow, or hip rotation without the foot in view), `f` hip rotation
taken from the foot (knee straight).

The tracker only shows an arm's angles when its shoulder and wrist are actually visible (a hidden elbow is
predicted, see Elbow prediction); otherwise it says which joint is out of view, because MediaPipe still reports
a guessed position for hidden joints.
Angles far outside a normal human range are flagged as a tracking problem.

Keys (click the video window first):

| Key | Action |
| --- | --- |
| `q` | quit |
| `z` | body mode: calibrate the torso from the neutral arm pose (see Desk below) and, when you stand upright in full view, the legs and feet (see Legs); POV mode: zero the hand pose |
| `h` | zero the head pose (the face panel then shows motion since that moment) |
| `e` | enroll your face for recognition under `--name` (default `me`); press it 3 to 5 times with your head turned slightly differently each time |
| `k` | calibrate the emotion tracker to your face (about 30 seconds of prompts; `k` again cancels) |
| `f` | cycle the finger joint table (off, then each tracked hand) |
| `c` | print checks (axes, arm lengths, hand agreement) to the terminal |
| `w` | toggle the wireframe |

## Latency (smooth hand tracking)

Everything that could make the hands lag has been taken out of the hand's path:
- **Camera:** a dedicated thread reads the camera continuously and keeps only the newest frame, so a slow
  moment never leaves you looking at old frames. The camera is asked for 60 fps at 1280x720 (MJPG on Windows
  and Linux, which many webcams need for 60 fps) with a one-frame buffer. The status line shows what the
  camera really delivers (`cam N fps`).
- **Models in parallel:** hands, body, face mesh and emotion each run in their own thread on the newest
  frame. MediaPipe lets threads run at the same time on different processor cores, so the hand model never
  waits for the body, face or emotion models. (Measured: two models in parallel took 47 ms per frame against
  84 ms one after the other.)
- **Hands first:** the hand and body models always run at full speed. The face mesh and emotion models use
  only the processor cores left over, and on a computer without spare cores they still run at a guaranteed
  10 Hz (head) and 3 Hz (emotion). They are also capped at 30 Hz and 10 Hz, because faster would not help.
- **Smoothing that does not lag:** One Euro filters smooth hard when you hold still and almost not at all
  when you move fast. Gestures use a 0.2 second majority vote that works the same at any frame rate.

The status line reads like `cam 30 fps | hand 30 Hz 35 ms | pose 30 Hz 40 ms | face 30 Hz 25 ms | emotion 10 Hz 30 ms`:
rate per model and latency from the moment the camera captured the frame to the moment the result was
ready. On a 2-core test computer (much slower than a typical laptop, with a 30 fps camera) the hand ran at
about 25 Hz alone and 15 to 18 Hz with all four models. With 4 or more cores the face and emotion models get
cores of their own and no longer slow the hands.

If the hands are not smooth:
1. **Light.** In a dim room most webcams lengthen the exposure and drop to 15 fps; check `cam N fps`. Light
   your hands well.
2. **Fewer models:** `--no-emotion`, then `--no-face`, then `--pose-model lite`.
3. **Other programs:** close video calls and browser tabs that use the camera or the processor.
4. **Smoothing:** `--smoothing responsive` for the least lag.

## Modes

**Standing (`--torso auto`, default).** Stand 1.5 to 2.5 m from the camera, facing it, with the tracked arm
and torso in view, hips included. Torso "up" comes from shoulders and hips.

**Desk (`--torso desk`).** No hips needed. Shoulders and the tracked arm must be in view. Sit upright, let the
tracked upper arm hang straight down beside your torso with the elbow bent 90 degrees, forearm forward and
thumb up (both arms like that when both are tracked), and press `z`. That pose becomes zero, and the calibration removes the camera's tilt (a laptop
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

## Gestures

16 gestures, each hand separately, in both body and POV mode. The hand can face any direction (except
thumbs up and down, which need the thumb pointing up or down).

| Gesture | Name shown | How to make it |
| --- | --- | --- |
| Open palm | `open_palm` | all five fingers out |
| Fist | `fist` | all fingers and the thumb curled |
| Point | `point` | index finger only |
| Peace | `peace` | index and middle out (a V), thumb tucked |
| Thumbs up | `thumbs_up` | fist with the thumb out, pointing up |
| Thumbs down | `thumbs_down` | fist with the thumb out, pointing down |
| OK | `ok` | thumb and index tips touching, the other three fingers out |
| Pinch | `pinch` | thumb and index tips touching, the other fingers curled |
| L shape | `l_shape` | thumb and index out (an L, or a finger gun) |
| Three | `three` | index, middle and ring out, thumb tucked |
| Three with thumb | `three_thumb` | thumb, index and middle out |
| Four | `four` | four fingers out, thumb tucked across the palm |
| Rock | `rock` | index and pinky out, thumb tucked (horns) |
| I love you | `i_love_you` | thumb, index and pinky out |
| Call me | `call_me` | thumb and pinky out (shaka) |
| Pinky | `pinky_up` | pinky only |

Anything else shows `unknown`. A finger counts as out when its tip is farther from the wrist than its middle
joint and it bends less than 90 degrees in total; these thresholds were set on real hand photos (relaxed open
hands, fists, pointing, peace and thumbs up), and the test photos of thumbs up, peace, pointing and open hands
are all classified correctly, mirrored or not. A gesture shows once it has been held for about 0.1 to 0.2
seconds and stops showing about 0.1 seconds after the hand changes, so it does not flicker.

## Head (6 DOF, 4 anchor points)

The head pose comes from 4 points on the rigid part of the skull, which do not move when you talk, smile or
open your mouth: **forehead, nose tip, and the outer corners of both eyes** (shown as dots on the video).
MediaPipe's face mesh gives their 3D positions; the best rigid fit of a standard head model to those 4 points
gives the rotation, and the camera's field of view gives the position.

| Value | Meaning |
| --- | --- |
| `x`, `y`, `z` | head position in cm from the camera: right, down, away (the point inside the head behind the nose) |
| `yaw` | turning: + toward your left |
| `pitch` | nodding: + looking up |
| `roll` | tilting: + top of the head toward your right shoulder |

All three angles are zero when you face the camera squarely and upright. Press `h` to zero the head where it
is now; the face panel then shows `dx dy dz` and `dyaw dpitch droll` since that moment, which is what a robot
head would follow. In body mode the 3D panel also turns the head relative to the torso (the neck angles,
recorded as `neck_yaw`, `neck_pitch`, `neck_roll`).

Accuracy: on synthetic faces the solution is exact (200 random poses, error under 0.001 degrees); on a real
portrait photo it agrees with MediaPipe's own full-face fit within 2 degrees, and follows 15 and 30 degree
image rotations within 1 to 2 degrees. Distance assumes an average-size face and depends on `--fov`
(your camera's horizontal field of view), so relative motion after `h` is more accurate than absolute
distance. A fit of only the 4 flat image positions was tested and rejected: it read 27 degrees of pitch on a
person looking straight at the camera.

## Face recognition and emotion

**Emotion.** The same approach as the reference project
[atulapra/Emotion-detection](https://github.com/atulapra/Emotion-detection) (find the face, crop it, classify
the expression with a neural network), with a stronger model that needs only OpenCV: HSEmotion's
EfficientNet-B0 (`enet_b0_8_best_vgaf` from
[av-savchenko/face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition), Apache 2.0
code, trained on the AffectNet dataset), on the face box from OpenCV's YuNet face detector.

It replaced the first model (OpenCV Zoo's MobileFaceNet, trained on RAF-DB), which almost never showed fear.
Both were measured with this code on 1048 faces from the AffectNet validation set, about 150 per emotion:

| Emotion | First model | HSEmotion (now) |
| --- | --- | --- |
| Fear | 9% | 78% |
| Anger | 23% | 79% |
| Disgust | 31% | 61% |
| Sadness | 38% | 67% |
| Surprise | 54% | 70% |
| Happiness | 100% | 89% |
| Neutral | 45% | 39% |
| Average | 43% | 69% |

The first model called most fear faces surprise. AffectNet is HSEmotion's own benchmark (its validation faces
were not used for training), so part of the gap is home advantage, but a second random sample run through the
app's own code gave the same picture (fear 71% against 10%). HSEmotion's weak spot is neutral: a relaxed face
often reads as mild anger or surprise. Calibrating to your face (key `k`, below) corrects exactly that. The
first model is still available with `--emotion-model opencv`; calibrations are kept per model, so calibrate
again after switching.

The face panel shows the **six cardinal emotions with percentages: happiness, sadness, anger, fear, surprise
and disgust**, the strongest in green. The model also scores neutral, shown underneath; the six and neutral
add up to 100%, so a relaxed face reads mostly neutral instead of inflating whichever emotion is slightly
ahead. Percentages are smoothed over about a third of a second. Recorded as `emotion_<name>_pct`.

Keep in mind that the model reads facial expressions, not feelings: a polite smile reads as happiness, and
lighting, glasses or a turned head lower its accuracy.

**Recognition.** Press `e` to enroll your face (run with `--name shyam` to store it under your name; letters,
digits, `-` and `_`). Each press adds one sample; 3 to 5 presses with your head turned slightly differently
works best. From then on the panel shows the name and a match score (cosine similarity, OpenCV's SFace model,
99.4% on the LFW benchmark). A score of 0.363 or more counts as the same person, OpenCV's recommended
threshold; below that it shows `unknown face`. In tests the same person matched at 0.94 to 0.97 (mirrored,
darker, smaller), and 22 different people all scored 0.32 or less against them.

**Calibration for your face.** Expression models are less accurate for some faces than others: a resting
face can read as sad or angry because of its shape rather than its expression, and people show each emotion
with different strength. Press `k` and follow the prompts at the bottom of the video: for each of the seven
expressions (neutral, happy, sad, angry, fear, surprise, disgust) you get 1.5 seconds to make it and 2.5
seconds of recording. A small correction of the model's raw scores is then fitted to your frames (one overall
scale and one offset per emotion, 8 numbers) and saved in `faces/<name>.emotion.json`.

It is applied whenever the recognizer identifies you (or, if nobody is enrolled, for `--name`); the face
panel says `emotion calibrated for <name>`. The calibration is per person, not per group of people, because
what it corrects is how the model reads your face, and only your own frames can measure that.

When it finishes, the readout shows the accuracy on frames the fit did not use (fitted on the first half of
each expression's frames, scored on the second half), before and after. In simulation, a face that the model
misread as sad went from 27 to 40% to 83 to 87% held-out accuracy, and a face it already read well stayed
the same (91 to 93%). Posed expressions are easier than spontaneous ones, so treat your number as a best
case; recalibrate if the lighting or camera changes a lot.

Privacy: enrolled faces stay on your computer in `faces/<name>.npy`, 128 numbers per sample and no images,
and calibrations in `faces/<name>.emotion.json`. Delete those files to remove a person. `faces/` and `models/`
are excluded from git.

## Legs and core (10 DOF per leg, 3 in the core)

Legs and core are tracked whenever they are in view, in body mode: stand about 2.5 to 3.5 m from the camera
so you are in the frame from head to feet; they are in the right half of the readout. Each leg needs its hip, knee
and ankle in view (both hips for the pelvis); ankle and foot angles also need the heel and toe.

| Joint | Angle | Positive direction |
| --- | --- | --- |
| Hip | `hip_flexion` | thigh forward (extension negative) |
| Hip | `hip_abduction` | thigh outward (adduction negative) |
| Hip | `hip_rotation` | internal (external negative), about the thigh's long axis |
| Hip | `pelvic_hike` | this side of the pelvis raised (dropped negative); the two legs read opposite signs |
| Knee | `knee_flexion` | bent; 0 is straight |
| Ankle | `ankle_dorsiflexion` | toes up (pointing the foot is negative) |
| Ankle | `ankle_inversion` | sole turned inward (eversion negative) |
| Ankle | `ankle_abduction` | toes turned outward (inward negative) |
| Foot | `foot_pitch` | foot relative to the floor: toes up |
| Foot | `foot_roll` | foot relative to the floor: outer edge down, sole facing inward |
| Core | `core_flexion` | bending forward (backward negative) |
| Core | `core_lateral_bend` | bending to your right |
| Core | `core_rotation` | chest turned to your left relative to the hips (same direction as head yaw) |

Zero is standing upright, legs straight, feet flat and pointing forward. Both legs use the same signs. Hip,
ankle and core angles are Z-X-Y sequences (lateral axis, then forward, then the long axis), the order the
ISB recommends; the core is the chest (shoulders) relative to the pelvis (hips). Recorded as `L_` and `R_`
columns plus `core_*`.

**Calibrate standing.** Stand upright, legs straight, feet flat on the floor and pointing forward, whole body
in view, and press `z`. This takes "up" from your legs (otherwise the camera's vertical is used, so a tilted
camera reads as hip and core flexion) and removes the offset between MediaPipe's heel, toe and ankle points
and your foot's real axes. The readout's last leg line says which "up" is in use.

What the camera cannot tell you, and what the tracker does about it:
- **Pelvic tilt (forward and back).** MediaPipe gives only the two hip joints for the pelvis, so its forward
  tilt cannot be seen. The pelvis is taken as level front to back: hip flexion is the thigh from vertical, and
  core flexion includes any pelvic tilt. Pelvic hike (side to side) and turning are measured.
- **Hip rotation with a straight knee.** The knee's hinge axis is only defined when the knee bends. Below 20
  degrees the hip rotation is taken from the direction your foot points, assuming the ankle itself is not
  turned (`ankle_abduction` then reads 0; the readout marks hip rotation `f`). Without the feet in view it is
  held (`h`).
- **Feet are rough.** Each foot has just three points (ankle, heel, toe tip), and their depth comes from a
  single camera. On a real yoga photo (warrior II), the bent front knee read 82 degrees (it looks about 90
  in the photo), but the straight back knee read 26 degrees instead of nearly 0, and its foot roll was clearly
  wrong.
  Mirroring the same photo read the straight knee at 10 degrees. Expect the knees and hips to be usable and
  the ankle and foot angles to be approximate, best when the legs move across the camera's view.

## Elbow prediction

When the elbow leaves the frame but the shoulder and wrist are still visible, the elbow is predicted.
Upper arm and forearm lengths are learned while all three joints are clearly visible (until then they are
estimated from shoulder width using average adult proportions), so the elbow must lie on a known circle
around the shoulder-wrist line. Where on that circle comes from MediaPipe's own guess for the hidden elbow,
continuity with the previous frame, and, only as a tie-breaker, the elbow hanging low. The readout says
"elbow predicted (out of view)" and the 3D panel draws that arm in a darker shade.

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

Press `f` to show the finger joint table (in POV mode it starts on). Angles are measured in the palm's own
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
4. **Head.** Face the camera squarely and check that yaw, pitch and roll read near zero; then turn, nod and tilt
   and confirm the signs in the Head table. If the distance `z` is clearly wrong, set `--fov` for your camera.
5. **Latency.** Watch the status line while you move: the hand should run near the camera rate.
6. **Legs.** Stand in full view and press `z`: the leg and core angles should read near zero. Bend one knee, lift a
   thigh forward and out, and turn a foot out, and confirm the signs in the Legs table.
7. **Emotion.** Press `k` and follow the prompts; the held-out accuracy afterwards tells you how well the
   tracker reads your face in your lighting.

Expect the largest errors in shoulder rotation and forearm rotation: they come from small differences between
noisy landmarks, and a few centimeters of elbow or wrist error is several degrees over a 25 to 30 cm segment.

## Other conventions and limits

- Angles use MediaPipe world landmarks (meters), not image coordinates.
- `--side` is the person's own arm. In body mode a hand is matched to that arm by image position, not by MediaPipe's left/right label.
- One RGB camera gives estimated depth, so angles are most reliable when the arm moves across the camera view.
- The command mapping stops at 0..1 per joint. Mapping to air muscle pressure needs a calibration curve per muscle.
- The shoulder angle order (Z-X-Y) should match the physical Homunculus shoulder once the arm is built and its axes are decided.
- Recording columns: per arm the gesture, the seven angles, flags and 21 finger angles (`L_` and `R_` prefixes with
  `--side both`); per leg the ten leg angles and the hip rotation source (`L_` and `R_`), the three core angles and
  `leg_up_source`; then `torso_source`, head position and angles, neck angles, the seven emotion percentages,
  `identity`, `match` and `emotion_calibrated_for`. A row is written whenever any model delivers a new result.
- Tested: 171 automated tests (Python 3.9 and 3.13), including kinematics round trips for both arms and both legs
  and the core (exact for random poses, headings and pelvic hikes), desk and standing calibration with a tilted
  camera, PnP hand position recovery, all gestures in random orientations for both hands, head pose round trips,
  emotion alignment against an independent least-squares solution, the emotion calibration fit (gradient checked,
  recovers known corrections), the threaded pipeline, and the whole app with simulated models, including a full
  emotion calibration. The app was also run headless with the real models on photos (body, both arms, hidden
  elbows, a full-body yoga pose, POV with thumbs up, peace, pointing and open hands). Not tested here: accuracy
  with a real camera on your setup. Use the checks above.
