# Tandem

**Tandem** is single-webcam motion capture built as the control input for **Homunculus**, a planned pneumatic
humanoid arm, so the arm can shadow a person. It tracks both arms, every finger, hand gestures, both legs and the
core, the head, where the eyes look, the face (who it is and the expression), and which objects each hand is
holding, with the grip force the arm should use for each one.

> **Status: the Homunculus arm is not built yet.** It is a planned build. This repository is the vision side
> only: it tracks a person and turns that into joint angles, gestures, gaze targets and grip forces ready for
> the arm, but nothing here has driven real hardware yet.

**85 tracked degrees of freedom from one camera:** 7 per arm (14), 21 per hand (42), 10 per leg (20), 3 in the
core and 6 for the head, plus 2 eye angles, 16 hand gestures, 7 emotion scores, and grasps of 16 everyday
objects with a grip force for each.

Pipeline: camera thread (newest frame only) -> one thread per model (MediaPipe hand, pose and face landmarks;
OpenCV face detection, emotion and recognition; YOLO-World object detection) -> One Euro smoothing -> joint
angles, gestures, head pose, gaze and target, emotions, held objects and grip forces -> on screen and in a CSV
recording, with a module that maps angles to 0..1 actuator commands with limits and slew-rate control.

## Contents

- [What it tracks](#what-it-tracks)
- [Quick start (step by step)](#quick-start-step-by-step)
- [The window](#the-window), [Keys](#keys), [Command-line options](#command-line-options)
- [Modes: standing, desk and head camera (POV)](#modes-standing-desk-and-head-camera-pov)
- Body: [Arms](#arms-joint-angles), [Elbow prediction](#elbow-prediction), [Fingers](#fingers),
  [Gestures](#gestures), [Legs and core](#legs-and-core-10-dof-per-leg-3-in-the-core)
- Head and face: [Head](#head-6-dof-4-anchor-points), [Eyes](#eyes-gaze-and-target-locking),
  [Face recognition and emotion](#face-recognition-and-emotion)
- [Grasp tracking: objects, what each hand holds, and the grip force](#grasp-tracking-objects-what-each-hand-
  holds-and-the-grip-force)
- [Speed and latency](#speed-and-latency), [Stability and smoothing](#stability-and-smoothing)
- [What the camera cannot tell you](#what-the-camera-cannot-tell-you)
- [Recording (CSV columns)](#recording-csv-columns)
- [For the arm: what is ready and what is not](#for-the-arm-what-is-ready-and-what-is-not)
- [Verify on your camera first](#verify-on-your-camera-first)
- [Project layout](#project-layout), [Models](#models), [Tests](#tests), [Troubleshooting](#troubleshooting)

## What it tracks

| Part | What you get | Mode |
| --- | --- | --- |
| Arms | per arm: shoulder flexion, abduction and rotation, elbow flexion, forearm rotation, wrist flexion and deviation; a hidden elbow is predicted | body |
| Hands | per hand: 21 finger and thumb joint angles, and 16 gestures | body and POV |
| Hand in space | position in cm and rotation of the hand relative to a head-mounted camera | POV |
| Legs | per leg: hip flexion, abduction, rotation and pelvic hike, knee flexion, ankle dorsiflexion, inversion and abduction, foot pitch and roll to the floor | body |
| Core | flexion, side bend and rotation of the chest relative to the hips | body |
| Head | position (x, y, z) and yaw, pitch, roll from 4 anchor points; neck angles relative to the torso | body |
| Eyes | eye yaw and pitch; after a 9-dot calibration, the point you look at on the video; lock a target by key, pinch or dwell | body |
| Face | who it is (after you enroll) and the six cardinal emotions plus neutral in percent, calibratable to your face | body |
| Grasps | 16 objects found and tracked; which hand holds what; the hand's shape; grip type, force in newtons, 0..1 command and hand opening for each object | body and POV |

Body mode needs a camera that sees you; POV mode is for a camera on your head or goggles, looking at your hands.

## Quick start (step by step)

You need a computer with a webcam (Windows, Mac or Linux), about 1 GB of free disk space, and an internet
connection the first time.

1. **Install Python.** Download it from [python.org](https://www.python.org/downloads/); 3.11 or 3.12 is the
   safest choice (the tests also pass on 3.9 and 3.13). On Windows, tick **"Add python.exe to PATH"** on the
   first screen of the installer.
2. **Get the code.** On GitHub, signed in, open the repository and click **Code > Download ZIP**, then unzip it
   (or, with Git installed: `git clone https://github.com/Shim-1122/Tandem.git`).
3. **Open a terminal in that folder.**
   - Windows: open the folder in File Explorer, click the address bar, type `powershell` and press Enter.
   - Mac: open Terminal, type `cd ` (with a space), drag the folder onto the window, and press Enter.
4. **Make a private Python environment and install the packages** (a few minutes):

   | Windows (PowerShell) | Mac or Linux |
   | --- | --- |
   | `py -3.12 -m venv .venv` (use the version you installed) | `python3 -m venv .venv` |
   | `.venv\Scripts\python -m pip install -r requirements.txt` | `.venv/bin/python -m pip install -r requirements.txt` |

5. **Check it** (no camera needed, about a minute): `.venv\Scripts\python -m pytest` on Windows,
   `.venv/bin/python -m pytest` on Mac or Linux. It should end with `215 passed`.
6. **Run it:** `.venv\Scripts\python scripts\live.py` on Windows, `.venv/bin/python scripts/live.py` on Mac or
   Linux. The first run downloads about 40 MB of models into `models/` and unpacks the object model (about a
   second). On a Mac, the first run asks for camera access: allow it for Terminal (System Settings > Privacy &
   Security > Camera), then run the command again.
7. **Use it:** click the video window so it receives keys (see [Keys](#keys)); `q` quits. Stand back so the
   camera sees you, or add `--torso desk` when sitting at a desk.
8. **Next time:** open a terminal in the folder again and repeat step 6.

More ways to run it (use the same Python as in step 6, and add any options to the end; all options are in
[Command-line options](#command-line-options)):

```
python scripts/live.py                                  # everything: arms, hands, legs, core, head, face, objects
python scripts/live.py --torso desk                     # seated at a desk, hips hidden
python scripts/live.py --side right                     # one arm only
python scripts/live.py --mode pov --hand right --fov 70 # head-mounted (POV) camera, hands and grasps only
python scripts/live.py --name shyam                     # then press e to enroll your face as shyam
python scripts/live.py --gaze-dwell 1.0                 # also lock a target by looking at it for 1 second
python scripts/live.py --object-model small             # faster object detection for slower computers
python scripts/live.py --record recordings/run1.csv     # save every update to a CSV file
```

## The window

The window has three parts:
- **Left: the video** (mirrored in body mode, like a mirror) with the landmark wireframe (cyan body, yellow
  hands), the head's four anchor points and its axes (red forward, green up, blue right), a box and label on
  each tracked object (a held object in its hand's color with its grip force), the gaze cursor and locked
  target once calibrated, and a compact readout in the top left corner. The readout has one table with every
  joint angle (both arms and the core on the left, both legs on the right, L and R columns), then lines for
  what is out of view, where "up" comes from, gestures, each hand's grasp, the objects in view, the eyes, the
  latest message, a status line with each model's rate and latency, and the keys. Press `f` for the finger
  joint table underneath.
- **Top right: the face panel.** Your face, who it is (after you enroll), the six cardinal emotions with
  percentages, and the head's 6 DOF (position and yaw, pitch, roll, plus motion since you zeroed it with `h`).
- **Bottom right: the 3D panel.** In body mode it rebuilds your whole body from the solved joint angles through
  forward kinematics: head (turned relative to the torso, with its four anchor points), both arms with red/
  green/blue axes for the upper arm, forearm and hand frames, both hands with every finger, the core, the
  pelvis and both legs with their feet. Anything not in view is drawn grey in a neutral pose (for example the
  legs while you sit at a desk). It shows what the solver believes, so if it moves like you the kinematics
  are right, and if it disagrees with the video the angles are wrong. A predicted elbow's arm is drawn darker,
  and where the eyes look is a cyan line from the head. In POV mode it shows the hand.

Legs, core and head are always tracked when in view. Flags after a value in the readout: `!` outside the normal
human range (a tracking problem), `h` held (shoulder rotation with a nearly straight elbow, or hip rotation
without the foot in view), `f` hip rotation taken from the foot (knee straight). The tracker only shows an arm's
angles when its shoulder and wrist are actually visible (a hidden elbow is predicted, see [Elbow
prediction](#elbow-prediction)); otherwise it says which joint is out of view, because MediaPipe still reports a
guessed position for hidden joints.

## Keys

Click the video window first so it receives the keys.

| Key | Action |
| --- | --- |
| `q` | quit |
| `z` | body mode: calibrate the torso from the neutral arm pose (see Desk in Modes) and, when you stand upright in full view, the legs and feet (see Legs); POV mode: zero the hand pose |
| `h` | zero the head pose (the face panel then shows motion since that moment) |
| `e` | enroll your face for recognition under `--name` (default `me`); press it 3 to 5 times with your head turned slightly differently each time |
| `k` | calibrate the emotion tracker to your face (about 30 seconds of prompts; `k` again cancels) |
| `g` | calibrate eye tracking: look at 9 dots in turn (about 20 seconds; `g` again cancels) |
| `t` | lock onto the spot you are looking at, or release it (a pinch with either hand does the same) |
| `f` | cycle the finger joint table (off, then each tracked hand) |
| `c` | print checks (axes, arm lengths, hand agreement) to the terminal |
| `w` | toggle the wireframe |
| `o` | show or hide the object boxes |

## Command-line options

| Option | Default | What it does |
| --- | --- | --- |
| `--mode body` / `pov` | `body` | body: a camera that sees you; pov: a camera on your head, hands only |
| `--camera N` | `0` | which camera (0 is the built-in webcam; try 1 or 2 for a USB camera) |
| `--width`, `--height`, `--fps` | `1280`, `720`, `60` | requested camera size and frame rate (the camera may give less; the status line shows what it delivers) |
| `--side left` / `right` / `both` | `both` | body mode: which arms to track (legs, core and head are always tracked) |
| `--torso auto` / `hips` / `desk` | `auto` | where the torso's "up" comes from: auto = hips if visible, else calibration (`z`), else camera vertical; desk = calibration, else camera vertical |
| `--pose-model lite` / `full` / `heavy` | `full` | body model size: heavier is more accurate and slower |
| `--smoothing responsive` / `balanced` / `stable` | `balanced` | least lag, the default trade-off, or least jitter |
| `--pose-only-wrist` | off | wrist angles from the body model only (if the hand model disagrees on axes, see Verify) |
| `--hand left` / `right` | `right` | POV mode: which hand to follow |
| `--swap-hands` | off | POV mode: use if it follows the wrong hand |
| `--fov DEGREES` | `70` | the camera's horizontal field of view (head distance and POV hand distance) |
| `--no-face` | off | turn off head pose and eye tracking |
| `--no-emotion` | off | turn off emotion and face recognition |
| `--emotion-model hsemotion` / `opencv` | `hsemotion` | emotion model: HSEmotion (much better at fear) or the first OpenCV model |
| `--name NAME` | `me` | name stored when you enroll your face (`e`); letters, digits, `-` and `_` |
| `--faces-dir`, `--model-dir` | `faces`, `models` | where enrolled faces and downloaded models are kept |
| `--gaze-dwell SECONDS` | `0` (off) | also lock a target by looking at one spot this long |
| `--no-objects` | off | turn off object detection and grasp tracking |
| `--object-model medium` / `small` | `medium` | object detector: medium is more accurate, small is 2 to 3 times faster |
| `--max-grip-n NEWTONS` | `50` | the robot hand's strongest grip; grip commands are the force divided by this |
| `--window-width PIXELS` | `1600` | maximum window width |
| `--record FILE.csv` | off | save every update to a CSV file (see [Recording](#recording-csv-columns)) |

`python scripts/live.py --help` prints the same list.

## Modes: standing, desk and head camera (POV)

**Standing (`--torso auto`, default).** Stand 1.5 to 2.5 m from the camera, facing it, with the tracked arm
and torso in view, hips included. Torso "up" comes from shoulders and hips.

**Desk (`--torso desk`).** No hips needed. Shoulders and the tracked arm must be in view. Sit upright, let the
tracked upper arm hang straight down beside your torso with the elbow bent 90 degrees, forearm forward and thumb
up (both arms like that when both are tracked), and press `z`. That pose becomes zero, and the calibration removes
the camera's tilt (a laptop screen leaned back 25 degrees would otherwise add 25 degrees of shoulder flexion).
Press `z` again whenever you move the camera or your chair. If the upper arm is not hanging straight down when you
press `z`, every flexion reading is offset by that amount. Before calibrating, the camera's vertical is used, so
keep the camera level.

**POV (`--mode pov`).** For a camera on your head or goggles. Your shoulders are not visible from there, so
the arm joint angles cannot be measured; instead it tracks:
- hand position relative to the camera in cm (right, down, away), from the hand landmarks and the camera's
  field of view (`--fov`; laptop webcams are about 65 to 78, wide action cameras 110 or more),
- hand motion and rotation relative to a zero pose you set with `z` (dx, dy, dz; twist, pitch, yaw),
- flexion at every finger joint (MCP, PIP, DIP) and the gesture,
- what the hand holds and the grip force for it (see Grasp tracking); a head camera sees held objects up close.

That relative hand motion is what a robot's end effector would follow during teleoperation. Two limits:
depth assumes an average-size hand (a bigger hand reads farther away), and because the camera moves with
your head, head motion looks like hand motion unless the goggles also report head orientation (an IMU).
MediaPipe's left/right label assumes a mirrored image; if POV mode follows the wrong hand, add `--swap-hands`.

## Arms: joint angles

Both arms are tracked by default (`--side left` or `--side right` for one); each arm has its own solver and
hand, the 3D panel draws them cyan (right) and pink (left), and a recording gets `L_` and `R_` columns. Both
arms share one torso frame, so the shoulders must be in view.

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

## Fingers

Press `f` to show the finger joint table (in POV mode it starts on). Angles are measured in the palm's own
frame, so they do not change when the whole hand moves, and a left hand is mirrored so the signs match.

| Digit | Angles (deg) |
| --- | --- |
| Index, middle, ring, pinky | `mcp_flex` knuckle flexion (toward the palm +), `mcp_abd` knuckle side bend (toward the thumb +), `pip`, `dip` |
| Thumb | `cmc_palmar` (out of the palm plane), `cmc_radial` (away from the index), `cmc_rotation` (opposition +; zero = bending across the palm toward the pinky), `mcp`, `ip` |

The 3D panels draw the hand rebuilt from these solved angles (in POV mode over the raw landmarks in grey), so
any mismatch between the model and the measurement is visible. The thumb's rotation is only measurable while
the thumb is bent; while it is straight the last value is held. Recordings get 21 finger columns per hand.

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

A pinch also locks or releases a gaze target (see [Eyes](#eyes-gaze-and-target-locking)), and the hand's shape
used for grasps (open, pinch, tripod, wrap, fist) is separate from the gestures (see Grasp tracking).

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

## Eyes: gaze and target locking

The face mesh model also finds the center of each iris. With the head pose, that gives where you are looking:
1. The eye corners and the iris centers are placed in 3D (the same depth model as the head pose) and turned
   into the head's own frame, which removes head rotation and perspective.
2. How far each iris sits from the middle of its eye corners gives the eye's rotation in the head, assuming
   a 12 mm eyeball radius, averaged over both eyes. A blink (lid gap under 13% of the eye width) is skipped
   and the last gaze point is kept.
3. The gaze ray, from between the eyes, is followed until it meets the plane of the camera, which is the
   screen the webcam sits on.

The readout shows `eyes yaw pitch` (degrees, the same directions as the head: yaw + toward your left, pitch +
up), and the 3D panel draws the gaze as a cyan line from the head.

**Calibrate (key `g`).** The video then shows only red dots, one at a time, 9 in all (about 20 seconds). Keep your
head still and look at each dot until it moves; a ring closes in on each dot to draw your eyes to it before
recording starts. An affine map (6 numbers) from the camera plane to the video is fitted, and then the video shows
a cyan circle where you are looking. Because the map is built on the 3D gaze point and not on raw iris positions,
it keeps working when you move or turn your head a little. When it finishes, the readout gives its accuracy,
measured by leaving each dot out of the fit in turn and predicting it (`gaze calibrated: accuracy about N deg (M
px), from 9 dots`). If your eyes were found on fewer than 6 dots, or your gaze hardly moved between them, it says
so and asks you to try again. Calibration is not saved: do it again after moving the window, the camera or your
seat.

**Target locking.** Look at something in the video and press `t` or pinch (thumb and index finger together,
either hand). A red box and crosshair mark the locked target; `t` or another pinch releases it. With
`--gaze-dwell 1.0`, looking at one spot for 1 second also locks it (an arc around the cursor fills up while
you hold still). The locked target is given in the camera's own pixels (the video is shown mirrored, the
recording is not) and as a unit ray from the camera, `target_ray_x/y/z` (x right, y down, z away), which is
what a robot with a camera next to this one would turn toward. Locking needs a calibration first.

**Accuracy, honestly.** Webcam eye tracking is coarse: the iris is a few pixels across, so expect a few
degrees after calibration (about a thumb's width at arm's length), more when you are far from the camera or
the light is dim. In simulation (synthetic faces, landmark noise of 0.25, 0.5 and 1 pixel) the calibration's
left-out error was 0.4, 0.8 and 1.5 degrees, and the frame-to-frame jitter of the smoothed cursor 0.3, 0.8
and 1.9 degrees. A real webcam is noisier than that; your own calibration result is the number to go by.
Without calibration, the eye angles alone are approximate (each person's eyes differ slightly from the model),
and the absolute direction is only as good as the head pose. Glasses with strong reflections and narrow eyes
can make the iris harder to find.

Recorded as `eyes_open`, `eye_yaw`, `eye_pitch`, `gaze_x_px`, `gaze_y_px` (camera pixels), `target_locked`,
`target_x_px`, `target_y_px` and `target_ray_x`, `target_ray_y`, `target_ray_z`.

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

## Grasp tracking: objects, what each hand holds, and the grip force

> These grip forces are targets for the planned arm, worked out from physics, not measured on a hand. Check them
> with a force sensor once the hand is built and edit `vtrack/grip.py`.

**Objects.** Sixteen objects are recognized: tennis ball, ping pong ball, soccer ball, basketball, American
football, tennis racket, ping pong paddle, cell phone, pen, pencil, ruler, eraser, knife, stuffed animal, and
water bottles (metal or plastic). The detector is YOLO-World v2 (medium size), which finds objects named in
plain English: each object is a few phrasings ("teddy bear", "stuffed animal") turned into CLIP text
embeddings, so the list is data, not part of the model (`vtrack/data/object_prompts.json`). A few extra
"helper" classes (hand, person, cup, mouse, remote...) are detected only so that they are not mistaken for one
of the sixteen.

It runs in its own low-priority thread, like the face models, and alternates: the whole frame every 0.6
seconds (to find objects anywhere), and in between only a zoomed square around each hand, where a held object
is seen large enough to recognize. Each object is tracked from run to run with a running score for every
class, so its label is a vote over many frames rather than one guess, and a held object moves along with the
hand between detections (the hand hides much of it). The video shows a box and label for each object; a held
one is drawn in its hand's color with the grip to use.

**Grasps.** A hand holds an object when its thumb tip and at least one other finger touch the object's box
for a quarter of a second (or the palm, for objects bigger than the hand). It lets go when the hand opens,
when the object is seen away from the hand, or when the hand is out of view for more than 0.6 seconds. The
readout shows each hand (`grasp right: holding tennis ball | spherical grip 1.0 N (0.02)`), the hand's
shape (open, pinch, tripod, wrap or fist) and the objects in view. The camera sees no depth at the contact, so a
hand passing in front of an object can look like a touch; the time threshold and the hand's shape (an open hand
never holds anything) filter most of that out.

**Grip force.** For each object, `vtrack/grip.py` works out the squeeze that holds it with a safety margin:

```
F_slip   = m (g + a) / (n mu)        friction on the opposing contacts carries the weight while moving
F_tip    = m (g + a) lever / span    a long object held off center (racket, knife) tips about the grip
F_target = safety (F_slip + F_tip),  at least 1 N, at most what the object tolerates
command  = F_target / 50 N           (0..1; set the hand's real strongest grip with --max-grip-n)
```

`m` is the heaviest typical mass, `a` = 3 m/s² of arm motion, `n` = 2 for fingertip grasps and 3 for
whole-hand wraps, `mu` the friction between rubber-like fingertip pads and the object (conservative: silicone
pads on smooth plastic measured 0.3 to 1.7, lower at higher squeeze, Leddy and Dollar 2020), and safety = 2
(3 for knives). People hold still objects with 40 to 50% of their grip force above the slip force, about 1.7 to
2 times it (Hiramatsu et al. 2015, PLOS ONE), so 2 is not generous for a robot with less touch feedback.

| Object | Grasp | Mass used | Friction | Grip force | Command (of 50 N) | Do not exceed | Open hand to |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tennis ball | spherical | 59 g | 0.6 | 1.0 N | 0.02 | - | 9 cm |
| ping pong ball | fingertip pinch | 3 g | 0.4 | 1.0 N | 0.02 | 20 N | 6 cm |
| soccer ball | spherical | 450 g | 0.6 | 6.4 N | 0.13 | - | 24 cm |
| basketball | spherical | 650 g | 0.7 | 7.9 N | 0.16 | - | 26 cm |
| american football | power wrap | 430 g | 0.7 | 5.2 N | 0.10 | - | 19 cm |
| tennis racket | power wrap | 340 g | 0.7 | 32.5 N | 0.65 | - | 5 cm |
| ping pong paddle | power wrap | 200 g | 0.5 | 10.2 N | 0.20 | - | 5 cm |
| cell phone | power wrap | 240 g | 0.4 | 5.1 N | 0.10 | 30 N | 10 cm |
| pen | tripod (pen grip) | 20 g | 0.4 | 1.0 N | 0.02 | 30 N | 3 cm |
| pencil | tripod (pen grip) | 8 g | 0.5 | 1.0 N | 0.02 | 40 N | 3 cm |
| ruler | lateral pinch | 70 g | 0.4 | 4.0 N | 0.08 | 30 N | 2 cm |
| eraser | fingertip pinch | 30 g | 0.8 | 1.0 N | 0.02 | 15 N | 4 cm |
| knife | power wrap | 250 g | 0.5 | 11.2 N | 0.22 | - | 4 cm |
| stuffed animal | power wrap | 500 g | 0.6 | 7.1 N | 0.14 | 15 N | 17 cm |
| plastic water bottle (500 ml, full) | power wrap | 550 g | 0.5 | 9.4 N | 0.19 | 15 N | 8 cm |
| metal water bottle (insulated, full) | power wrap | 1.2 kg | 0.4 | 25.6 N | 0.51 | - | 10 cm |
| water bottle, material unsure | power wrap | 1.2 kg | 0.4 | 15.0 N | 0.30 | 15 N | 10 cm |

Masses for the balls come from their rules (ITF, ITTF, IFAB Laws of the Game, FIBA, NFL); the rest are typical
retail values, so weigh yours. Notes worth knowing: the tennis racket needs the most because it is held level by
the end of its handle (held head up it needs about 4 N); a ping pong ball weighs almost nothing, so the 1 N floor
sets its force, and the ITTF hardness test presses it with 50 N, so 20 N is a wide margin; a phone's side buttons
take only a few newtons, so squeeze across its front and back; an open plastic bottle dents and spills when
squeezed; the knife is for holding and moving, by the handle, not for cutting. Every entry, its sources and its
notes are in `vtrack/grip.py`.

**Bottle material.** The detector also scores "metal water bottle" against "plastic water bottle" phrasings.
It leans toward "metal" for every bottle, so the bottle counts as metal only when that score is far ahead (on
labeled photos insulated bottles were 5 of 6 metal, 1 unsure; ordinary water bottles 33 plastic, 41 unsure, 8
metal). When unsure, the grip is the metal bottle's force held to the plastic bottle's limit (15 N): a full
metal bottle could then slip if moved fast, which is the safer mistake.

**Accuracy.** Measured on real photos with labeled objects (LVIS v1 validation set, which labels everyday
objects in COCO photos; objects at least 24 pixels across). "Found" counts an object when a box with the right
label overlaps it (intersection over union 0.5 or more); "false" is how often the label appears in photos
verified not to contain it; "in a hand" crops a square around a hand holding the object (from COCO's wrist
keypoints, crops at least 150 pixels wide), like the app's zoomed search, and checks the top label on it.

| Object | Objects | Found: medium | Found: small | False per photo (medium) | In a hand: medium | In a hand: small |
| --- | --- | --- | --- | --- | --- | --- |
| tennis ball | 34 | 74% | 65% | 0.00 | 67% of 15 | 53% |
| soccer ball | 57 | 95% | 82% | 0.05 | 1 of 1 | 1 of 1 |
| basketball | 7 | 43% | 14% | 0.06 | - | - |
| american football | 7 | 29% | 43% | 0.00 | - | - |
| tennis racket | 186 | 83% | 74% | 0.00 | 75% of 8 | 50% |
| cell phone | 133 | 83% | 77% | 0.03 | 95% of 41 | 85% |
| pen | 79 | 42% | 13% | 0.00 | 3 of 5 | 1 of 5 |
| pencil | 37 | 43% | 27% | 0.03 | 0 of 1 | 0 of 1 |
| eraser | 7 | 0% | 0% | 0.03 | - | - |
| knife | 162 | 51% | 32% | 0.00 | 46% of 39 | 18% |
| stuffed animal (teddy bear) | 188 | 79% | 70% | 0.11 | 77% of 26 | 85% |
| water bottle | 101 | 81% | 78% | 0.33 | 73% of 11 | 82% |
| insulated bottle | 7 | 86% | 71% | 0.18 | 1 of 1 | 1 of 1 |
| **average** | | **61%** | **50%** | **0.06** | **72% of 148** | **59% of 148** |

What this means: phones, rackets, balls, stuffed animals and bottles are found reliably; knives, pens and pencils
about half the time per frame (the vote over many frames helps); erasers in these photos were tiny and never
found; and there were no labeled photos of ping pong balls, ping pong paddles or rulers, so those three are
untested. Ball types can be mixed up: an American football is often called a soccer ball. The phrasings were
chosen on this same data (for example "football" in a phrasing made every ball read as an American football and
was removed), so treat the numbers as optimistic. On a webcam the objects you hold are closer and larger than in
these photos, which helps.

Speed on a slow 2-core test computer: medium 394 ms for the whole frame and 125 ms per hand crop, small 169 and 52
ms. Use `--object-model small` (2 to 3 times faster, about 10 to 13 points less accurate) if the status line shows
the objects model below 2 Hz, or `--no-objects` to turn it off.

**Your own objects.** Add a line to `vtrack/data/object_prompts.json` (a name and a few phrasings), run
`python scripts/object_vocab.py` (downloads the CLIP text encoder once, 254 MB, MIT license), and add an
`ObjectSpec` with its mass, size, grasp and friction to `OBJECTS` in `vtrack/grip.py` to give it a grip force.

Recorded per hand (`L_` and `R_` with both arms): `grasp_state` (free, touching, holding), `grasp_object`,
`hand_shape`, `grip_type`, `grip_n`, `grip_cmd`, `grip_open_cm`, and `objects_seen` (label@track id).

The detector weights are Ultralytics' YOLO-World v2 (AGPL-3.0 license), exported to ONNX here with the class
embeddings as an input and stored in half precision (`vtrack/data`; any score changes by 0.03 at most). That is
fine in this private repository; publishing the repository or shipping it in a product would bring in the
AGPL's terms.

## Speed and latency

Everything that could make the hands lag has been taken out of the hand's path:
- **Camera:** a dedicated thread reads the camera continuously and keeps only the newest frame, so a slow
  moment never leaves you looking at old frames. The camera is asked for 60 fps at 1280x720 (MJPG on Windows
  and Linux, which many webcams need for 60 fps) with a one-frame buffer. The status line shows what the
  camera really delivers (`cam N fps`).
- **Models in parallel:** hands, body, face mesh, emotion and objects each run in their own thread on the newest
  frame. The models let threads run at the same time on different processor cores, so the hand model never waits
  for the body, face, emotion or object models. (Measured: two models in parallel took 47 ms per frame against 84
  ms one after the other.)
- **Hands first:** the hand and body models always run at full speed. The face mesh, emotion and object
  models use only the processor cores left over, and on a computer without spare cores they still run at a
  guaranteed 10 Hz (head), 3 Hz (emotion) and 2 Hz (objects). They are also capped at 30, 10 and 10 Hz,
  because faster would not help, and the object model uses at most 4 threads and leaves 2 cores free.
- **Smoothing that does not lag:** One Euro filters smooth hard when you hold still and almost not at all
  when you move fast. Gestures use a 0.2 second majority vote that works the same at any frame rate.

The status line reads like `cam 30 fps | hand 30 Hz 35 ms | pose 30 Hz 40 ms | face 30 Hz 25 ms | emotion 10 Hz 30
ms | objects 8 Hz 90 ms`: rate per model and latency from the moment the camera captured the frame to the moment
the result was ready. On a 2-core test computer (much slower than a typical laptop, with a 30 fps camera) the hand
ran at about 25 Hz alone and 15 to 18 Hz with the body, face and emotion models. Adding the medium object model
brought it to 14 to 15 Hz on that computer; with the small object model and no emotion model it ran at 21 to 23
Hz. With 4 or more cores the background models get cores of their own and no longer slow the hands.

If the hands are not smooth:
1. **Light.** In a dim room most webcams lengthen the exposure and drop to 15 fps; check `cam N fps`. Light
   your hands well.
2. **Fewer or smaller models:** `--object-model small`, then `--no-emotion`, `--no-objects`, `--no-face`, then
   `--pose-model lite`.
3. **Other programs:** close video calls and browser tabs that use the camera or the processor.
4. **Smoothing:** `--smoothing responsive` for the least lag.

## Stability and smoothing

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

## What the camera cannot tell you

- **Elbow nearly straight (under 20 degrees):** shoulder rotation and forearm rotation cannot be separated,
  because the elbow bend plane is undefined. Shoulder rotation is held at its last good value,
  `shoulder_rotation_valid` is false, and the hand's extra twist shows up as forearm rotation. Their sum stays
  correct.
- **Arm held horizontally out to the side (abduction within 5 degrees of 90):** the sagittal and transverse axes
  line up (gimbal lock). Rotation is held and flexion takes the difference, so the rotation matrix is still right
  but the two angles are not individually meaningful there.
- **No hand detected:** wrist angles use the pose's own pinky and index points if available; otherwise
  `wrist_valid` is false.
- **Depth from one camera:** every landmark's distance is estimated, so angles are most reliable when the limb
  moves across the camera's view rather than toward it.
- **Contact in grasps:** the camera sees no depth where the hand meets an object, so a hand passing in front of
  an object can look like a touch; grasps need a quarter second of contact and a hand that is not open.
- **Hidden joints:** MediaPipe reports a guessed position for joints out of view, so the tracker checks their
  visibility and says what is out of view instead of showing made-up angles (except the predicted elbow).

## Recording (CSV columns)

With `--record FILE.csv` a row is written whenever any model delivers a new result, so rows arrive at the
fastest model's rate. `t` is the capture time in seconds. Angles are in degrees, positions as stated; `nan`
means not measured (out of view, no face, nothing held). With both arms tracked (the default), per-side columns
start with `L_` or `R_`; with one arm there is no prefix.

**Body mode** (137 columns with both arms), in order:

| Group | Columns |
| --- | --- |
| Time | `t` |
| Each arm (the `L_` block, then `R_`) | `gesture`, `shoulder_flexion`, `shoulder_abduction`, `shoulder_rotation`, `elbow_flexion`, `forearm_rotation`, `wrist_flexion`, `wrist_deviation`, `shoulder_rotation_valid`, `wrist_valid`, `elbow_predicted` (1 or 0), then that hand's 21 finger angles: `index_mcp_flex`, `index_mcp_abd`, `index_pip`, `index_dip`, the same for `middle`, `ring`, `pinky`, then `thumb_cmc_palmar`, `thumb_cmc_radial`, `thumb_cmc_rotation`, `thumb_mcp`, `thumb_ip` |
| Each leg (`L_`, `R_`) | `hip_flexion`, `hip_abduction`, `hip_rotation`, `pelvic_hike`, `knee_flexion`, `ankle_dorsiflexion`, `ankle_inversion`, `ankle_abduction`, `foot_pitch`, `foot_roll`, `hip_rotation_source` (knee, foot or held) |
| Core | `core_flexion`, `core_lateral_bend`, `core_rotation`, `leg_up_source` (standing calibration or camera vertical) |
| Torso and head | `torso_source` (hips, calibrated or camera vertical), `head_x_cm`, `head_y_cm`, `head_z_cm`, `head_yaw`, `head_pitch`, `head_roll`, `neck_yaw`, `neck_pitch`, `neck_roll` |
| Face | `emotion_angry_pct`, `emotion_disgust_pct`, `emotion_fearful_pct`, `emotion_happy_pct`, `emotion_neutral_pct`, `emotion_sad_pct`, `emotion_surprised_pct`, `identity`, `match`, `emotion_calibrated_for` |
| Eyes and target | `eyes_open`, `eye_yaw`, `eye_pitch`, `gaze_x_px`, `gaze_y_px`, `target_locked`, `target_x_px`, `target_y_px`, `target_ray_x`, `target_ray_y`, `target_ray_z` (pixels are the camera's own, not mirrored) |
| Each hand's grasp | `grasp_state` (free, touching, holding), `grasp_object`, `hand_shape`, `grip_type`, `grip_n`, `grip_cmd`, `grip_open_cm` |
| Objects | `objects_seen` (each tracked object as label@id, separated by `;`) |

**POV mode** (40 columns): `t`, `gesture`, the hand position `x_m`, `y_m`, `z_m` (right, down, away from the
camera), its motion since `z` was pressed `dx_cm`, `dy_cm`, `dz_cm`, `twist`, `pitch`, `yaw`, the 21 finger
angles, the 7 grasp columns, and `objects_seen`.

## For the arm: what is ready and what is not

Ready, from the vision side:
- **Joint angles** for both arms, hands, legs, core and head, in fixed conventions (zero poses and sign tables
  above), smoothed, with flags for values the camera cannot separate.
- **Command mapping** (`vtrack/mapping.py`): `JointChannel` maps an angle range to a 0..1 command (with
  inversion and limits) and `SlewLimiter` caps how fast a command may change, to protect the hardware. It is a
  library for the arm's controller; the live app records angles, not commands.
- **Grip:** for whatever a hand holds, the grasp type, the hand opening, the target squeeze in newtons and a
  0..1 grip command scaled to the hand's strongest grip (`--max-grip-n`).
- **Targets:** a locked gaze target as a ray from the camera, and 16 gestures that can trigger actions.

Not done yet, and needed once the arm exists:
- A pressure calibration curve per air muscle, to turn 0..1 commands into pressure setpoints.
- Matching the arm's joint axes and rotation order (the shoulder is Z-X-Y here) to the physical build.
- Measuring the hand's real strongest grip and checking the grip forces with a force sensor.
- Checking every angle against a goniometer or a leader arm before any closed-loop control.

## Verify on your camera first

None of this has been checked on a live camera yet; do these once on yours.

1. **Axes.** Face the camera, hold the hand still in view, and press `c`. The torso forward vector should have a
   clearly negative z (toward the camera); if not, the world axes are mirrored and every sagittal sign is flipped.
   Then hold the wrist straight (hand in line with the forearm) and check `hand_vs_forearm_deg`: under about 20
   means the hand model and pose model agree on axes. Much larger means the wrist angles from the hand model
   cannot be trusted; use `--pose-only-wrist`. `long_axis_deg` and `palm_normal_deg` compare against the pose
   model's rough finger points and are expected to be larger.
2. **Sign check.** Do each motion in the table and confirm the displayed sign.
3. **Error against ground truth.** Compare against a goniometer or a leader arm before any control.
4. **Head.** Face the camera squarely and check that yaw, pitch and roll read near zero; then turn, nod and tilt
   and confirm the signs in the Head table. If the distance `z` is clearly wrong, set `--fov` for your camera.
5. **Latency.** Watch the status line while you move: the hand should run near the camera rate.
6. **Legs.** Stand in full view and press `z`: the leg and core angles should read near zero. Bend one knee, lift a
   thigh forward and out, and turn a foot out, and confirm the signs in the Legs table.
7. **Emotion.** Press `k` and follow the prompts; the held-out accuracy afterwards tells you how well the
   tracker reads your face in your lighting.
8. **Eyes.** Press `g` and follow the dots, then look at a few things in the video: the cyan circle should
   follow your eyes within the accuracy the calibration reported. Lock one with `t` and check the red box.
9. **Grasps.** Hold each object up to the camera, then pick it up: the box should get the right label within a
   second or two, and the readout should say which hand holds it. Note which objects your camera misses.

Expect the largest errors in shoulder rotation and forearm rotation: they come from small differences between
noisy landmarks, and a few centimeters of elbow or wrist error is several degrees over a 25 to 30 cm segment.

## Project layout

| Module | Purpose |
| --- | --- |
| `vtrack/app.py` | The live application: runs the models in threads, combines their results, draws the screen, records CSV |
| `vtrack/pipeline.py` | Low-latency capture and model threads (newest frame only), CPU priority for hands, rate and latency meters |
| `vtrack/tracker.py` | MediaPipe Tasks wrappers (hand, pose, face mesh; VIDEO mode), model download, hand-to-arm matching by position |
| `vtrack/head.py` | Head pose with 6 DOF (position and yaw, pitch, roll) from 4 anchor points on the face |
| `vtrack/gaze.py` | Eye tracking from the iris landmarks, 9-dot gaze calibration onto the video, target locking |
| `vtrack/face.py` | Face detection, emotion (six cardinal emotions plus neutral, in percent) and face recognition |
| `vtrack/emotion_cal.py` | Per-person emotion calibration: guided capture, fitted correction, held-out accuracy |
| `vtrack/objects.py` | Object detection from plain-English names (YOLO-World, ONNX), zoomed search around the hands |
| `vtrack/grasp.py` | Object tracking with a label vote, which hand holds what, the hand's shape |
| `vtrack/grip.py` | Grip force per object from physics (weight, friction, tipping, crush limit), every value sourced |
| `vtrack/clip_text.py` | CLIP text embeddings, only needed to change the list of objects |
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
| `scripts/object_vocab.py` | Rebuilds the object list after editing `vtrack/data/object_prompts.json` |
| `scripts/eval_stability.py` | Jitter and lag of each smoothing preset on simulated noisy landmarks |

| Folder or file | Contents |
| --- | --- |
| `vtrack/data/` | the object detector (graph and half-precision weights, split into files under 50 MB), `object_prompts.json` (the objects and their phrasings) and `object_vocab.npz` (their text embeddings) |
| `tests/` | 215 automated tests (see [Tests](#tests)) |
| `models/` | downloaded and unpacked models (created on first run; not in git) |
| `faces/` | enrolled faces and emotion calibrations (created when you enroll; not in git) |
| `requirements.txt` | numpy, mediapipe, opencv-python, onnxruntime, certifi, pytest |

## Models

| Model | Used for | Size | Source |
| --- | --- | --- | --- |
| MediaPipe hand landmarker | hands, fingers, gestures | 8 MB | Google MediaPipe |
| MediaPipe pose landmarker (lite / full / heavy) | arms, legs, core | 6 / 9 MB / larger | Google MediaPipe |
| MediaPipe face landmarker | head pose, eyes (iris) | 4 MB | Google MediaPipe |
| YuNet face detector | the face box for emotion and recognition | 0.2 MB | OpenCV Zoo |
| HSEmotion EfficientNet-B0 (`enet_b0_8_best_vgaf`) | emotion (default) | 16 MB | av-savchenko/face-emotion-recognition (Apache 2.0 code) |
| MobileFaceNet expression model | emotion with `--emotion-model opencv` | 5 MB | OpenCV Zoo |
| SFace | face recognition (downloaded on first enroll) | 38 MB | OpenCV Zoo |
| YOLO-World v2 medium / small | objects | 113 / 51 MB unpacked (in the repository: 53 / 24 MB) | Ultralytics (AGPL-3.0), exported here |
| CLIP ViT-B/32 text encoder | only to change the object list | 254 MB | Xenova/clip-vit-base-patch32 on Hugging Face (MIT) |

Everything runs on the processor; no graphics card is needed. Downloads go to `--model-dir` (default
`models/`). The YOLO-World weights are under the AGPL-3.0 license: fine in this private repository, but
publishing the repository or shipping it in a product would bring in the AGPL's terms.

## Tests

```
python -m pytest            # all 215 tests, about a minute, no camera needed
python -m pytest -k grasp   # just one area
```

All 215 pass on Python 3.9 and 3.13. Covered: kinematics round trips for both arms and both legs and the core
(exact for random poses, headings and pelvic hikes), desk and standing calibration with a tilted camera, PnP hand
position recovery, all gestures in random orientations for both hands, head pose round trips, eye angles and gaze
points (exact for random head poses), gaze calibration that survives head motion, target locking, emotion
alignment against an independent least-squares solution, the emotion calibration fit (gradient checked, recovers
known corrections), object detector decoding with known boxes in the whole frame and in zoomed regions, unpacking
the stored models (checksum, sizes), the grip force formula and its limits, object tracking (label vote, aging,
carrying), the grasp state machine (touch, hold, carry, release, hidden object), the threaded pipeline, and the
whole app with simulated models, including a full emotion calibration, a full gaze calibration followed by locking
a target with a key and with pinches, and holding an object in body and POV modes. The app was also run headless
with the real models on photos (body, both arms, hidden elbows, a full-body yoga pose, POV with thumbs up, peace,
pointing and open hands; the eyes of a portrait looking into the camera read yaw 0 and pitch 5 degrees, and a gaze
calibration on a still photo was correctly refused; a woman holding up a phone read as her left hand holding a
cell phone, grip 5.1 N, and in POV mode a phone held in both hands read as held). The object detector was measured
on labeled photos (see Grasp tracking). Not tested here: accuracy with a real camera on your setup. Use the checks
above.

## Troubleshooting

- **"the camera did not deliver any frames" or a black window.** Another program may be using the camera
  (close video calls); try `--camera 1`; on a Mac allow camera access for Terminal in System Settings > Privacy
  & Security > Camera, then run again.
- **Choppy hands or a low `cam N fps`.** Light the room well (dim light drops most webcams to 15 fps), then see
  [Speed and latency](#speed-and-latency).
- **A download fails on first run** (often a certificate error with Python from python.org on a Mac). Run
  `pip install certifi` (it is in `requirements.txt`) or run "Install Certificates.command" from the Python
  folder in Applications, then try again. Files that finished downloading are kept in `models/`.
- **POV mode follows the wrong hand.** Add `--swap-hands`.
- **The 3D body looks wrong while standing.** Stand upright with your whole body in view and press `z`.
- **Arm angles are offset at a desk.** Use `--torso desk` and press `z` in the neutral pose (see Modes).
- **The head's distance is wrong.** Set `--fov` to your camera's horizontal field of view.
- **An object is not recognized.** Hold it closer and in good light, keep it in view for a second or two, and try
  `--object-model medium` if you switched to small; see the accuracy table in Grasp tracking for what to expect,
  and add phrasings in `vtrack/data/object_prompts.json` (then `python scripts/object_vocab.py`).
- **Emotion reads wrong for your face.** Press `k` to calibrate it to your face.

## Other conventions

- Angles use MediaPipe world landmarks (meters), not image coordinates.
- `--side` is the person's own arm. In body mode a hand is matched to that arm by image position, not by
  MediaPipe's left/right label.
- The shoulder angle order (Z-X-Y) should match the physical Homunculus shoulder once the arm is built and its
  axes are decided.
- Units: degrees for angles, centimeters for head and hand motion, meters for POV hand position, newtons for grip
  force, camera pixels (not mirrored) for gaze and targets.
